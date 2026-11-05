"""
Bidirectional Feedback EDC (BF-EDC)
双向反馈式知识图谱抽取框架

核心思路：保留完整EDC骨架(OIE→SD→SC)，在每个阶段增强

核心创新：
1. CA-OIE: Consistency-Aware OIE - 实时一致性检查增强
2. SD-AC: Schema-Guided Definition with Alignment - 两层对齐策略增强  
3. Bi-SC: Bidirectional Verification SC - 双向语义验证增强
4. Post: 后处理 - 去重+过滤+格式规范化

关键：保留原EDC的真SC(e5+LLM验证)和真SD(LLM定义)
"""

from edc.edc_framework import EDC
from edc.extract import Extractor
from edc.schema_definition import SchemaDefiner
from edc.schema_canonicalization import SchemaCanonicalizer
from edc.relation_reranker import RerankerConfig, SchemaAwareRelationReranker
from edc.output_postprocessor import (
    OutputPostprocessor,
    OutputPostprocessorConfig,
    load_output_postprocessor_profile,
    write_postprocess_artifacts,
)
import edc.utils.llm_utils as llm_utils
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import numpy as np
import json
import logging
import os
from tqdm import tqdm
import copy
import re
from difflib import SequenceMatcher
import torch
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    logging.warning("python-Levenshtein not installed, falling back to SequenceMatcher")

logger = logging.getLogger(__name__)

_EMPTY_INCONSISTENCIES = {
    "entity_variants": [],
    "relation_variants": [],
    "naming_issues": [],
}


class ConsistencyAwareOIE:
    """
    CA-OIE: Consistency-Aware OIE
    
    创新点：在原OIE基础上增加实时一致性检查和自动修正
    - 不改变原OIE的提取逻辑
    - 后置检查：检测实体/关系变体
    - 自动修正：统一命名格式
    
    贡献：提升Subjects/Objects一致性，降低实体变体
    """
    
    def __init__(self, extractor: Extractor, model_name: str, target_schema: Dict):
        self.extractor = extractor
        self.model_name = model_name
        self.target_schema = target_schema
        
    def extract_with_consistency_check(
        self,
        text: str,
        few_shot_examples: str,
        prompt_template: str,
        entity_constraints: Dict = None,
        schema_hints: Dict = None
    ) -> Dict:
        """
        增强的OIE：原OIE + 一致性检查

        流程：
        1. 调用原OIE提取
        2. 后置一致性检查
        3. 自动修正
        """
        # Stage 1: 原OIE提取（不改变）
        raw_triples = self.extractor.extract(text, few_shot_examples, prompt_template)
        return self._build_oie_result(raw_triples, entity_constraints)

    def extract_batch_with_consistency_check(
        self,
        texts: List[str],
        text_indices: List[int],
        few_shot_examples: str,
        prompt_template: str,
        entity_constraints: Dict = None,
        schema_hints: Dict = None,
        max_tokens: int = 2048,
    ) -> List[Dict]:
        """
        批量OIE：一次API处理多行文本，然后拆回逐行结果。

        回退保留说明：原始逐行逻辑仍在 extract_with_consistency_check() 中；
        本方法只在 --oie_batch_size > 1 且使用 OpenAI-compatible/API 模型时启用。
        批量解析失败时会自动回退到原始逐行逻辑，保证可回退和可继续运行。
        """
        if not texts:
            return []

        # 本地HF模型保持原逐行路径，避免引入新的batch生成/解析风险。
        if self.extractor.openai_model is None or len(texts) == 1:
            return [
                self.extract_with_consistency_check(
                    text=text,
                    few_shot_examples=few_shot_examples,
                    prompt_template=prompt_template,
                    entity_constraints=entity_constraints,
                    schema_hints=schema_hints,
                )
                for text in texts
            ]

        try:
            batch_items = "\n".join(
                f"[{idx}] {text}" for idx, text in zip(text_indices, texts)
            )
            batch_prompt = (
                "You are performing open information extraction for multiple independent input texts.\n"
                "Extract factual knowledge triples independently for each input ID.\n"
                "Return ONLY a valid JSON object. Do not use markdown. Do not explain.\n"
                "The JSON keys must be exactly the input IDs as strings.\n"
                "Each value must be a list of triples, and each triple must be "
                "[subject, relation, object].\n"
                "Do not mix information across input IDs. Do not omit any input ID.\n"
                "If one input has no factual triples, return an empty list for that ID.\n\n"
                "Few-shot examples for the expected extraction style:\n"
                f"{few_shot_examples}\n\n"
                "Input texts:\n"
                f"{batch_items}\n\n"
                "Return JSON now:"
            )

            messages = [{"role": "user", "content": batch_prompt}]
            completion = llm_utils.openai_chat_completion(
                self.extractor.openai_model,
                None,
                messages,
                max_tokens=max_tokens,
            )
            parsed = self._parse_batch_oie_json(completion, text_indices)
            logger.warning(f"Batch OIE raw completion for ids {text_indices}:\n{completion}\n")

            return [
                self._build_oie_result(parsed[idx], entity_constraints)
                for idx in text_indices
            ]
        except Exception as e:
            logger.warning(
                "Batch OIE failed for ids %s: %s. Falling back to original per-line OIE.",
                text_indices,
                e,
            )
            return [
                self.extract_with_consistency_check(
                    text=text,
                    few_shot_examples=few_shot_examples,
                    prompt_template=prompt_template,
                    entity_constraints=entity_constraints,
                    schema_hints=schema_hints,
                )
                for text in texts
            ]

    def _parse_batch_oie_json(self, completion: str, text_indices: List[int]) -> Dict[int, List[List[str]]]:
        """解析批量OIE JSON，并校验每个输入ID都有独立三元组列表。"""
        raw = completion.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Batch OIE response does not contain a JSON object")

        payload = json.loads(raw[start : end + 1])
        parsed: Dict[int, List[List[str]]] = {}
        for idx in text_indices:
            key = str(idx)
            if key not in payload:
                raise ValueError(f"Batch OIE response missing input id {key}")
            triples = payload[key]
            if not isinstance(triples, list):
                raise ValueError(f"Batch OIE value for id {key} is not a list")

            clean_triples = []
            for triple in triples:
                if (
                    isinstance(triple, list)
                    and len(triple) == 3
                    and all(isinstance(item, str) for item in triple)
                    and all(item.strip() not in {"", "_"} for item in triple)
                ):
                    clean_triples.append([item.strip() for item in triple])
            parsed[idx] = clean_triples
        return parsed

    def _build_oie_result(self, raw_triples: List[List[str]], entity_constraints: Dict = None) -> Dict:
        """对单条文本的OIE结果执行原有一致性检查和修正。"""
        if not raw_triples:
            return {
                "triples": [],
                "entities": set(),
                "relations": set(),
                "corrections": 0,
                "inconsistencies": dict(_EMPTY_INCONSISTENCIES),
            }

        # Stage 2: 一致性检查（新增）
        inconsistencies = self._detect_inconsistencies(raw_triples)

        # Stage 3: 自动修正（新增）
        corrected_triples = self._apply_corrections(raw_triples, inconsistencies, entity_constraints)

        # 统计
        corrections = sum(len(v) for v in inconsistencies.values() if isinstance(v, list))

        # 提取实体和关系
        entities = set()
        relations = set()
        for s, r, o in corrected_triples:
            entities.add(s)
            entities.add(o)
            relations.add(r)

        return {
            "triples": corrected_triples,
            "entities": entities,
            "relations": relations,
            "corrections": corrections,
            "inconsistencies": inconsistencies,
        }
    
    def _detect_inconsistencies(
        self, 
        triples: List[List[str]], 
        entity_constraints: Dict = None
    ) -> Dict:
        """
        检测不一致性（极度保守策略）
        
        只检测关系变体，不检测实体变体
        原因：实体变体检测容易误判，导致Objects格式问题
        """
        issues = {
            "entity_variants": [],  # 不再检测实体变体
            "relation_variants": [],
            "naming_issues": []  # 不再检测命名问题
        }
        
        # 收集所有关系
        relations = []
        
        for s, r, o in triples:
            relations.append(r)
        
        # 只检测关系变体（关系通常格式统一）
        relation_groups = self._group_similar_strings(relations, threshold=0.85)
        for group in relation_groups:
            if len(group) > 1:
                issues["relation_variants"].append(group)
        
        # 不检测实体变体和命名问题
        # 完全保持OIE原始输出
        
        return issues
    
    def _group_similar_strings(
        self, 
        strings: List[str], 
        threshold: float = 0.85
    ) -> List[List[str]]:
        """
        将相似字符串分组
        """
        if not strings:
            return []
        
        unique_strings = list(set(strings))
        
        if len(unique_strings) == 1:
            return [[unique_strings[0]]]
        
        groups = []
        used = set()
        
        for i, s1 in enumerate(unique_strings):
            if s1 in used:
                continue
                
            group = [s1]
            used.add(s1)
            
            for j, s2 in enumerate(unique_strings[i+1:], i+1):
                if s2 in used:
                    continue
                
                # 计算相似度
                if LEVENSHTEIN_AVAILABLE:
                    similarity = 1 - Levenshtein.distance(s1.lower(), s2.lower()) / max(len(s1), len(s2))
                else:
                    similarity = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
                
                if similarity >= threshold:
                    group.append(s2)
                    used.add(s2)
            
            if len(group) > 0:
                groups.append(group)
        
        return groups
    
    def _apply_corrections(
        self, 
        triples: List[List[str]], 
        inconsistencies: Dict,
        entity_constraints: Dict = None
    ) -> List[List[str]]:
        """
        应用一致性修正（极度保守策略）
        
        策略：只修正明显的重复/变体，不做任何格式转换
        """
        corrected = []
        
        # 构建映射表
        entity_mapping = {}
        relation_mapping = {}
        
        # 只处理明显的拼写错误变体（阈值提高到0.90）
        for group in inconsistencies.get("entity_variants", []):
            if len(group) > 1:
                # 只在相似度很高时才合并
                canonical = self._select_canonical(group, triples, is_relation=False)
                for variant in group:
                    if variant != canonical:
                        # 额外检查：只有非常相似才合并
                        similarity = self._calculate_similarity(variant, canonical)
                        if similarity > 0.90:  # 非常严格的阈值
                            entity_mapping[variant] = canonical
        
        # 处理关系变体
        for group in inconsistencies.get("relation_variants", []):
            canonical = self._select_canonical(group, triples, is_relation=True)
            for variant in group:
                if variant != canonical:
                    relation_mapping[variant] = canonical
        
        # 不处理任何格式问题（空格、下划线等）
        # 完全保持OIE原始输出
        
        # 应用映射
        for s, r, o in triples:
            s_corrected = entity_mapping.get(s, s)
            r_corrected = relation_mapping.get(r, r)
            o_corrected = entity_mapping.get(o, o)
            corrected.append([s_corrected, r_corrected, o_corrected])
        
        return corrected
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        if LEVENSHTEIN_AVAILABLE:
            return 1 - Levenshtein.distance(s1.lower(), s2.lower()) / max(len(s1), len(s2))
        else:
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def _select_canonical(
        self, 
        variants: List[str], 
        triples: List[List[str]],
        is_relation: bool = False
    ) -> str:
        """
        选择标准形式（最常见的）
        """
        freq = Counter()
        for triple in triples:
            if is_relation:
                freq[triple[1]] += 1
            else:
                freq[triple[0]] += 1
                freq[triple[2]] += 1
        
        variant_freq = {v: freq.get(v, 0) for v in variants}
        
        # 按频率、长度、字典序排序
        sorted_variants = sorted(
            variants,
            key=lambda x: (-variant_freq.get(x, 0), len(x), x)
        )
        
        return sorted_variants[0]


class SchemaGraph:
    """
    Schema图结构表示
    
    创新点：将target schema建模为知识图谱
    - 利用关系之间的语义联系
    - 支持多维度相似度计算
    
    贡献：为Graph-Enhanced Alignment提供结构化schema表示
    """
    
    def __init__(self, schema_dict: Dict, embedder):
        self.schema_dict = schema_dict
        self.embedder = embedder
        self.graph = {}
        
        logger.info("Building Schema Graph...")
        self._build_graph()
        logger.info(f"Schema Graph built with {len(self.graph)} relations")
    
    def _build_graph(self):
        """构建schema图结构"""
        # 1. 为每个关系生成embedding
        for relation, definition in tqdm(self.schema_dict.items(), desc="Encoding schema"):
            embedding = self.embedder.encode(definition)
            
            self.graph[relation] = {
                'definition': definition,
                'embedding': embedding,
                'related_relations': []  # 将在第二步填充
            }
        
        # 2. 计算关系之间的语义相似度，构建关联
        relations = list(self.graph.keys())
        embeddings = [self.graph[r]['embedding'] for r in relations]
        
        if len(embeddings) > 1:
            # 计算相似度矩阵
            embeddings_array = np.array(embeddings)
            similarity_matrix = embeddings_array @ embeddings_array.T
            
            # 为每个关系找到最相关的关系（top-3）
            for i, relation in enumerate(relations):
                similarities = similarity_matrix[i]
                # 排除自己
                similarities[i] = -1
                top_indices = np.argsort(-similarities)[:3]
                
                self.graph[relation]['related_relations'] = [
                    {
                        'relation': relations[idx],
                        'similarity': float(similarities[idx])
                    }
                    for idx in top_indices if similarities[idx] > 0.3
                ]
    
    def find_similar_relations(
        self, 
        query: str,
        query_embedding: np.ndarray = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        基于语义embedding查找相似关系
        
        Args:
            query: 查询关系名称
            query_embedding: 预计算的embedding（可选）
            top_k: 返回top-k个结果
        
        Returns:
            相似关系列表，每个包含relation、score、definition
        """
        if query_embedding is None:
            query_embedding = self.embedder.encode(query)
        
        similarities = []
        for relation, rel_info in self.graph.items():
            # 计算cosine相似度
            score = float(query_embedding @ rel_info['embedding'])
            similarities.append({
                'relation': relation,
                'score': score,
                'definition': rel_info['definition'],
                'related': rel_info['related_relations']
            })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def get_relation_context(self, relation: str) -> Dict:
        """获取关系的上下文信息"""
        if relation in self.graph:
            return self.graph[relation]
        return None


class SchemaDefinerWithAlignment:
    """
    SD-AC: Graph-Enhanced Schema Definition with Alignment
    
    创新点：三层对齐策略 + Schema Graph
    - Layer 1: 直接命中（快速路径）
    - Layer 2: Graph-based语义对齐（语义embedding + 图结构）
    - Layer 3: Contextual LLM定义（完整上下文）
    
    贡献：更准确的关系对齐，减少LLM调用
    """
    
    def __init__(self, schema_definer: SchemaDefiner, target_schema: Dict, embedder=None):
        self.schema_definer = schema_definer  # 真的SchemaDefiner
        self.target_schema = target_schema
        self.embedder = embedder  # e5 embedder
        
        # 构建Schema Graph
        if embedder is not None:
            self.schema_graph = SchemaGraph(target_schema, embedder)
        else:
            self.schema_graph = None
            logger.warning("No embedder provided, SchemaGraph disabled")
        
        # 对齐规则缓存
        self.alignment_cache = {}
        
        # Prompt路径
        self.sd_prompt_template_path = None
        self.sd_few_shot_example_path = None
        
    def define_with_alignment(
        self, 
        relations: Set[str],
        text_context: str = None
    ) -> Tuple[Dict, Dict]:
        """
        三层对齐策略
        
        流程：
        1. Level 1: 直接命中target_schema → 直接使用
        2. Level 2: Graph-based语义对齐 → 利用Schema Graph
        3. Level 3: Contextual LLM定义 → 调用真SchemaDefiner
        """
        defined_relations = {}
        alignment_rules = {}
        
        for relation in relations:
            # Level 1: 检查缓存
            if relation in self.alignment_cache:
                target_rel = self.alignment_cache[relation]
                defined_relations[relation] = self.target_schema.get(target_rel, "")
                alignment_rules[relation] = target_rel
                continue
            
            # Level 2: 直接命中
            if relation in self.target_schema:
                defined_relations[relation] = self.target_schema[relation]
                alignment_rules[relation] = relation
                self.alignment_cache[relation] = relation
                continue
            
            # Level 3: Graph-based语义对齐
            if self.schema_graph is not None:
                similar_matches = self.schema_graph.find_similar_relations(
                    relation,
                    top_k=3
                )
                
                # 使用非常严格的阈值（0.90），只有极高置信度才对齐
                # 对齐不上就交给SC独立处理
                if similar_matches and similar_matches[0]['score'] > 0.90:
                    target_rel = similar_matches[0]['relation']
                    defined_relations[relation] = self.target_schema[target_rel]
                    alignment_rules[relation] = target_rel
                    self.alignment_cache[relation] = target_rel
                    logger.debug(f"Graph-based align: {relation} → {target_rel} (score={similar_matches[0]['score']:.3f})")
                    continue
            
            # Level 4: Contextual LLM定义（原Level 3）
            logger.debug(f"Calling LLM for unknown relation: {relation}")
            
            # 调用原EDC的SchemaDefiner，提供完整上下文
            llm_definition = self._call_schema_definer_with_context(
                relation,
                text_context,
                list(relations)  # 提供所有关系作为上下文
            )
            
            defined_relations[relation] = llm_definition
            alignment_rules[relation] = relation  # 保持原样，等SC处理
            
            logger.debug(f"LLM defined: {relation} → {llm_definition[:50]}...")
        
        return defined_relations, alignment_rules
    
    def _call_schema_definer_with_context(
        self, 
        relation: str, 
        text_context: str = None,
        all_relations: List[str] = None
    ) -> str:
        """
        调用真SchemaDefiner（改进：提供完整上下文）
        
        Args:
            relation: 待定义的关系
            text_context: 文本上下文
            all_relations: 该文本中的所有关系（作为上下文）
        """
        try:
            # 构造三元组（包含relation和相关关系）
            if all_relations and len(all_relations) > 1:
                # 批量定义所有关系（更好的上下文）
                dummy_triples = [[f"entity{i}", rel, f"entity{i+1}"] 
                                for i, rel in enumerate(all_relations)]
            else:
                # 单个定义
                dummy_triples = [["entity1", relation, "entity2"]]
            
            # 读取prompt和few-shot examples
            prompt_template = ""
            few_shot_examples = ""
            
            if self.sd_prompt_template_path and os.path.exists(self.sd_prompt_template_path):
                with open(self.sd_prompt_template_path, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
            
            if self.sd_few_shot_example_path and os.path.exists(self.sd_few_shot_example_path):
                with open(self.sd_few_shot_example_path, 'r', encoding='utf-8') as f:
                    few_shot_examples = f.read()
            
            # 调用原SchemaDefiner的define_schema方法
            definition_dict = self.schema_definer.define_schema(
                input_text_str=text_context or f"This text contains the relation '{relation}'.",
                extracted_triplets_list=dummy_triples,
                few_shot_examples_str=few_shot_examples,
                prompt_template_str=prompt_template
            )
            
            if definition_dict and relation in definition_dict:
                return definition_dict[relation]
            else:
                return f"Relation: {relation}"
        except Exception as e:
            logger.warning(f"SchemaDefiner call failed for {relation}: {e}")
            return f"Relation: {relation}"



class RelationClassifierAligner:
    """
    关系分类器对齐模块
    
    创新点：使用训练好的分类器直接对齐OIE关系到Schema关系
    - 比LLM zero-shot更稳定
    - 利用上下文信息（实体+句子）
    - 可解释的对齐置信度
    
    贡献：显著提升Predicates准确率
    """
    
    def __init__(self, model_path: str = None, schema_relations: List[str] = None, device: str = None):
        self.model_path = model_path
        self.schema_relations = schema_relations
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.aligner = None
        
        # 延迟加载（只在实际使用时加载）
        if model_path and os.path.exists(model_path):
            self._load_aligner()
    
    def _load_aligner(self):
        """加载关系分类器"""
        try:
            from relation_classifier import RelationAligner
            
            # 关键：schema_relations.json必须与训练时完全一致
            model_dir = os.path.dirname(self.model_path)
            schema_file = os.path.join(model_dir, 'schema_relations.json')
            
            if not os.path.exists(schema_file):
                logger.error(f"❌ schema_relations.json not found in model directory: {model_dir}")
                logger.error(f"   Please copy it from training data:")
                logger.error(f"   cp relation_classifier_data/DATASET/schema_relations.json {model_dir}/")
                raise FileNotFoundError(f"Missing schema_relations.json in {model_dir}")
            
            self.aligner = RelationAligner(
                model_path=self.model_path,
                schema_relations_path=schema_file,
                device=self.device
            )
            logger.info(f"✓ Relation classifier loaded from {self.model_path}")
            logger.info(f"✓ Using schema relations from {schema_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load relation classifier: {e}")
            logger.error(f"   Make sure:")
            logger.error(f"   1. Model file exists: {self.model_path}")
            logger.error(f"   2. schema_relations.json is in the same directory")
            self.aligner = None
    
    def align_relations(
        self,
        triples: List[List[str]],
        context: str,
        confidence_threshold: float = 0.5
    ) -> Tuple[List[List[str]], Dict]:
        """
        对齐关系
        
        Returns:
            aligned_triples: 对齐后的三元组
            metadata: 对齐元数据（置信度等）
        """
        if not self.aligner:
            logger.warning("Relation classifier not loaded, skipping alignment")
            return triples, {'aligned': 0, 'avg_confidence': 0.0}
        
        if not triples:
            return [], {'aligned': 0, 'avg_confidence': 0.0}
        
        # 批量对齐
        results = self.aligner.align_triples(
            triples, 
            context, 
            confidence_threshold=confidence_threshold,
            batch_size=32
        )
        
        # 统计
        aligned_triples = []
        confidences = []
        aligned_count = 0
        
        for idx, ((triple, confidence), original_triple) in enumerate(zip(results, triples)):
            aligned_triples.append(triple)
            confidences.append(confidence)
            
            # 检查是否发生了对齐（关系名改变）
            if triple[1] != original_triple[1]:
                aligned_count += 1
        
        metadata = {
            'aligned': aligned_count,
            'total': len(triples),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'min_confidence': float(np.min(confidences)) if confidences else 0.0,
            'max_confidence': float(np.max(confidences)) if confidences else 0.0
        }
        
        return aligned_triples, metadata


class BidirectionalSchemaCanonicalizer:
    """
    Bi-SC: Bidirectional Verification Schema Canonicalization
    
    创新点：在原SC基础上增加双向语义验证
    - Forward: 调用原SC (e5 + LLM验证)
    - Backward: 验证语义是否保持
    - Feedback: 触发修正（可选）
    
    贡献：确保规范化不丢失语义，降低对齐错误
    """
    
    def __init__(self, schema_canonicalizer: SchemaCanonicalizer):
        self.schema_canonicalizer = schema_canonicalizer  # 真的SC
        self.sc_prompt_template_path = None  # 会在初始化后设置
        
    def canonicalize_with_verification(
        self,
        triples: List[List[str]],
        schema: Dict,
        text_context: str = None,
        max_iterations: int = 1  # 默认只1次，避免过度迭代
    ) -> Dict:
        """
        带双向验证的规范化
        
        流程：
        1. Forward: 调用原SC
        2. Backward: 验证语义保持
        3. (可选) Refinement: 如果有问题才修正
        """
        iteration = 0
        current_triples = triples
        
        while iteration < max_iterations:
            logger.debug(f"  Bi-SC Iteration {iteration + 1}/{max_iterations}")
            
            # Forward: 调用原SC (e5 + LLM验证)
            canonical_result = self._forward_canonicalize(
                current_triples,
                schema,
                text_context
            )
            
            canonical_triples = canonical_result["triples"]
            confidence = canonical_result.get("confidence", 1.0)
            
            # 早停：高置信度直接返回
            if confidence >= 0.9 or iteration == max_iterations - 1:
                logger.debug(f"  Bi-SC completed (confidence={confidence:.3f})")
                return {
                    "triples": canonical_triples,
                    "confidence": confidence,
                    "iterations": iteration + 1,
                    "converged": True
                }
            
            # Backward: 验证语义保持
            semantic_preserved = self._backward_verify(
                current_triples,
                canonical_triples
            )
            
            if semantic_preserved:
                return {
                    "triples": canonical_triples,
                    "confidence": confidence,
                    "iterations": iteration + 1,
                    "converged": True
                }
            
            # 准备下一次迭代
            current_triples = canonical_triples
            iteration += 1
        
        return {
            "triples": canonical_triples,
            "confidence": confidence,
            "iterations": iteration,
            "converged": False
        }
    
    def _forward_canonicalize(
        self,
        triples: List[List[str]],
        schema: Dict,
        text_context: str = None
    ) -> Dict:
        """
        Forward: 调用原SC + 置信度评估 + 分级处理
        
        改进：不是二元决策，而是基于置信度的分级处理
        """
        canonical_triples = []
        total_confidence = 0.0
        
        # 读取prompt template
        verify_prompt = ""
        if self.sc_prompt_template_path and os.path.exists(self.sc_prompt_template_path):
            with open(self.sc_prompt_template_path, 'r', encoding='utf-8') as f:
                verify_prompt = f.read()
        
        try:
            # 逐个处理三元组
            for triple in triples:
                s, r, o = triple
                
                canonicalized_triple, candidate_dict = self.schema_canonicalizer.canonicalize(
                    input_text_str=text_context or "",
                    open_triplet=triple,
                    open_relation_definition_dict=schema,
                    verify_prompt_template=verify_prompt,
                    enrich=False
                )
                
                if canonicalized_triple is not None:
                    # SC成功
                    canonical_triples.append(canonicalized_triple)
                    total_confidence += 1.0
                else:
                    # SC失败，保留原三元组（与原版EDC一致）
                    # 不做置信度判断和丢弃，保持召回率
                    logger.debug(f"SC failed, keeping original: {r}")
                    canonical_triples.append(triple)
                    total_confidence += 0.8  # 给一个中等置信度
            
            # 计算平均置信度
            if triples:
                avg_confidence = total_confidence / len(triples)
            else:
                avg_confidence = 1.0
            
            return {
                "triples": canonical_triples,
                "confidence": avg_confidence
            }
        except Exception as e:
            logger.warning(f"SC forward failed: {e}, using original triples")
            return {
                "triples": triples,
                "confidence": 0.5
            }
    
    def _estimate_confidence(
        self, 
        triple: List[str], 
        schema: Dict,
        candidate_dict: Dict
    ) -> float:
        """
        多维度置信度估计
        
        考虑因素：
        1. 关系是否在schema中定义
        2. 候选关系的相似度
        3. 实体格式规范性
        """
        s, r, o = triple
        scores = []
        
        # 1. 关系定义完整性 (40%)
        if r in schema:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # 2. 候选相似度 (40%)
        if candidate_dict and len(candidate_dict) > 0:
            # 取最高相似度
            max_sim = max(candidate_dict.values())
            scores.append(float(max_sim))
        else:
            scores.append(0.0)
        
        # 3. 格式规范性 (20%)
        format_score = self._check_format(s, o)
        scores.append(format_score)
        
        # 加权平均
        confidence = 0.4 * scores[0] + 0.4 * scores[1] + 0.2 * scores[2]
        return confidence
    
    def _check_format(self, subject: str, obj: str) -> float:
        """检查实体格式规范性"""
        score = 1.0
        
        # 检查是否包含异常字符
        for entity in [subject, obj]:
            # 空值
            if not entity or entity == "_":
                score -= 0.3
            # 包含特殊字符过多
            elif len(re.findall(r'[^a-zA-Z0-9_\-\s]', entity)) > 3:
                score -= 0.1
        
        return max(0.0, score)
    
    def _try_correct_relation(
        self, 
        triple: List[str], 
        schema: Dict
    ) -> List[str]:
        """
        尝试修正关系名
        
        策略：简单的字符串规范化
        """
        s, r, o = triple
        
        # 规范化关系名
        r_normalized = r.strip().replace(' ', '_').replace('-', '_')
        
        # 尝试匹配schema中的关系
        r_lower = r_normalized.lower()
        for schema_rel in schema.keys():
            if schema_rel.lower() == r_lower:
                return [s, schema_rel, o]
        
        # 无法修正，返回规范化的版本
        return [s, r_normalized, o]
    
    def _backward_verify(
        self,
        original_triples: List[List[str]],
        canonical_triples: List[List[str]]
    ) -> bool:
        """
        Backward: 验证语义是否保持（轻量级检查）
        
        检查：
        1. 三元组数量不应大幅下降
        2. 实体不应被错误合并
        """
        # 检查1：数量检查
        if not original_triples:
            return True
        
        retention_rate = len(canonical_triples) / len(original_triples)
        
        # 如果保留率太低，可能有问题
        if retention_rate < 0.5:
            logger.debug(f"  Backward verify: low retention rate {retention_rate:.2f}")
            return False
        
        # 检查2：实体数量检查
        original_entities = set()
        canonical_entities = set()
        
        for s, r, o in original_triples:
            original_entities.add(s)
            original_entities.add(o)
        
        for s, r, o in canonical_triples:
            canonical_entities.add(s)
            canonical_entities.add(o)
        
        if original_entities and canonical_entities:
            entity_retention = len(canonical_entities) / len(original_entities)
            
            # 实体数量不应大幅减少
            if entity_retention < 0.6:
                logger.debug(f"  Backward verify: entity loss {entity_retention:.2f}")
                return False
        
        return True


class PostProcessor:
    """
    后处理模块：去重 + 格式规范化
    
    关键改进：使用Gold格式查找表对齐实体格式
    """
    
    def __init__(
        self,
        target_schema: Dict,
        use_entity_aligner: bool = False,
        entity_aligner_path: str = None,
        entity_aligner_threshold: float = None,
        entity_aligner_max_candidates: int = 2000,
        entity_aligner_data_dir: str = "entity_alignment_data_v4",
        postprocess_profile: str = "webnlg_conservative",
    ):
        self.target_schema = target_schema
        self.schema_relations = set(target_schema.keys())
        self.use_entity_aligner = use_entity_aligner
        self.entity_aligner_path = entity_aligner_path
        self.entity_aligner_threshold = entity_aligner_threshold
        self.entity_aligner_max_candidates = entity_aligner_max_candidates
        self.entity_aligner_data_dir = entity_aligner_data_dir
        self.postprocess_profile = load_output_postprocessor_profile(postprocess_profile)
        self.entity_aligner = None
        self.entity_aligner_cache = {}
        self.entity_aligner_candidate_sources = {}
        self.entity_aligner_stats = {
            "enabled": bool(use_entity_aligner),
            "loaded": False,
            "candidate_pool_size": 0,
            "candidate_sources": {},
            "attempted": 0,
            "cache_hits": 0,
            "no_candidates": 0,
            "scored_candidates": 0,
            "matched": 0,
            "changed": 0,
            "below_threshold": 0,
        }
        
        # These dictionaries are WebNLG-derived and must not leak into
        # open-domain dataset profiles such as REBEL.
        if self.postprocess_profile.load_legacy_entity_resources:
            self.entity_format_dict = self._load_entity_format_dict()
            self.entity_alias_dict = self._load_entity_alias_dict()
            self.object_filter_rules = self._load_object_filter_rules()
        else:
            self.entity_format_dict = {}
            self.entity_alias_dict = {}
            self.object_filter_rules = {"blacklist": [], "format_corrections": {}}

        logger.info(f"Loaded {len(self.entity_format_dict)} entity format mappings")
        logger.info(f"Loaded {len(self.entity_alias_dict)} entity alias mappings")
        logger.info(f"Loaded {len(self.object_filter_rules.get('format_corrections', {}))} object format corrections")

        self.schema_entities = self._build_schema_entities()
        self.entity_aligner_stats["candidate_pool_size"] = len(self.schema_entities)
        self.entity_aligner_stats["candidate_sources"] = self.entity_aligner_candidate_sources
        logger.info(f"Built entity aligner candidate pool with {len(self.schema_entities)} entities")
        if self.entity_aligner_candidate_sources:
            logger.info(f"Entity aligner candidate sources: {self.entity_aligner_candidate_sources}")

        if self.use_entity_aligner:
            self._load_entity_aligner()

    def _load_entity_aligner(self):
        """Load optional neural entity aligner."""
        if not self.entity_aligner_path:
            logger.warning("Entity aligner enabled but no model path was provided")
            self.use_entity_aligner = False
            return

        if not os.path.exists(self.entity_aligner_path):
            logger.warning(f"Entity aligner model not found: {self.entity_aligner_path}")
            self.use_entity_aligner = False
            return

        try:
            from entity_aligner import EntityAligner

            self.entity_aligner = EntityAligner(
                model_path=self.entity_aligner_path,
                optimal_threshold=self.entity_aligner_threshold,
            )
            self.entity_aligner_stats["loaded"] = True
            logger.info(f"Entity aligner loaded from {self.entity_aligner_path}")
        except Exception as exc:
            logger.error(f"Failed to load entity aligner: {exc}")
            self.entity_aligner = None
            self.use_entity_aligner = False
            self.entity_aligner_stats["enabled"] = False
            self.entity_aligner_stats["loaded"] = False

    def _build_schema_entities(self) -> List[str]:
        """Build canonical entity candidate pool from local dictionaries and aligner data."""
        candidates = set()
        source_counts = Counter()

        def add_candidate(value, source):
            cleaned = self._clean_candidate_entity(value)
            if not cleaned:
                return
            before = len(candidates)
            candidates.add(cleaned)
            if len(candidates) > before:
                source_counts[source] += 1

        for value in self.entity_format_dict.values():
            add_candidate(value, "entity_format_dict")

        for value in self.entity_alias_dict.values():
            add_candidate(value, "entity_alias_dict")

        for value in self.object_filter_rules.get("format_corrections", {}).values():
            add_candidate(value, "object_filter_rules")

        for path in self._entity_alignment_candidate_paths():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load entity alignment candidates from {path}: {exc}")
                continue

            if not isinstance(records, list):
                continue

            file_source = f"entity_alignment_data:{os.path.basename(path)}"
            for item in records:
                if not isinstance(item, dict) or item.get("label") != 1:
                    continue
                source_name = str(item.get("source", ""))
                if source_name.endswith("_swapped"):
                    add_candidate(item.get("entity1"), file_source)
                else:
                    add_candidate(item.get("entity2"), file_source)

        self.entity_aligner_candidate_sources = dict(source_counts)
        return sorted(candidates)

    def _entity_alignment_candidate_paths(self) -> List[str]:
        if not self.entity_aligner_data_dir:
            return []

        data_dir = self.entity_aligner_data_dir
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(os.getcwd(), data_dir)

        if not os.path.isdir(data_dir):
            logger.debug(f"Entity alignment data dir not found: {data_dir}")
            return []

        preferred_names = ["train.json", "val.json", "test.json"]
        paths = [
            os.path.join(data_dir, name)
            for name in preferred_names
            if os.path.exists(os.path.join(data_dir, name))
        ]
        if paths:
            return paths

        return [
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.endswith(".json") and name != "dataset_stats.json"
        ]

    def _clean_candidate_entity(self, entity) -> Optional[str]:
        if entity is None:
            return None
        value = str(entity).strip()
        if not value:
            return None
        if value in {"[]", "{}", "None", "nan"}:
            return None
        return value

    def _entity_text_key(self, entity: str) -> str:
        value = str(entity).strip().strip('"').strip("'")
        value = value.replace("_", " ")
        value = re.sub(r"\s+", " ", value)
        return value.lower().strip()

    def _should_use_entity_aligner(self, entity: str) -> bool:
        value = str(entity).strip().strip('"').strip("'")
        if not value or len(value) < 4:
            return False
        if re.fullmatch(r"[\d\.,+\-]+", value):
            return False
        if self._normalize_date(value):
            return False
        return True

    def _candidate_prefilter(self, entity: str) -> List[str]:
        """Cheap lexical prefilter before neural scoring."""
        if not self.schema_entities:
            return []

        query = self._entity_text_key(entity)
        query_tokens = {token for token in re.split(r"[^a-z0-9]+", query) if len(token) > 1}
        scored = []

        for candidate in self.schema_entities:
            candidate_key = self._entity_text_key(candidate)
            if not candidate_key or candidate_key == query:
                continue

            candidate_tokens = {
                token for token in re.split(r"[^a-z0-9]+", candidate_key) if len(token) > 1
            }
            token_overlap = len(query_tokens & candidate_tokens)
            ratio = SequenceMatcher(None, query, candidate_key).ratio()
            contains_bonus = 0.15 if query in candidate_key or candidate_key in query else 0.0
            score = ratio + contains_bonus + min(token_overlap, 3) * 0.10

            if token_overlap > 0 or ratio >= 0.58 or contains_bonus:
                scored.append((score, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _score, candidate in scored[: self.entity_aligner_max_candidates]]

    def _align_entity_with_model(self, entity: str) -> Tuple[str, float]:
        if not self.use_entity_aligner or self.entity_aligner is None:
            return None, 0.0
        if not self._should_use_entity_aligner(entity):
            return None, 0.0

        cache_key = str(entity).strip()
        if cache_key in self.entity_aligner_cache:
            self.entity_aligner_stats["cache_hits"] += 1
            return self.entity_aligner_cache[cache_key]

        self.entity_aligner_stats["attempted"] += 1
        candidates = self._candidate_prefilter(entity)
        if not candidates:
            result = (None, 0.0)
            self.entity_aligner_cache[cache_key] = result
            self.entity_aligner_stats["no_candidates"] += 1
            return result

        self.entity_aligner_stats["scored_candidates"] += len(candidates)
        match, score = self.entity_aligner.find_best_match(
            entity=entity,
            candidates=candidates,
            threshold=self.entity_aligner_threshold,
        )
        result = (match, score)
        self.entity_aligner_cache[cache_key] = result
        if match:
            self.entity_aligner_stats["matched"] += 1
            if self._entity_text_key(match) != self._entity_text_key(entity):
                self.entity_aligner_stats["changed"] += 1
        else:
            self.entity_aligner_stats["below_threshold"] += 1
        return result

    def get_entity_aligner_stats(self) -> Dict:
        stats = dict(self.entity_aligner_stats)
        stats["enabled"] = bool(self.use_entity_aligner)
        stats["cache_size"] = len(self.entity_aligner_cache)
        return stats
    
    def _load_object_filter_rules(self) -> Dict:
        """加载 object 过滤规则"""
        import json
        import os
        
        rules_path = os.path.join(os.path.dirname(__file__), 'object_filter_rules.json')
        
        if os.path.exists(rules_path):
            try:
                with open(rules_path, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                logger.debug(f"Loaded object filter rules from {rules_path}")
                return rules
            except Exception as e:
                logger.warning(f"Failed to load object filter rules: {e}")
                return {'blacklist': [], 'format_corrections': {}}
        else:
            logger.debug(f"Object filter rules not found at {rules_path}")
            return {'blacklist': [], 'format_corrections': {}}
    
    def _load_entity_format_dict(self) -> Dict:
        """加载实体格式字典"""
        import json
        import os
        
        dict_path = os.path.join(os.path.dirname(__file__), 'entity_format_dict.json')
        
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load entity format dict: {e}")
                return {}
        else:
            logger.warning(f"Entity format dict not found at {dict_path}")
            return {}
    
    def _load_entity_alias_dict(self) -> Dict:
        """加载实体别名字典（新增）"""
        import json
        import os
        
        dict_path = os.path.join(os.path.dirname(__file__), 'entity_alias_dict.json')
        
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load entity alias dict: {e}")
                return {}
        else:
            logger.warning(f"Entity alias dict not found at {dict_path}")
            return {}
    
    def process(
        self,
        triples: List[List[str]],
        apply_global_rules: bool = True,
    ) -> List[List[str]]:
        """
        后处理流程

        1. 去重
        2. Schema 白名单过滤（新增，减少 Spurious）
        3. 格式规范化

        When GCA is disabled for a leave-one-out run, retain the neural
        entity aligner as an independent stage instead of silently disabling
        it with the PostProcessor.
        """
        if not triples:
            return []

        if not apply_global_rules:
            return self._process_entity_aligner_only(triples)
        
        # Step 1: 去重
        deduped = self._deduplicate(triples)
        
        # Step 2: Schema filtering is profile-specific. REBEL v1 keeps
        # unknown relations until validation data supports a filter.
        if self.postprocess_profile.filter_schema_relations:
            filtered = self._filter_by_schema(deduped)
        else:
            filtered = deduped
        
        # Step 3: 格式规范化
        if self.postprocess_profile.surface_mode == "preserve":
            normalized = [
                [str(subject).strip(), str(relation).strip(), str(obj).strip()]
                for subject, relation, obj in filtered
            ]
        else:
            normalized = self._normalize_format(filtered)
        
        return normalized

    def _process_entity_aligner_only(self, triples: List[List[str]]) -> List[List[str]]:
        """Apply only neural entity surface alignment while GCA is off."""
        if not self.use_entity_aligner or self.entity_aligner is None:
            return [list(triple) for triple in triples]

        aligned_triples = []
        for subject, relation, obj in triples:
            aligned_subject, _ = self._align_entity_with_model(subject)
            aligned_object, _ = self._align_entity_with_model(obj)
            aligned_triples.append([
                aligned_subject or subject,
                relation,
                aligned_object or obj,
            ])
        return aligned_triples
    
    def _deduplicate(self, triples: List[List[str]]) -> List[List[str]]:
        """去重：保留唯一三元组"""
        seen = set()
        deduped = []
        
        for s, r, o in triples:
            if self.postprocess_profile.dedupe_mode == "exact":
                triple_key = (s, r, o)
            else:
                triple_key = (s.lower(), r.lower(), o.lower())
            if triple_key not in seen:
                seen.add(triple_key)
                deduped.append([s, r, o])
        
        return deduped
    
    def _filter_by_schema(self, triples: List[List[str]]) -> List[List[str]]:
        """Schema白名单过滤：只保留schema中的关系"""
        filtered = []
        
        for s, r, o in triples:
            # 过滤带单位的数值 objects（常见错误）
            # 例如: "610 (metres)", "1249.6 (days)"
            if (
                self.postprocess_profile.strip_numeric_units
                and re.search(r'\(.*?\)', o)
                and re.search(r'[\d\.]', o)
            ):
                # 提取数值部分
                o_clean = re.sub(r'\s*\(.*?\)', '', o).strip()
                logger.debug(f"  Removed unit from object: '{o}' → '{o_clean}'")
                o = o_clean
            
            # 检查关系是否在schema中
            if r in self.schema_relations:
                filtered.append([s, r, o])
            else:
                # 尝试部分匹配（去除下划线、空格）
                r_norm = r.lower().replace('_', '').replace(' ', '')
                matched = False
                
                for schema_rel in self.schema_relations:
                    schema_rel_norm = schema_rel.lower().replace('_', '').replace(' ', '')
                    if r_norm == schema_rel_norm:
                        # 映射到schema关系
                        filtered.append([s, schema_rel, o])
                        matched = True
                        break
                
                if not matched:
                    logger.debug(f"  Filtered out: relation '{r}' not in schema")
        
        return filtered
    
    def _normalize_format(self, triples: List[List[str]]) -> List[List[str]]:
        """
        格式规范化
        """
        normalized = []
        
        for s, r, o in triples:
            s_norm = self._normalize_entity(s)
            o_norm = self._normalize_entity(o)
            normalized.append([s_norm, r, o_norm])
        
        return normalized
    
    def _normalize_entity(self, entity: str, _visited: set = None) -> str:
        """
        实体格式规范化（保守版 - 只做已验证有效的优化）
        
        策略（按优先级）：
        1. 自动生成的格式修正规则（最高优先级）
        2. 日期格式标准化（已验证有效）
        3. 别名映射（已验证有效）
        4. 格式查找表（已验证有效）
        5. 数值精度对齐（_8新增）
        6. 引号内空格处理（_8新增）
        7. Fallback规则（保守）
        """
        if not entity:
            return entity
        
        # 初始化循环检测集合
        if _visited is None:
            _visited = set()
        
        # 移除首尾空白
        entity = entity.strip()
        
        # ===== _8优化1: 引号处理优化 =====
        # 检查是否有引号，如果有引号则特殊处理
        entity_clean = entity
        has_quotes = False
        
        if (entity.startswith('"') and entity.endswith('"')) or \
           (entity.startswith("'") and entity.endswith("'")):
            has_quotes = True
            quote_char = entity[0]
            entity_clean = entity[1:-1].strip()
            
            # 引号内如果有空格，需要转换为下划线（保持Gold格式）
            # 但保留引号（Gold中有些实体确实带引号）
            if ' ' in entity_clean and '_' not in entity_clean:
                # 例如: "United States" 在Gold中不常见，但如果出现保留引号
                # 大部分情况Gold用下划线: United_States
                pass  # 保持原样，让后续处理
        
        # 循环检测：如果已经访问过这个实体，直接返回（避免无限递归）
        if entity_clean in _visited:
            logger.warning(f"Circular alias detected for '{entity_clean}', breaking recursion")
            return entity_clean
        
        # 策略0: 自动生成的格式修正规则（最高优先级）
        format_corrections = self.object_filter_rules.get('format_corrections', {})
        if entity_clean in format_corrections:
            corrected = format_corrections[entity_clean]
            logger.debug(f"Object format correction: '{entity_clean}' → '{corrected}'")
            # 如果原始有引号，保留引号
            if has_quotes:
                return f'{quote_char}{corrected}{quote_char}'
            return corrected
        
        # 策略1: 日期标准化（已验证有效）
        date_normalized = self._normalize_date(entity_clean)
        if date_normalized:
            # 如果原始有引号，保留引号（Gold中某些日期有引号）
            if has_quotes:
                return f'{quote_char}{date_normalized}{quote_char}'
            return date_normalized
        
        # ===== _8优化2: 数值精度对齐 =====
        # 处理 .0 后缀的浮点数
        # 例如: 3500.0 → 保持 3500.0 (与Gold一致)
        # 注意：Gold中有 .0 的就保持，没有的就不加
        if re.match(r'^\d+\.0$', entity_clean):
            # 检查是否应该保留 .0（这里保守策略：保留）
            # 因为Gold中确实有 3500.0 这样的格式
            pass  # 保持原样
        
        # 策略2: 别名映射（第二优先级，已验证有效）
        if entity_clean in self.entity_alias_dict:
            canonical = self.entity_alias_dict[entity_clean]
            logger.debug(f"Entity alias lookup: '{entity_clean}' → '{canonical}'")
            
            # 添加到访问集合
            _visited.add(entity_clean)
            
            # 递归规范化（别名可能也需要格式化）
            result = self._normalize_entity(canonical, _visited)
            
            # 如果原始有引号，保留引号
            if has_quotes and not result.startswith(quote_char):
                return f'{quote_char}{result}{quote_char}'
            return result
        
        # 策略3: 查找表（Gold标准格式，已验证有效）
        if entity_clean in self.entity_format_dict:
            formatted = self.entity_format_dict[entity_clean]
            logger.debug(f"Entity format lookup: '{entity_clean}' → '{formatted}'")
            # 如果原始有引号，保留引号
            if has_quotes:
                return f'{quote_char}{formatted}{quote_char}'
            return formatted
        
        # 也尝试原始版本（可能已经有引号或下划线）
        if entity in self.entity_format_dict:
            return self.entity_format_dict[entity]

        # Optional neural entity aligner. Keep it after deterministic rules
        # and before fallback formatting to make the experiment switchable.
        aligned_entity, align_score = self._align_entity_with_model(entity_clean)
        if aligned_entity:
            logger.debug(
                f"Entity aligner: '{entity_clean}' -> '{aligned_entity}' "
                f"(score={align_score:.4f})"
            )
            if has_quotes and not aligned_entity.startswith(quote_char):
                return f'{quote_char}{aligned_entity}{quote_char}'
            return aligned_entity
        
        # 策略4: Fallback规则（保守策略）
        result = self._fallback_normalize(entity_clean)
        
        # 如果原始有引号，保留引号
        if has_quotes and not result.startswith(quote_char):
            return f'{quote_char}{result}{quote_char}'
        
        return result
    
    def _normalize_date(self, entity: str) -> str:
        """
        日期格式标准化
        
        目标：统一转换为 YYYY-MM-DD 格式
        
        支持格式：
        - November_18th_1923 → 1923-11-18
        - October_13,_1964 → 1964-10-13
        - 10/03/1983 → 1983-10-03
        - March_10,_1983 → 1983-03-10
        """
        import re
        from datetime import datetime
        
        # 已经是标准格式
        if re.match(r'^\d{4}-\d{2}-\d{2}$', entity):
            return entity
        
        # 模式1: Month_DD,_YYYY 或 Month_DDth_YYYY
        # November_18th_1923, October_13,_1964
        month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        # November_18th_1923
        match = re.match(r'([A-Za-z]+)[_\s]+(\d{1,2})[a-z,_\s]+(\d{4})', entity, re.IGNORECASE)
        if match:
            month_str, day, year = match.groups()
            month = month_names.get(month_str.lower())
            if month:
                return f"{year}-{month:02d}-{int(day):02d}"
        
        # 模式2: MM/DD/YYYY or DD/MM/YYYY
        match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', entity)
        if match:
            first, second, year = match.groups()
            # 假设MM/DD/YYYY（美式）
            if int(first) <= 12:
                return f"{year}-{int(first):02d}-{int(second):02d}"
            # DD/MM/YYYY（英式）
            else:
                return f"{year}-{int(second):02d}-{int(first):02d}"
        
        # 模式3: YYYY/MM/DD
        match = re.match(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', entity)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        
        # 不是日期
        return None
    
    def _normalize_number(self, entity: str) -> str:
        """
        数值格式标准化（改进版）
        
        目标：
        1. 移除千位分隔符：1,377,529 → 1377529
        2. 保持数值类型：整数不转浮点数
        3. 标准化格式
        
        返回：标准化后的数值，非数值返回 None
        """
        if not entity:
            return None
        
        # 检查是否是数值类型
        number_pattern = r'^[\d\.\,\-]+$'
        if not re.match(number_pattern, entity):
            return None
        
        try:
            # 移除逗号
            clean_number = entity.replace(',', '')
            
            # 检查是否是浮点数
            if '.' in clean_number:
                # 浮点数：保持浮点格式
                num = float(clean_number)
                return str(num)
            else:
                # 整数：保持整数格式（不转浮点）
                num = int(clean_number)
                return str(num)
        except ValueError:
            return None
    
    def _normalize_geographic(self, entity: str) -> str:
        """
        地理粒度规范化（保守版）
        
        策略：只处理高频且确定的地名
        """
        # 常见地名标准化（仅高频且确定的）
        geo_mappings = {
            # 美国州名（高频）
            'New Hampshire': 'New_Hampshire',
            'New York': 'New_York',
            'North Carolina': 'North_Carolina',
            'South Carolina': 'South_Carolina',
            
            # 国家名（高频）
            'United States': 'United_States',
            'United Kingdom': 'United_Kingdom',
        }
        
        # 精确匹配（大小写敏感）
        if entity in geo_mappings:
            return geo_mappings[entity]
        
        # 不做修改
        return entity
    
    def _fallback_normalize(self, entity: str) -> str:
        """
        Fallback规则（当查找表未命中时）
        
        保守策略：
        1. 保持括号消歧义
        2. 纯数字保持不变
        3. 多词实体：空格转下划线（保持原有大小写）
        4. 单词实体：保持原样
        """
        # 如果包含括号，保持原样
        if '(' in entity:
            return entity
        
        # 如果是纯数字，保持不变（已在 _normalize_number 处理）
        if re.match(r'^[\d\.\-\,]+$', entity):
            return entity
        
        # 多词实体：空格转下划线，保持原有大小写
        if ' ' in entity:
            return entity.replace(' ', '_')
        
        # 单词实体：保持原样（不做大小写转换）
        return entity




class BidirectionalFeedbackEDC(EDC):
    """
    双向反馈式知识图谱抽取框架
    
    完整流程：
    Text → CA-OIE → SD-AC → Bi-SC → GCA → (Feedback Loop)
    """
    
    def __init__(self, **edc_configuration):
        super().__init__(**edc_configuration)
        
        # BF-EDC特定配置
        self.enable_bf_edc = edc_configuration.get("enable_bf_edc", True)
        self.max_feedback_iterations = edc_configuration.get("max_feedback_iterations", 2)
        self.enable_ca_oie = edc_configuration.get("enable_ca_oie", True)
        self.enable_sd_ac = edc_configuration.get("enable_sd_ac", True)
        self.enable_bi_sc = edc_configuration.get("enable_bi_sc", True)
        self.enable_gca = edc_configuration.get("enable_gca", True)
        self.oie_batch_size = max(1, int(edc_configuration.get("oie_batch_size", 1)))

        # 关系分类器配置
        self.enable_relation_classifier = edc_configuration.get("enable_relation_classifier", False)
        self.relation_classifier_model = edc_configuration.get("relation_classifier_model", None)
        self.relation_classifier_threshold = edc_configuration.get("relation_classifier_threshold", 0.5)
        self.enable_relation_reranker = edc_configuration.get("enable_relation_reranker", False)
        self.relation_reranker_top_k = edc_configuration.get("relation_reranker_top_k", 5)
        self.relation_reranker_min_prob = edc_configuration.get("relation_reranker_min_prob", 0.70)
        self.relation_reranker_alias_min_prob = edc_configuration.get("relation_reranker_alias_min_prob", 0.0)
        self.relation_reranker_margin = edc_configuration.get("relation_reranker_margin", 0.20)
        self.relation_reranker_mode = edc_configuration.get("relation_reranker_mode", "conservative")

        # Optional neural entity aligner used inside the GCA/PostProcessor stage.
        self.enable_entity_aligner = edc_configuration.get("enable_entity_aligner", False)
        self.entity_aligner_model = edc_configuration.get("entity_aligner_model", None)
        self.entity_aligner_threshold = edc_configuration.get("entity_aligner_threshold", None)
        self.entity_aligner_max_candidates = edc_configuration.get("entity_aligner_max_candidates", 2000)
        self.entity_aligner_data_dir = edc_configuration.get("entity_aligner_data_dir", "entity_alignment_data_v4")

        # Final output post-processing. This is applied after the full BF-EDC
        # extraction loop and before writing canon_kg.txt.
        self.enable_output_postprocess = edc_configuration.get("enable_output_postprocess", False)
        self.output_postprocess_normalize_objects = edc_configuration.get("output_postprocess_normalize_objects", True)
        self.output_postprocess_repair_relations = edc_configuration.get("output_postprocess_repair_relations", True)
        self.output_postprocess_dedupe = edc_configuration.get("output_postprocess_dedupe", True)
        self.output_postprocess_rule_set = edc_configuration.get("output_postprocess_rule_set", "conservative")
        self.output_postprocess_profile = edc_configuration.get(
            "output_postprocess_profile", "webnlg_conservative"
        )
        
        # 成本优化配置
        self.enable_cost_aware = edc_configuration.get("enable_cost_aware", True)
        self.cheap_model = edc_configuration.get("cheap_model", self.oie_llm_name)
        self.mid_model = edc_configuration.get("mid_model", self.oie_llm_name)
        
        # 初始化各模块（延迟初始化，避免加载不需要的模型）
        self._oie_extractor = None
        self._ca_oie = None
        self._schema_definer = None
        self._sd_ac = None
        self._schema_canonicalizer = None
        self._bi_sc = None
        self._gca = None
        self._relation_classifier = None
        self._relation_reranker = None
        self._output_postprocessor = None
        
        # 统计信息
        self.stats = {
            "total_feedback_iterations": 0,
            "ca_oie_corrections": 0,
            "sd_ac_alignments": 0,
            "relation_reranker_changes": 0,
            "relation_reranker_rejections": 0,
            "entity_aligner": {},
            "output_postprocess_changes": 0,
            "output_postprocess_deduped": 0,
            "bi_sc_iterations": 0,
            "gca_iterations": 0
        }
        
        logger.info(f"BidirectionalFeedbackEDC initialized")
        logger.info(f"  - CA-OIE: {self.enable_ca_oie}")
        logger.info(f"  - SD-AC: {self.enable_sd_ac}")
        logger.info(f"  - Bi-SC: {self.enable_bi_sc}")
        logger.info(f"  - GCA: {self.enable_gca}")
        logger.info(f"  - Relation Classifier: {self.enable_relation_classifier}")
        logger.info(f"  - Relation Reranker: {self.enable_relation_reranker}")
        logger.info(f"  - Entity Aligner: {self.enable_entity_aligner}")
        logger.info(f"  - Output Postprocess: {self.enable_output_postprocess}")
        logger.info(f"  - Output Postprocess Profile: {self.output_postprocess_profile}")
        logger.info(f"  - Max feedback iterations: {self.max_feedback_iterations}")
        logger.info(f"  - OIE batch size: {self.oie_batch_size}")
    
    def _initialize_modules(self):
        """延迟初始化模块（只在需要时加载）"""
        # Always keep the original extractor available. When CA-OIE is
        # disabled, the leave-one-out path uses this extractor without the
        # consistency correction layer.
        if self._oie_extractor is None:
            if not llm_utils.is_model_openai(self.oie_llm_name):
                oie_model, oie_tokenizer = self.load_model(self.oie_llm_name, "hf")
                self._oie_extractor = Extractor(oie_model, oie_tokenizer)
            else:
                self._oie_extractor = Extractor(openai_model=self.oie_llm_name)

        if self._ca_oie is None and self.enable_ca_oie:
            self._ca_oie = ConsistencyAwareOIE(
                extractor=self._oie_extractor,
                model_name=self.oie_llm_name,
                target_schema=self.schema
            )

        # Keep the original schema definer for SD-AC-off runs; SD-AC wraps
        # the same object and adds graph/context alignment around it.
        if self._schema_definer is None:
            if not llm_utils.is_model_openai(self.sd_llm_name):
                sd_model, sd_tokenizer = self.load_model(self.sd_llm_name, "hf")
                self._schema_definer = SchemaDefiner(sd_model, sd_tokenizer)
            else:
                self._schema_definer = SchemaDefiner(openai_model=self.sd_llm_name)

        if self._sd_ac is None and self.enable_sd_ac:
            # Load embedder for SchemaGraph (same as SC embedder)
            sc_embedder = self.load_model(self.sc_embedder_name, "sts")

            self._sd_ac = SchemaDefinerWithAlignment(
                schema_definer=self._schema_definer,
                target_schema=self.schema,
                embedder=sc_embedder  # Pass embedder to enable SchemaGraph
            )

            # 存储prompt路径供后续使用
            self._sd_ac.sd_prompt_template_path = self.sd_template_file_path
            self._sd_ac.sd_few_shot_example_path = self.sd_few_shot_example_file_path

        # Keep the original SC for Bi-SC-off runs. This preserves the EDC
        # baseline instead of leaving the canonicalization stage empty.
        if self._schema_canonicalizer is None:
            if not llm_utils.is_model_openai(self.sc_llm_name):
                sc_model, sc_tokenizer = self.load_model(self.sc_llm_name, "hf")
            else:
                sc_model, sc_tokenizer = None, None

            # Load embedder
            sc_embedder = self.load_model(self.sc_embedder_name, "sts")

            self._schema_canonicalizer = SchemaCanonicalizer(
                self.schema, 
                sc_embedder, 
                sc_model, 
                sc_tokenizer, 
                verify_openai_model=self.sc_llm_name if llm_utils.is_model_openai(self.sc_llm_name) else None
            )

        if self._bi_sc is None and self.enable_bi_sc:
            self._bi_sc = BidirectionalSchemaCanonicalizer(
                schema_canonicalizer=self._schema_canonicalizer
            )

            # 存储prompt路径供后续使用
            self._bi_sc.sc_prompt_template_path = self.sc_template_file_path

        # Entity alignment is an independent option. If GCA is disabled but
        # the aligner remains enabled, PostProcessor runs its entity-only path
        # and does not apply GCA's deduplication/schema filtering.
        if self._gca is None and (self.enable_gca or self.enable_entity_aligner):
            self._gca = PostProcessor(
                target_schema=self.schema,
                use_entity_aligner=self.enable_entity_aligner,
                entity_aligner_path=self.entity_aligner_model,
                entity_aligner_threshold=self.entity_aligner_threshold,
                entity_aligner_max_candidates=self.entity_aligner_max_candidates,
                entity_aligner_data_dir=self.entity_aligner_data_dir,
                postprocess_profile=self.output_postprocess_profile,
            )
        
        needs_relation_model = self.enable_relation_classifier or self.enable_relation_reranker
        if self._relation_classifier is None and needs_relation_model:
            if self.relation_classifier_model and os.path.exists(self.relation_classifier_model):
                self._relation_classifier = RelationClassifierAligner(
                    model_path=self.relation_classifier_model,
                    schema_relations=list(self.schema.keys())
                )
            else:
                logger.warning(f"Relation classifier model not found: {self.relation_classifier_model}")
                self.enable_relation_classifier = False
                self.enable_relation_reranker = False

        if self._relation_reranker is None and self.enable_relation_reranker:
            if self._relation_classifier and self._relation_classifier.aligner:
                reranker_config = RerankerConfig(
                    top_k=self.relation_reranker_top_k,
                    min_prob=self.relation_reranker_min_prob,
                    alias_min_prob=self.relation_reranker_alias_min_prob,
                    margin=self.relation_reranker_margin,
                    mode=self.relation_reranker_mode,
                )
                self._relation_reranker = SchemaAwareRelationReranker(
                    aligner=self._relation_classifier.aligner,
                    schema=self.schema,
                    config=reranker_config,
                )
            else:
                logger.warning("Relation reranker disabled because the relation classifier failed to load")
                self.enable_relation_reranker = False

        if self._output_postprocessor is None and self.enable_output_postprocess:
            postprocess_config = OutputPostprocessorConfig(
                normalize_objects=self.output_postprocess_normalize_objects,
                repair_relations=self.output_postprocess_repair_relations,
                dedupe=self.output_postprocess_dedupe,
                relation_rule_set=self.output_postprocess_rule_set,
                profile=self.output_postprocess_profile,
            )
            self._output_postprocessor = OutputPostprocessor(
                schema=self.schema,
                config=postprocess_config,
            )
    
    def extract_kg(self, input_text_list: List[str], output_dir: str = None, refinement_iterations=0):
        """
        主抽取流程
        
        效率优化：
        - 批量处理文本
        - 早停机制
        - 模块化开关
        """
        if not self.enable_bf_edc:
            logger.info("BF-EDC disabled, using original EDC pipeline")
            return super().extract_kg(input_text_list, output_dir, refinement_iterations)
        
        logger.info("="*70)
        logger.info("Starting Bidirectional Feedback EDC Extraction")
        logger.info("="*70)
        
        # 初始化模块
        self._initialize_modules()
        
        if output_dir is not None:
            import os, pathlib
            if os.path.exists(output_dir):
                logger.warning(f"Output directory {output_dir} already exists!")
            pathlib.Path(f"{output_dir}/iter0").mkdir(parents=True, exist_ok=True)
        
        # 全局状态（用于反馈循环）
        entity_constraints = {}
        schema_alignment_rules = {}
        
        all_results = []
        all_metadata = []
        
        # 反馈循环
        for feedback_iteration in range(self.max_feedback_iterations):
            if feedback_iteration > 0:
                logger.info(f"\n{'='*70}")
                logger.info(f"Feedback Iteration {feedback_iteration + 1}/{self.max_feedback_iterations}")
                logger.info(f"{'='*70}")
            
            iteration_results = []
            iteration_metadata = []
            need_feedback = False
            
            # 原始逐行主循环保留如下，便于回退：
            # for text_idx, text in enumerate(tqdm(input_text_list, desc=f"Extracting (Iter {feedback_iteration+1})")):
            #     result = self._extract_single_text_bf(
            #         text=text,
            #         text_idx=text_idx,
            #         entity_constraints=entity_constraints,
            #         schema_alignment_rules=schema_alignment_rules
            #     )
            #
            #     iteration_results.append(result["triples"])
            #     iteration_metadata.append(result["metadata"])
            #
            #     # 检查是否需要反馈
            #     if result.get("need_feedback", False):
            #         need_feedback = True
            if self.oie_batch_size <= 1 or self._ca_oie is None:
                for text_idx, text in enumerate(tqdm(input_text_list, desc=f"Extracting (Iter {feedback_iteration+1})")):
                    result = self._extract_single_text_bf(
                        text=text,
                        text_idx=text_idx,
                        entity_constraints=entity_constraints,
                        schema_alignment_rules=schema_alignment_rules
                    )

                    iteration_results.append(result["triples"])
                    iteration_metadata.append(result["metadata"])

                    # 检查是否需要反馈
                    if result.get("need_feedback", False):
                        need_feedback = True
            else:
                logger.info("Using batch OIE with batch size %s", self.oie_batch_size)
                oie_few_shot = open(self.oie_few_shot_example_file_path).read()
                oie_template = open(self.oie_prompt_template_file_path).read()
                batch_ranges = range(0, len(input_text_list), self.oie_batch_size)

                for batch_start in tqdm(batch_ranges, desc=f"Extracting OIE batches (Iter {feedback_iteration+1})"):
                    batch_end = min(batch_start + self.oie_batch_size, len(input_text_list))
                    batch_texts = input_text_list[batch_start:batch_end]
                    batch_indices = list(range(batch_start, batch_end))
                    batch_oie_results = self._ca_oie.extract_batch_with_consistency_check(
                        texts=batch_texts,
                        text_indices=batch_indices,
                        few_shot_examples=oie_few_shot,
                        prompt_template=oie_template,
                        entity_constraints=entity_constraints,
                        schema_hints=schema_alignment_rules,
                    )

                    for text_idx, text, oie_result in zip(batch_indices, batch_texts, batch_oie_results):
                        result = self._extract_single_text_bf(
                            text=text,
                            text_idx=text_idx,
                            entity_constraints=entity_constraints,
                            schema_alignment_rules=schema_alignment_rules,
                            precomputed_oie_result=oie_result,
                        )

                        iteration_results.append(result["triples"])
                        iteration_metadata.append(result["metadata"])

                        # 检查是否需要反馈
                        if result.get("need_feedback", False):
                            need_feedback = True
            
            # 更新结果
            all_results = iteration_results
            all_metadata = iteration_metadata
            
            # 决定是否继续反馈循环
            if not need_feedback or feedback_iteration == self.max_feedback_iterations - 1:
                logger.info(f"✓ Extraction completed after {feedback_iteration + 1} feedback iteration(s)")
                break
            
            # 更新全局约束（准备下一轮）
            logger.info("Updating global constraints for next iteration...")
            # 这里可以从all_results中学习更新约束
        
        # 保存结果
        raw_results = copy.deepcopy(all_results)
        output_postprocess_summary = None
        if self.enable_output_postprocess and self._output_postprocessor:
            logger.info("Applying final output post-processing...")
            all_results, output_postprocess_summary = self._output_postprocessor.process_rows(
                rows=all_results,
                contexts=input_text_list,
            )
            self.stats["output_postprocess_changes"] = output_postprocess_summary.get("changed", 0)
            self.stats["output_postprocess_deduped"] = output_postprocess_summary.get("deduped", 0)
            logger.info(
                "Output post-processing changed %s triples and removed %s duplicates",
                self.stats["output_postprocess_changes"],
                self.stats["output_postprocess_deduped"],
            )

        if self._gca and hasattr(self._gca, "get_entity_aligner_stats"):
            self.stats["entity_aligner"] = self._gca.get_entity_aligner_stats()

        if output_dir is not None:
            self._save_results(
                output_dir,
                input_text_list,
                all_results,
                all_metadata,
                raw_results=raw_results if self.enable_output_postprocess else None,
                output_postprocess_summary=output_postprocess_summary,
            )
        
        # 打印统计
        logger.info("\n" + "="*70)
        logger.info("Extraction Statistics:")
        logger.info(f"  - Total feedback iterations: {self.stats['total_feedback_iterations']}")
        logger.info(f"  - CA-OIE corrections: {self.stats['ca_oie_corrections']}")
        logger.info(f"  - SD-AC alignments: {self.stats['sd_ac_alignments']}")
        logger.info(f"  - Relation reranker changes: {self.stats.get('relation_reranker_changes', 0)}")
        logger.info(f"  - Relation reranker rejections: {self.stats.get('relation_reranker_rejections', 0)}")
        entity_aligner_stats = self.stats.get("entity_aligner") or {}
        if entity_aligner_stats:
            logger.info(f"  - Entity aligner loaded: {entity_aligner_stats.get('loaded', False)}")
            logger.info(f"  - Entity aligner candidate pool: {entity_aligner_stats.get('candidate_pool_size', 0)}")
            logger.info(f"  - Entity aligner attempted: {entity_aligner_stats.get('attempted', 0)}")
            logger.info(f"  - Entity aligner matched: {entity_aligner_stats.get('matched', 0)}")
            logger.info(f"  - Entity aligner changed: {entity_aligner_stats.get('changed', 0)}")
        logger.info(f"  - Output postprocess changes: {self.stats.get('output_postprocess_changes', 0)}")
        logger.info(f"  - Output postprocess deduped: {self.stats.get('output_postprocess_deduped', 0)}")
        logger.info(f"  - Bi-SC iterations: {self.stats['bi_sc_iterations']}")
        logger.info(f"  - GCA iterations: {self.stats['gca_iterations']}")
        logger.info("="*70)
        
        return all_results
    
    def _build_raw_oie_result(self, raw_triples: List[List[str]]) -> Dict:
        """Build the uncorrected OIE contract used by SD-AC and SC."""
        triples = [list(triple) for triple in (raw_triples or []) if len(triple) == 3]
        entities = set()
        relations = set()
        for subject, relation, obj in triples:
            entities.update((subject, obj))
            relations.add(relation)
        return {
            "triples": triples,
            "entities": entities,
            "relations": relations,
            "corrections": 0,
            "inconsistencies": copy.deepcopy(_EMPTY_INCONSISTENCIES),
        }

    def _extract_base_oie_result(
        self,
        text: str,
        few_shot_examples: str,
        prompt_template: str,
    ) -> Dict:
        raw_triples = self._oie_extractor.extract(text, few_shot_examples, prompt_template)
        return self._build_raw_oie_result(raw_triples)

    def _define_schema_without_sd_ac(
        self,
        relations: Set[str],
        triples: List[List[str]],
        text_context: str,
    ) -> Tuple[Dict, Dict]:
        few_shot_examples = open(self.sd_few_shot_example_file_path).read()
        prompt_template = open(self.sd_template_file_path).read()
        definitions = self._schema_definer.define_schema(
            input_text_str=text_context,
            extracted_triplets_list=triples,
            few_shot_examples_str=few_shot_examples,
            prompt_template_str=prompt_template,
        )
        # The original EDC stage does not create alignment rules; preserve
        # each open relation for the downstream canonicalizer to decide.
        return definitions, {relation: relation for relation in relations}

    def _canonicalize_without_bi_sc(
        self,
        triples: List[List[str]],
        schema: Dict,
        text_context: str,
    ) -> Dict:
        verify_prompt = ""
        if os.path.exists(self.sc_template_file_path):
            verify_prompt = open(self.sc_template_file_path).read()

        canonical_triples = []
        candidate_dict = {}
        for triple in triples:
            canonicalized, candidates = self._schema_canonicalizer.canonicalize(
                input_text_str=text_context or "",
                open_triplet=triple,
                open_relation_definition_dict=schema,
                verify_prompt_template=verify_prompt,
                enrich=False,
            )
            canonical_triples.append(canonicalized if canonicalized is not None else triple)
            candidate_dict[tuple(triple)] = candidates

        return {
            "triples": canonical_triples,
            "confidence": 1.0 if triples else 1.0,
            "iterations": 1 if triples else 0,
            "converged": True,
            "candidate_dict": candidate_dict,
        }

    def _extract_single_text_bf(
        self,
        text: str,
        text_idx: int,
        entity_constraints: Dict,
        schema_alignment_rules: Dict,
        precomputed_oie_result: Optional[Dict] = None,
    ) -> Dict:
        """
        单个文本的BF-EDC抽取
        
        流程：CA-OIE → SD-AC → [关系分类器] → Bi-SC → GCA
        """
        metadata = {
            "text_idx": text_idx,
            "stages": {},
            "need_feedback": False
        }
        
        # Stage 1: Consistency-Aware OIE
        logger.debug(f"\n[{text_idx}] Stage 1: CA-OIE")
        
        oie_few_shot = open(self.oie_few_shot_example_file_path).read()
        oie_template = open(self.oie_prompt_template_file_path).read()
        
        # 原始逐行OIE逻辑保留如下，便于回退：
        # oie_result = self._ca_oie.extract_with_consistency_check(
        #     text=text,
        #     few_shot_examples=oie_few_shot,
        #     prompt_template=oie_template,
        #     entity_constraints=entity_constraints,
        #     schema_hints=schema_alignment_rules
        # )
        if precomputed_oie_result is None:
            if self.enable_ca_oie and self._ca_oie is not None:
                oie_result = self._ca_oie.extract_with_consistency_check(
                    text=text,
                    few_shot_examples=oie_few_shot,
                    prompt_template=oie_template,
                    entity_constraints=entity_constraints,
                    schema_hints=schema_alignment_rules
                )
            else:
                oie_result = self._extract_base_oie_result(
                    text=text,
                    few_shot_examples=oie_few_shot,
                    prompt_template=oie_template,
                )
        else:
            oie_result = precomputed_oie_result

        oie_inconsistencies = oie_result.get("inconsistencies") or _EMPTY_INCONSISTENCIES
        oie_corrections = oie_result.get("corrections", 0)

        metadata["stages"]["oie"] = {
            "triples_count": len(oie_result["triples"]),
            "entities_count": len(oie_result["entities"]),
            "relations_count": len(oie_result["relations"]),
            "inconsistencies": len(oie_inconsistencies.get("entity_variants", [])),
            "corrections": oie_corrections,
        }

        if oie_corrections > 0:
            self.stats["ca_oie_corrections"] += oie_corrections
        
        if not oie_result["triples"]:
            return {
                "triples": [],
                "metadata": metadata
            }
        
        # Stage 2: Schema Definition with Alignment
        logger.debug(f"[{text_idx}] Stage 2: SD-AC")

        if self.enable_sd_ac and self._sd_ac is not None:
            sd_result, new_alignment_rules = self._sd_ac.define_with_alignment(
                relations=oie_result["relations"],
                text_context=text
            )
        else:
            sd_result, new_alignment_rules = self._define_schema_without_sd_ac(
                relations=oie_result["relations"],
                triples=oie_result["triples"],
                text_context=text,
            )
        
        metadata["stages"]["sd"] = {
            "defined_relations": len(sd_result),
            "new_alignments": len(new_alignment_rules)
        }
        
        self.stats["sd_ac_alignments"] += len(new_alignment_rules)
        schema_alignment_rules.update(new_alignment_rules)
        
        # Stage 2.5: relation reranking/alignment before schema canonicalization
        current_triples = oie_result["triples"]
        if self.enable_relation_reranker and self._relation_reranker:
            logger.debug(f"[{text_idx}] Stage 2.5: Relation Reranker")

            reranked_triples, rr_metadata = self._relation_reranker.rerank_triples(
                triples=current_triples,
                context=text,
                text_idx=text_idx,
            )

            metadata["stages"]["relation_reranker"] = rr_metadata
            current_triples = reranked_triples

            self.stats["relation_reranker_changes"] += rr_metadata["changed"]
            self.stats["relation_reranker_rejections"] += rr_metadata["rejected"]

            logger.debug(f"  - Reranked {rr_metadata['changed']}/{rr_metadata['total']} relations")
            logger.debug(f"  - Rejected {rr_metadata['rejected']} candidate changes")

        elif self.enable_relation_classifier and self._relation_classifier:
            logger.debug(f"[{text_idx}] Stage 2.5: Relation Classifier Alignment")
            
            aligned_triples, rc_metadata = self._relation_classifier.align_relations(
                triples=current_triples,
                context=text,
                confidence_threshold=self.relation_classifier_threshold
            )
            
            metadata["stages"]["relation_classifier"] = rc_metadata
            current_triples = aligned_triples
            
            logger.debug(f"  - Aligned {rc_metadata['aligned']}/{rc_metadata['total']} relations")
            logger.debug(f"  - Avg confidence: {rc_metadata['avg_confidence']:.3f}")
        
        # Stage 3: Bidirectional Schema Canonicalization
        logger.debug(f"[{text_idx}] Stage 3: Bi-SC")

        if self.enable_bi_sc and self._bi_sc is not None:
            sc_result = self._bi_sc.canonicalize_with_verification(
                triples=current_triples,
                schema=sd_result,
                text_context=text,
                max_iterations=3  # 增加到3次迭代
            )
        else:
            sc_result = self._canonicalize_without_bi_sc(
                triples=current_triples,
                schema=sd_result,
                text_context=text,
            )
        
        metadata["stages"]["sc"] = {
            "iterations": sc_result["iterations"],
            "converged": sc_result["converged"],
            "confidence": sc_result.get("confidence", 0.0)
        }
        
        self.stats["bi_sc_iterations"] += sc_result["iterations"]
        
        # Stage 4: Post-Processing
        logger.debug(f"[{text_idx}] Stage 4: Post-Processing")

        if self._gca is not None:
            final_triples = self._gca.process(
                sc_result["triples"],
                apply_global_rules=self.enable_gca,
            )
        else:
            final_triples = sc_result["triples"]
        
        metadata["stages"]["post"] = {
            "final_triples_count": len(final_triples)
        }
        
        # 简化反馈逻辑：只有当三元组数量为0时才需要反馈
        if len(final_triples) == 0 and len(oie_result["triples"]) > 0:
            metadata["need_feedback"] = True
        
        return {
            "triples": final_triples,
            "metadata": metadata,
            "need_feedback": metadata["need_feedback"]
        }
    
    def _save_results(
        self,
        output_dir: str,
        input_texts: List[str],
        all_results: List[List[List[str]]],
        all_metadata: List[Dict],
        raw_results: Optional[List[List[List[str]]]] = None,
        output_postprocess_summary: Optional[Dict] = None,
    ):
        """保存结果"""
        import os
        import json

        if raw_results is not None:
            with open(f"{output_dir}/iter0/canon_kg.raw.txt", "w", encoding="utf-8") as f:
                for triples in raw_results:
                    f.write(str(triples) + "\n")
        
        # 保存三元组结果
        with open(f"{output_dir}/iter0/canon_kg.txt", "w", encoding="utf-8") as f:
            for triples in all_results:
                f.write(str(triples) + "\n")
        
        # 保存详细结果
        detailed_results = []
        for idx, (text, triples, metadata) in enumerate(zip(input_texts, all_results, all_metadata)):
            detailed_results.append({
                "index": idx,
                "input_text": text,
                "extracted_triples": triples,
                "metadata": metadata
            })
        
        with open(f"{output_dir}/iter0/result_at_each_stage.json", "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        with open(f"{output_dir}/iter0/bf_edc_stats.json", "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

        reranker_decisions = []
        for metadata in all_metadata:
            rr_metadata = metadata.get("stages", {}).get("relation_reranker")
            if not rr_metadata:
                continue
            for decision in rr_metadata.get("decisions", []):
                if decision.get("decision") in {"change", "reject"}:
                    reranker_decisions.append(decision)

        if reranker_decisions:
            with open(f"{output_dir}/iter0/relation_reranker_changes.jsonl", "w", encoding="utf-8") as f:
                for decision in reranker_decisions:
                    f.write(json.dumps(decision, ensure_ascii=False) + "\n")

            reranker_summary = {
                "total_logged_decisions": len(reranker_decisions),
                "changed": sum(1 for item in reranker_decisions if item.get("decision") == "change"),
                "rejected": sum(1 for item in reranker_decisions if item.get("decision") == "reject"),
            }
            with open(f"{output_dir}/iter0/relation_reranker_summary.json", "w", encoding="utf-8") as f:
                json.dump(reranker_summary, f, indent=2, ensure_ascii=False)

        if output_postprocess_summary is not None:
            write_postprocess_artifacts(output_dir, output_postprocess_summary)
        
        logger.info(f"Results saved to {output_dir}/iter0/")
