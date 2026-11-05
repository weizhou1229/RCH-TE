"""
实体对齐器（推理模块）

用于在BF-EDC中进行实体对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import json
import os
from typing import List, Tuple, Dict
import numpy as np


class SiameseBERT(nn.Module):
    """Siamese BERT（与训练脚本v4保持一致 - 使用Mean Pooling）"""
    
    def __init__(self, model_name='bert-base-uncased', projection_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling - 考虑attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 使用Mean Pooling而不是CLS token
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        projected = self.projection(pooled)
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized
    
    def forward(self, input_ids1, mask1, input_ids2, mask2):
        emb1 = self.encode(input_ids1, mask1)
        emb2 = self.encode(input_ids2, mask2)
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity


class EntityAligner:
    """
    实体对齐器（推理接口）
    
    用于判断两个实体字符串是否指向同一对象
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        max_length: int = 128,  # 改为128，与v4训练时一致
        optimal_threshold: float = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # 加载checkpoint
        print(f"Loading entity aligner from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取模型配置
        model_config = checkpoint.get('model_config', {})
        model_name = model_config.get('model_name', 'bert-base-uncased')
        projection_dim = model_config.get('projection_dim', 128)
        
        # 加载最优阈值（从checkpoint或参数）
        # _9版本：使用训练得到的最优阈值0.8593
        if optimal_threshold is not None:
            self.optimal_threshold = optimal_threshold
        else:
            self.optimal_threshold = checkpoint.get(
                'optimal_threshold', checkpoint.get('threshold', 0.8593)
            )
        
        # _10版本尝试（已回退）：降低阈值到0.75效果不佳
        # self.optimal_threshold = optimal_threshold or checkpoint.get('optimal_threshold', 0.75)
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 加载模型
        self.model = SiameseBERT(
            model_name=model_name,
            projection_dim=projection_dim
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Entity aligner loaded")
        print(f"  - Device: {self.device}")
        print(f"  - Model: {model_name}")
        print(f"  - Projection dim: {projection_dim}")
        print(f"  - Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"  - Max length: {self.max_length}")
    
    def predict_similarity(
        self,
        entity1: str,
        entity2: str,
        context: str = None
    ) -> float:
        """
        预测两个实体的相似度
        
        Returns:
            similarity score in [0, 1] (实际是cosine similarity归一化后的值)
        """
        # 构造输入
        text1 = f"{entity1} {context}" if context else entity1
        text2 = f"{entity2} {context}" if context else entity2
        
        encoding1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids1 = encoding1['input_ids'].to(self.device)
        mask1 = encoding1['attention_mask'].to(self.device)
        input_ids2 = encoding2['input_ids'].to(self.device)
        mask2 = encoding2['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            similarity = self.model(input_ids1, mask1, input_ids2, mask2)
            score = similarity.item()
        
        # 将cosine similarity从[-1, 1]映射到[0, 1]
        # 但实际上我们返回原始的cosine similarity（更直观）
        return score
    
    def predict_batch(
        self,
        entity_pairs: List[Tuple[str, str]],
        contexts: List[str] = None,
        batch_size: int = 32
    ) -> List[float]:
        """
        批量预测实体对的相似度
        
        Args:
            entity_pairs: [(entity1, entity2), ...]
            contexts: [context1, context2, ...] (optional)
            batch_size: 批处理大小
        
        Returns:
            [similarity1, similarity2, ...]
        """
        if contexts is None:
            contexts = [None] * len(entity_pairs)
        
        all_scores = []
        
        for i in range(0, len(entity_pairs), batch_size):
            batch_pairs = entity_pairs[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            # 编码所有entity1
            texts1 = []
            texts2 = []
            
            for (e1, e2), ctx in zip(batch_pairs, batch_contexts):
                text1 = f"{e1} {ctx}" if ctx else e1
                text2 = f"{e2} {ctx}" if ctx else e2
                texts1.append(text1)
                texts2.append(text2)
            
            encodings1 = self.tokenizer(
                texts1,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            encodings2 = self.tokenizer(
                texts2,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids1 = encodings1['input_ids'].to(self.device)
            mask1 = encodings1['attention_mask'].to(self.device)
            input_ids2 = encodings2['input_ids'].to(self.device)
            mask2 = encodings2['attention_mask'].to(self.device)
            
            # 预测
            with torch.no_grad():
                similarities = self.model(input_ids1, mask1, input_ids2, mask2)
                scores = similarities.cpu().numpy().tolist()
            
            all_scores.extend(scores)
        
        return all_scores
    
    def find_best_match(
        self,
        entity: str,
        candidates: List[str],
        context: str = None,
        threshold: float = None  # 如果不指定，使用训练得到的最优阈值
    ) -> Tuple[str, float]:
        """
        在候选实体中找到最佳匹配
        
        Args:
            entity: 待对齐的实体
            candidates: 候选实体列表
            context: 上下文（可选）
            threshold: 最低相似度阈值（默认使用训练时的最优阈值）
        
        Returns:
            (best_match, score) 或 (None, 0.0) 如果没有匹配
        """
        if not candidates:
            return None, 0.0
        
        # 使用训练得到的最优阈值
        if threshold is None:
            threshold = self.optimal_threshold
        
        # 批量预测
        entity_pairs = [(entity, cand) for cand in candidates]
        contexts = [context] * len(candidates) if context else None
        
        scores = self.predict_batch(entity_pairs, contexts)
        
        # 找到最高分
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        if best_score >= threshold:
            return candidates[best_idx], best_score
        else:
            return None, best_score


# 用于测试的简单函数
def test_entity_aligner(model_path: str):
    """测试实体对齐器"""
    
    print("="*60)
    print("Testing Entity Aligner")
    print("="*60)
    
    aligner = EntityAligner(model_path)
    
    # 测试用例
    test_cases = [
        ("United States", "United_States", "相同（空格vs下划线）"),
        ("New York", "New York City", "不同（城市vs州）"),
        ("USA", "United States", "相同（缩写）"),
        ("3500", "3500.0", "相同（数值）"),
        ("1983-10-03", "\"1983-10-03\"", "相同（引号）"),
        ("Adolfo_Suarez", "Adolfo_Suárez", "相同（特殊字符）"),
        ("Apple", "Microsoft", "不同（公司）"),
    ]
    
    print("\nTest Cases:")
    print("-" * 60)
    
    for e1, e2, description in test_cases:
        score = aligner.predict_similarity(e1, e2)
        prediction = "SAME" if score > 0.5 else "DIFFERENT"
        
        print(f"{e1:20s} ⟷ {e2:20s}")
        print(f"  Score: {score:.4f} → {prediction} ({description})")
        print()
    
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python entity_aligner.py <model_path>")
        print("Example: python entity_aligner.py entity_aligner_models/entity_aligner_best.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_entity_aligner(model_path)
