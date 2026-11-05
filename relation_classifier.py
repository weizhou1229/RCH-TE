"""
关系分类器模型 V2

V2 改进（借鉴实体对齐V4成功经验）：
1. Siamese BERT架构：对OIE关系和Schema关系分别编码
2. Focal Loss：处理类别不平衡
3. Mean Pooling：可选的更好的句子表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import json
import os
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


class RelationDataset(Dataset):
    """关系分类数据集"""
    
    def __init__(self, samples: List[Dict], relation_to_id: Dict, tokenizer, max_length=128):
        self.samples = samples
        self.relation_to_id = relation_to_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 检查未知关系
        unknown_relations = set()
        for sample in samples:
            if sample['label'] not in relation_to_id:
                unknown_relations.add(sample['label'])
        
        if unknown_relations:
            print(f"⚠️  Warning: {len(unknown_relations)} unknown relations in dataset:")
            print(f"   {list(unknown_relations)[:10]}")
            print(f"   These will be mapped to class 0 (may cause training issues)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构造输入文本
        # Format: [CLS] subject [SEP] oie_relation [SEP] object [SEP] context [SEP]
        text = f"{sample['subject']} [SEP] {sample['oie_relation']} [SEP] {sample['object']} [SEP] {sample['context']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Label - 关键修复：如果不在schema中，跳过（由数据准备阶段保证）
        if sample['label'] not in self.relation_to_id:
            # 这不应该发生（数据准备时已过滤），但保留保护
            label = 0
        else:
            label = self.relation_to_id[sample['label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class FocalLoss(nn.Module):
    """
    Focal Loss: 处理类别不平衡
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    参数：
        alpha: 平衡因子（default: 0.25）
        gamma: 聚焦参数（default: 2.0）
        
    效果：
        - 降低简单样本权重
        - 聚焦困难样本
    """
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t = probability of true class
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class SiameseRelationClassifier(nn.Module):
    """
    V2: Siamese BERT架构（借鉴实体对齐V4）
    
    Architecture:
        OIE Relation Text → BERT → Embedding_1 ┐
                                                 ├→ Cosine Similarity → MLP → Output
        Schema Relations → BERT → Embedding_2  ┘
    
    优势：
        - 学习关系语义空间
        - 更好的泛化能力
        - 可以处理未见过的OIE格式
    """
    
    def __init__(
        self,
        num_relations: int,
        model_name: str = 'bert-base-uncased',
        dropout: float = 0.3,
        hidden_dim: int = 256,
        use_mean_pooling: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.use_mean_pooling = use_mean_pooling
        
        # Shared BERT encoder
        if 'roberta' in model_name.lower():
            self.encoder = RobertaModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = BertModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        
        # Siamese: 输入是2个embedding的相似度特征
        # [cos_sim, |emb1-emb2|, emb1*emb2] = 1 + hidden + hidden = 1 + 2*hidden
        self.fc1 = nn.Linear(self.hidden_size * 2 + 1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_relations)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling - 考虑attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, input_ids, attention_mask):
        """编码文本为embedding"""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if self.use_mean_pooling:
            # Mean pooling
            embedding = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            # [CLS] token
            embedding = outputs.pooler_output
        
        return embedding
    
    def forward(self, oie_input_ids, oie_attention_mask, schema_input_ids, schema_attention_mask):
        """
        Args:
            oie_input_ids: [batch_size, seq_len] - OIE关系文本
            oie_attention_mask: [batch_size, seq_len]
            schema_input_ids: [batch_size, num_relations, seq_len] - 所有schema关系
            schema_attention_mask: [batch_size, num_relations, seq_len]
        
        Returns:
            logits: [batch_size, num_relations]
        """
        batch_size = oie_input_ids.size(0)
        num_relations = schema_input_ids.size(1)
        
        # 编码OIE关系
        oie_emb = self.encode(oie_input_ids, oie_attention_mask)  # [batch_size, hidden]
        
        # 编码Schema关系（批处理）
        schema_input_ids_flat = schema_input_ids.view(-1, schema_input_ids.size(-1))
        schema_attention_mask_flat = schema_attention_mask.view(-1, schema_attention_mask.size(-1))
        
        schema_emb_flat = self.encode(schema_input_ids_flat, schema_attention_mask_flat)
        schema_emb = schema_emb_flat.view(batch_size, num_relations, -1)  # [batch_size, num_relations, hidden]
        
        # 计算相似度特征
        oie_emb_expanded = oie_emb.unsqueeze(1).expand(-1, num_relations, -1)  # [batch_size, num_relations, hidden]
        
        # 1. Cosine similarity
        cos_sim = F.cosine_similarity(oie_emb_expanded, schema_emb, dim=-1).unsqueeze(-1)  # [batch_size, num_relations, 1]
        
        # 2. Element-wise difference
        diff = torch.abs(oie_emb_expanded - schema_emb)  # [batch_size, num_relations, hidden]
        
        # 3. Element-wise product
        prod = oie_emb_expanded * schema_emb  # [batch_size, num_relations, hidden]
        
        # 拼接特征
        features = torch.cat([cos_sim, diff, prod], dim=-1)  # [batch_size, num_relations, 1+2*hidden]
        
        # MLP分类
        features_flat = features.view(-1, features.size(-1))
        x = self.dropout(features_flat)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        logits_flat = self.fc2(x)
        
        logits = logits_flat.view(batch_size, num_relations)
        
        return logits


# ============================================================================
# V1: 原始BERT分类器（保留用于对比）
# ============================================================================
class RelationClassifier(nn.Module):
    """
    关系分类器模型
    
    Architecture:
        BERT/RoBERTa Encoder
        → [CLS] token embedding
        → Dense(768 → 256)
        → Dropout
        → Output(256 → num_relations)
    """
    
    def __init__(
        self, 
        num_relations: int, 
        model_name: str = 'bert-base-uncased',
        dropout: float = 0.3,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Load pretrained encoder
        if 'roberta' in model_name.lower():
            self.encoder = RobertaModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = BertModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_relations)
        
    def forward(self, input_ids, attention_mask):
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token embedding
        pooled = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Classification
        x = self.dropout(pooled)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class RelationAligner:
    """
    关系对齐器（推理接口）
    
    用于将OIE输出的关系对齐到Schema关系
    """
    
    def __init__(
        self, 
        model_path: str, 
        schema_relations_path: str,
        model_name: str = 'bert-base-uncased',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Load schema relations
        with open(schema_relations_path, 'r', encoding='utf-8') as f:
            self.schema_relations = json.load(f)
        
        self.relation_to_id = {r: i for i, r in enumerate(self.schema_relations)}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}
        
        # Load tokenizer
        if 'roberta' in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = RelationClassifier(
            num_relations=len(self.schema_relations),
            model_name=model_name
        )
        
        # Load weights
        # 原始加载方式保留如下，便于回退：
        # checkpoint = torch.load(model_path, map_location=device)
        # PyTorch 2.6+ 默认 weights_only=True，旧checkpoint会因包含 numpy scalar 等对象而加载失败。
        # 这些checkpoint来自当前项目自训练产物，属于可信来源，因此这里显式关闭 weights_only。
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"RelationAligner loaded from {model_path}")
        print(f"  - Device: {device}")
        print(f"  - Schema relations: {len(self.schema_relations)}")
    
    def align_single_triple(
        self, 
        subject: str, 
        oie_relation: str, 
        obj: str, 
        context: str,
        confidence_threshold: float = 0.5
    ) -> Tuple[str, float]:
        """
        对齐单个三元组的关系
        
        Returns:
            (aligned_relation, confidence)
        """
        # 构造输入
        text = f"{subject} [SEP] {oie_relation} [SEP] {obj} [SEP] {context}"
        
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_id = torch.max(probs, dim=1)
            
            confidence = confidence.item()
            pred_id = pred_id.item()
        
        # 对齐结果
        if confidence >= confidence_threshold:
            aligned_relation = self.id_to_relation[pred_id]
            return aligned_relation, confidence
        else:
            # 置信度不足，返回原关系
            return oie_relation, confidence
    
    def align_triples(
        self, 
        triples: List[List[str]], 
        context: str,
        confidence_threshold: float = 0.5,
        batch_size: int = 32
    ) -> List[Tuple[List[str], float]]:
        """
        批量对齐三元组
        
        Returns:
            [(aligned_triple, confidence), ...]
        """
        if not triples:
            return []
        
        results = []
        
        # 批处理
        for i in range(0, len(triples), batch_size):
            batch_triples = triples[i:i+batch_size]
            
            # 构造批次输入
            texts = []
            for s, r, o in batch_triples:
                text = f"{s} [SEP] {r} [SEP] {o} [SEP] {context}"
                texts.append(text)
            
            encodings = self.tokenizer(
                texts,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # 批量预测
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                confidences, pred_ids = torch.max(probs, dim=1)
            
            # 处理结果
            for j, (s, r, o) in enumerate(batch_triples):
                confidence = confidences[j].item()
                pred_id = pred_ids[j].item()
                
                if confidence >= confidence_threshold:
                    aligned_relation = self.id_to_relation[pred_id]
                    results.append(([s, aligned_relation, o], confidence))
                else:
                    # 置信度不足，保留原关系
                    results.append(([s, r, o], confidence))
        
        return results
    
    def get_top_k_predictions(
        self,
        subject: str,
        oie_relation: str,
        obj: str,
        context: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        获取Top-K预测结果（用于分析）
        
        Returns:
            [(relation, confidence), ...]
        """
        text = f"{subject} [SEP] {oie_relation} [SEP] {obj} [SEP] {context}"
        
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_ids = torch.topk(probs, k=min(top_k, len(self.schema_relations)), dim=1)
        
        results = []
        for prob, idx in zip(top_probs[0], top_ids[0]):
            relation = self.id_to_relation[idx.item()]
            results.append((relation, prob.item()))
        
        return results


def compute_metrics(predictions, labels, num_relations):
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Top-1准确率
    top1_acc = acc
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top1_accuracy': top1_acc
    }


def save_checkpoint(
    model, 
    optimizer, 
    epoch, 
    train_loss, 
    val_metrics, 
    save_path,
    is_best=False
):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_metrics': val_metrics
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f"  ✓ Best model saved to {best_path}")
