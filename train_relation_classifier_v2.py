"""
训练关系分类器 V2

V2 改进（借鉴实体对齐V4成功经验）：
1. 大数据集：WebNLG全集480K样本（V1仅23K）
2. 三层负样本策略：Very Hard + Hard + Random
3. Focal Loss：处理类别不平衡
4. Early Stopping：防止过拟合
5. 详细评估指标：对齐率、置信度分布

推荐用法（简单BERT架构）：
    python train_relation_classifier_v2.py \
        --data_dir relation_classifier_data/webnlg_v2 \
        --output_dir relation_classifier_models/webnlg_v2 \
        --use_focal_loss \
        --early_stopping_patience 3 \
        --epochs 20 \
        --batch_size 128

注意：Siamese架构由于内存限制不推荐使用（--use_siamese已废弃）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from relation_classifier import (
    RelationDataset, 
    RelationClassifier,
    FocalLoss,
    compute_metrics, 
    save_checkpoint
)

# Siamese架构已废弃（内存问题），如果需要可以单独导入
try:
    from relation_classifier import SiameseRelationDataset, SiameseRelationClassifier
    SIAMESE_AVAILABLE = True
except ImportError:
    SIAMESE_AVAILABLE = False
    print("Note: Siamese architecture not available (recommended to use simple architecture)")



class EarlyStopping:
    """Early Stopping回调"""
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        """
        Args:
            patience: 多少个epoch没有改善就停止
            min_delta: 最小改善量
            mode: 'max' (F1等) 或 'min' (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def compute_detailed_metrics(predictions, labels, confidences, num_relations):
    """
    V2: 计算详细评估指标
    
    Returns:
        {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'alignment_rate': float,  # 高置信度对齐率
            'avg_confidence': float,   # 平均置信度
            'high_conf_acc': float     # 高置信度样本准确率
        }
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # 对齐率：置信度>0.5的样本比例
    high_conf_mask = confidences > 0.5
    alignment_rate = high_conf_mask.mean()
    
    # 平均置信度
    avg_confidence = confidences.mean()
    
    # 高置信度样本准确率
    if high_conf_mask.sum() > 0:
        high_conf_acc = (predictions[high_conf_mask] == labels[high_conf_mask]).mean()
    else:
        high_conf_acc = 0.0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'alignment_rate': alignment_rate,
        'avg_confidence': avg_confidence,
        'high_conf_acc': high_conf_acc
    }


def train_epoch_v1(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    """V1训练函数（简单BERT架构）"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().detach().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_detailed_metrics(
        np.array(all_preds), 
        np.array(all_labels),
        np.array(all_confidences),
        model.fc2.out_features
    )
    
    return avg_loss, metrics


def train_epoch_siamese(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    """V2训练函数（Siamese架构）"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in progress_bar:
        oie_input_ids = batch['oie_input_ids'].to(device)
        oie_attention_mask = batch['oie_attention_mask'].to(device)
        schema_input_ids = batch['schema_input_ids'].to(device)
        schema_attention_mask = batch['schema_attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(oie_input_ids, oie_attention_mask, schema_input_ids, schema_attention_mask)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().detach().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_detailed_metrics(
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confidences),
        model.fc2.out_features
    )
    
    return avg_loss, metrics


def evaluate_v1(model, dataloader, criterion, device, split_name="Val"):
    """V1评估函数（简单BERT架构）"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    progress_bar = tqdm(dataloader, desc=f"{split_name}")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().detach().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_detailed_metrics(
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confidences),
        model.fc2.out_features
    )
    
    return avg_loss, metrics


def evaluate_siamese(model, dataloader, criterion, device, split_name="Val"):
    """V2评估函数（Siamese架构）"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    progress_bar = tqdm(dataloader, desc=f"{split_name}")
    
    with torch.no_grad():
        for batch in progress_bar:
            oie_input_ids = batch['oie_input_ids'].to(device)
            oie_attention_mask = batch['oie_attention_mask'].to(device)
            schema_input_ids = batch['schema_input_ids'].to(device)
            schema_attention_mask = batch['schema_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(oie_input_ids, oie_attention_mask, schema_input_ids, schema_attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().detach().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_detailed_metrics(
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confidences),
        model.fc2.out_features
    )
    
    return avg_loss, metrics


def main(args):
    print("="*70)
    print("Training Relation Classifier V2")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Architecture: {'Siamese' if args.use_siamese else 'Simple'}")
    print(f"Loss: {'Focal Loss' if args.use_focal_loss else 'CrossEntropy'}")
    
    # 1. Load data
    print("\n[1] Loading data...")
    data_dir = args.data_dir
    
    with open(f"{data_dir}/train.json", 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    with open(f"{data_dir}/val.json", 'r', encoding='utf-8') as f:
        val_samples = json.load(f)
    with open(f"{data_dir}/schema_relations.json", 'r', encoding='utf-8') as f:
        schema_relations = json.load(f)
    
    print(f"  - Train samples: {len(train_samples):,}")
    print(f"  - Val samples: {len(val_samples):,}")
    print(f"  - Schema relations: {len(schema_relations)}")
    
    # 2. Create relation mapping
    relation_to_id = {r: i for i, r in enumerate(schema_relations)}
    id_to_relation = {i: r for r, i in relation_to_id.items()}
    
    # 3. Load tokenizer
    print(f"\n[2] Loading tokenizer: {args.model_name}")
    if 'roberta' in args.model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # 4. Create datasets
    print(f"\n[3] Creating datasets...")
    
    if args.use_siamese:
        if not SIAMESE_AVAILABLE:
            print("❌ Error: Siamese architecture not available!")
            print("   Please use simple architecture (remove --use_siamese flag)")
            return
        # Siamese架构
        train_dataset = SiameseRelationDataset(
            train_samples, relation_to_id, schema_relations, tokenizer, args.max_length
        )
        val_dataset = SiameseRelationDataset(
            val_samples, relation_to_id, schema_relations, tokenizer, args.max_length
        )
    else:
        # 简单架构（推荐）
        train_dataset = RelationDataset(train_samples, relation_to_id, tokenizer, args.max_length)
        val_dataset = RelationDataset(val_samples, relation_to_id, tokenizer, args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 5. Create model
    print(f"\n[4] Creating model...")
    
    if args.use_siamese:
        if not SIAMESE_AVAILABLE:
            print("❌ Error: Siamese architecture not available!")
            return
        model = SiameseRelationClassifier(
            num_relations=len(schema_relations),
            model_name=args.model_name,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            use_mean_pooling=args.use_mean_pooling
        )
    else:
        model = RelationClassifier(
            num_relations=len(schema_relations),
            model_name=args.model_name,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim
        )
    
    model = model.to(device)
    
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 6. Optimizer and scheduler
    print("\n[5] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Warmup steps: {warmup_steps:,}")
    
    # 7. Loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, num_classes=len(schema_relations))
        print(f"  - Using Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  - Using CrossEntropy Loss")
    
    # 8. Early Stopping
    early_stopping = None
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode='max')
        print(f"  - Early Stopping enabled (patience={args.early_stopping_patience})")
    
    # 9. Training loop
    print("\n[6] Training...")
    print("="*70)
    
    best_val_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_f1': [],
        'train_alignment_rate': [],
        'val_loss': [],
        'val_f1': [],
        'val_alignment_rate': [],
        'val_avg_confidence': []
    }
    
    # 选择训练/评估函数
    if args.use_siamese:
        train_fn = train_epoch_siamese
        eval_fn = evaluate_siamese
    else:
        train_fn = train_epoch_v1
        eval_fn = evaluate_v1
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_metrics = train_fn(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train F1:   {train_metrics['f1']:.4f}")
        print(f"  Train Align Rate: {train_metrics['alignment_rate']:.4f}")
        print(f"  Train Avg Conf:   {train_metrics['avg_confidence']:.4f}")
        
        # Validate
        val_loss, val_metrics = eval_fn(model, val_loader, criterion, device, "Val")
        
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}")
        print(f"  Val Prec:   {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val Align Rate: {val_metrics['alignment_rate']:.4f}")
        print(f"  Val Avg Conf:   {val_metrics['avg_confidence']:.4f}")
        print(f"  Val High Conf Acc: {val_metrics['high_conf_acc']:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['train_alignment_rate'].append(train_metrics['alignment_rate'])
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_alignment_rate'].append(val_metrics['alignment_rate'])
        history['val_avg_confidence'].append(val_metrics['avg_confidence'])
        
        # Save checkpoint
        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1']
        
        checkpoint_path = f"{args.output_dir}/checkpoint_epoch_{epoch}.pt"
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_metrics,
            checkpoint_path, is_best=is_best
        )
        
        if is_best:
            print(f"  ✓ New best F1: {best_val_f1:.4f}")
        
        # Early Stopping
        if early_stopping:
            if early_stopping(val_metrics['f1']):
                print(f"\n⚠️  Early Stopping triggered after {epoch} epochs")
                print(f"   No improvement for {early_stopping.patience} epochs")
                break
    
    # 10. Save final results
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print(f"Best validation F1: {best_val_f1:.4f}")
    
    # Save history - convert numpy types to Python types
    history_serializable = {}
    for key, values in history.items():
        history_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in values]
    
    with open(f"{args.output_dir}/training_history.json", 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"  - Best model: {args.output_dir}/checkpoint_epoch_*_best.pt")
    print(f"  - Training history: {args.output_dir}/training_history.json")
    
    # V2特有的建议
    if args.use_siamese and best_val_f1 > 0.75:
        print(f"\n✅ V2 Model Performance: GOOD (F1 > 0.75)")
        print(f"   Ready for full-scale BF-EDC integration!")
    elif best_val_f1 < 0.75:
        print(f"\n⚠️  V2 Model Performance: NEEDS IMPROVEMENT (F1 < 0.75)")
        print(f"   Consider:")
        print(f"   1. Generate more training data")
        print(f"   2. Adjust focal loss parameters")
        print(f"   3. Use larger BERT model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save models')
    
    # Model
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-large-uncased',
                                'roberta-base', 'roberta-large'],
                        help='Pretrained model name')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of classification head')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Architecture
    parser.add_argument('--use_siamese', action='store_true',
                        help='Use Siamese architecture (V2)')
    parser.add_argument('--use_mean_pooling', action='store_true',
                        help='Use mean pooling instead of [CLS] token')
    
    # Loss
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal Loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma parameter')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (smaller for Siamese due to memory)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max sequence length')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience (0 to disable)')
    
    args = parser.parse_args()
    
    main(args)
