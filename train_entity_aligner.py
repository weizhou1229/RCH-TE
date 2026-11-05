"""
实体对齐模型训练脚本 V2 - 优化版

改进：
1. 类别权重平衡
2. Focal Loss处理样本不平衡
3. 动态阈值优化
4. Early Stopping
5. 更好的评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from collections import defaultdict


class EntityPairDataset(Dataset):
    """实体对数据集"""
    
    def __init__(self, samples: list, tokenizer, max_length=64, use_context=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        entity1 = sample['entity1']
        entity2 = sample['entity2']
        label = sample['label']
        context = sample.get('context', '') if self.use_context else ''
        
        text1 = f"{entity1} {context}" if context else entity1
        encoding1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text2 = f"{entity2} {context}" if context else entity2
        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class SiameseBERT(nn.Module):
    """Siamese BERT for Entity Alignment (优化版)"""
    
    def __init__(
        self, 
        model_name='bert-base-uncased',
        projection_dim=128,
        dropout=0.1,
        pooling_method='mean'  # 新增：支持mean/cls pooling
    ):
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling_method = pooling_method
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        print(f"Model initialized:")
        print(f"  - Encoder: {model_name}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Projection dim: {projection_dim}")
        print(f"  - Pooling method: {pooling_method}")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling - 考虑attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, input_ids, attention_mask):
        """编码单个实体（支持多种pooling）"""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 选择pooling方法
        if self.pooling_method == 'mean':
            # Mean pooling of all tokens
            pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            # CLS token (default)
            pooled = outputs.pooler_output
        
        # Projection
        projected = self.projection(pooled)
        
        # L2 normalization
        normalized = F.normalize(projected, p=2, dim=1)
        
        return normalized
    
    def forward(self, input_ids1, mask1, input_ids2, mask2):
        """计算两个实体的相似度"""
        emb1 = self.encode(input_ids1, mask1)
        emb2 = self.encode(input_ids2, mask2)
        
        similarity = F.cosine_similarity(emb1, emb2)
        
        return similarity


def focal_contrastive_loss(similarity, label, margin=0.5, gamma=2.0, pos_weight=2.0):
    """
    Focal Contrastive Loss with Class Weighting
    
    改进：
    1. 添加focal term (1-p)^gamma 关注难分类样本
    2. 给正样本更高权重pos_weight
    """
    # 相同实体：希望similarity接近1
    positive_loss = label * (1 - similarity) ** 2
    # Focal term: 关注那些similarity还不够高的样本
    positive_focal = (1 - similarity) ** gamma
    positive_loss = positive_loss * positive_focal * pos_weight
    
    # 不同实体：希望similarity小于-margin
    negative_loss = (1 - label) * F.relu(similarity + margin) ** 2
    # Focal term: 关注那些similarity还太高的负样本
    negative_focal = F.relu(similarity + margin) ** gamma
    negative_loss = negative_loss * negative_focal
    
    return (positive_loss + negative_loss).mean()


def find_optimal_threshold(similarities, labels):
    """
    使用验证集找到最优的相似度阈值
    
    目标：最大化F1 score
    """
    thresholds = np.linspace(-1, 1, 200)
    best_f1 = 0.0
    best_threshold = 0.0
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def evaluate(model, dataloader, device, threshold=0.0, use_focal=True, margin=0.5, gamma=2.0, pos_weight=2.0):
    """评估模型"""
    model.eval()
    
    all_similarities = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids1 = batch['input_ids1'].to(device)
            mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)
            
            similarity = model(input_ids1, mask1, input_ids2, mask2)
            
            if use_focal:
                loss = focal_contrastive_loss(similarity, labels, margin, gamma, pos_weight)
            else:
                # 标准contrastive loss
                positive_loss = labels * (1 - similarity) ** 2
                negative_loss = (1 - labels) * F.relu(similarity + margin) ** 2
                loss = (positive_loss + negative_loss).mean()
            
            total_loss += loss.item()
            
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    # 使用给定阈值
    predictions = (all_similarities > threshold).astype(int)
    
    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary', zero_division=0
    )
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'similarities': all_similarities,
        'labels': all_labels,
        'predictions': predictions  # 添加predictions
    }


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    output_dir,
    use_focal=True,
    margin=0.5,
    gamma=2.0,
    pos_weight=2.0,
    patience=5
):
    """训练模型（带Early Stopping + 反馈机制）"""
    
    best_val_f1 = 0.0
    best_epoch = 0
    best_threshold = 0.0
    patience_counter = 0
    
    # 学习曲线记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'threshold': []
    }
    
    print("\n" + "="*60)
    print("Starting training...")
    print(f"  - Use Focal Loss: {use_focal}")
    print(f"  - Margin: {margin}")
    print(f"  - Gamma: {gamma}")
    print(f"  - Pos Weight: {pos_weight}")
    print(f"  - Early Stopping Patience: {patience}")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids1 = batch['input_ids1'].to(device)
            mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            similarity = model(input_ids1, mask1, input_ids2, mask2)
            
            if use_focal:
                loss = focal_contrastive_loss(similarity, labels, margin, gamma, pos_weight)
            else:
                positive_loss = labels * (1 - similarity) ** 2
                negative_loss = (1 - labels) * F.relu(similarity + margin) ** 2
                loss = (positive_loss + negative_loss).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证集评估（先用threshold=0.0）
        val_results = evaluate(model, val_loader, device, threshold=0.0, 
                              use_focal=use_focal, margin=margin, gamma=gamma, pos_weight=pos_weight)
        
        # 找到最优阈值
        optimal_threshold, optimal_f1 = find_optimal_threshold(
            val_results['similarities'], val_results['labels']
        )
        
        # 用最优阈值重新评估
        val_results = evaluate(model, val_loader, device, threshold=optimal_threshold,
                              use_focal=use_focal, margin=margin, gamma=gamma, pos_weight=pos_weight)
        
        # 记录学习曲线
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_accuracy'].append(val_results['accuracy'])
        history['val_precision'].append(val_results['precision'])
        history['val_recall'].append(val_results['recall'])
        history['val_f1'].append(val_results['f1'])
        history['threshold'].append(optimal_threshold)
        
        # 混淆矩阵分析
        predictions = val_results['predictions']
        labels = val_results['labels']
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # 相似度分布分析
        pos_similarities = val_results['similarities'][labels == 1]
        neg_similarities = val_results['similarities'][labels == 0]
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f}")
        print(f"  Val Accuracy: {val_results['accuracy']:.4f}")
        print(f"  Val Precision: {val_results['precision']:.4f}")
        print(f"  Val Recall: {val_results['recall']:.4f}")
        print(f"  Val F1: {val_results['f1']:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        
        # 反馈机制1：混淆矩阵详情
        print(f"\n  📊 Confusion Matrix:")
        print(f"    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        if fp > 0:
            print(f"    ⚠️  False Positives: {fp} (误判为相同)")
        if fn > 0:
            print(f"    ⚠️  False Negatives: {fn} (漏判为不同)")
        
        # 反馈机制2：相似度分布
        if len(pos_similarities) > 0:
            print(f"\n  📈 Similarity Distribution:")
            print(f"    Positive (same entity): mean={pos_similarities.mean():.3f}, std={pos_similarities.std():.3f}")
        if len(neg_similarities) > 0:
            print(f"    Negative (diff entity): mean={neg_similarities.mean():.3f}, std={neg_similarities.std():.3f}")
        
        # 反馈机制3：难样本展示（每5个epoch）
        if (epoch + 1) % 5 == 0:
            print(f"\n  🔍 Hard Samples Analysis:")
            # 找出最难分类的正样本（相似度最低的）
            if len(pos_similarities) > 0:
                hardest_pos_idx = np.where(labels == 1)[0][np.argmin(pos_similarities)]
                print(f"    Hardest Positive: similarity={val_results['similarities'][hardest_pos_idx]:.3f}")
            # 找出最难分类的负样本（相似度最高的）
            if len(neg_similarities) > 0:
                hardest_neg_idx = np.where(labels == 0)[0][np.argmax(neg_similarities)]
                print(f"    Hardest Negative: similarity={val_results['similarities'][hardest_neg_idx]:.3f}")
        
        # 保存最佳模型
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            best_epoch = epoch + 1
            best_threshold = optimal_threshold
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'threshold': best_threshold,
                'optimal_threshold': best_threshold,
                'model_config': {
                    'model_name': model.model_name,
                    'projection_dim': model.projection_dim,
                    'dropout': model.dropout_rate,
                    'pooling_method': model.pooling_method,
                },
            }, os.path.join(output_dir, 'entity_aligner_best.pt'))
            
            print(f"  ✓ Best model saved! (F1={best_val_f1:.4f}, Threshold={best_threshold:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
    
    # 保存学习曲线
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to: {history_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print("="*60)
    
    return best_val_f1, best_threshold


def main():
    parser = argparse.ArgumentParser(description='Train Entity Alignment Model V2')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='entity_aligner_models_v2')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling_method', type=str, default='mean', choices=['mean', 'cls'],
                       help='Pooling method: mean or cls')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=128)  # 增加到128
    parser.add_argument('--use_context', action='store_true')
    
    # Loss参数
    parser.add_argument('--use_focal', action='store_true', default=True,
                       help='Use Focal Contrastive Loss')
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--pos_weight', type=float, default=3.0,
                       help='Weight for positive samples')
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("Entity Alignment Model Training V2")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_name}")
    print(f"Loss: Focal Contrastive (gamma={args.gamma}, pos_weight={args.pos_weight})")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print("="*60)
    
    # 加载数据
    print("\nLoading data...")
    train_data = json.load(open(os.path.join(args.data_dir, 'train.json'), 'r', encoding='utf-8'))
    val_data = json.load(open(os.path.join(args.data_dir, 'val.json'), 'r', encoding='utf-8'))
    test_data = json.load(open(os.path.join(args.data_dir, 'test.json'), 'r', encoding='utf-8'))
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # 统计正负样本比例
    train_pos = sum(1 for s in train_data if s['label'] == 1)
    train_neg = len(train_data) - train_pos
    print(f"\n  Train distribution:")
    print(f"    Positive: {train_pos} ({train_pos/len(train_data)*100:.1f}%)")
    print(f"    Negative: {train_neg} ({train_neg/len(train_data)*100:.1f}%)")
    
    # Tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Datasets
    train_dataset = EntityPairDataset(train_data, tokenizer, args.max_length, args.use_context)
    val_dataset = EntityPairDataset(val_data, tokenizer, args.max_length, args.use_context)
    test_dataset = EntityPairDataset(test_data, tokenizer, args.max_length, args.use_context)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 模型
    print("\nInitializing model...")
    model = SiameseBERT(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        pooling_method=args.pooling_method  # 新增
    ).to(args.device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练
    start_time = time.time()
    best_val_f1, best_threshold = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        epochs=args.epochs,
        output_dir=args.output_dir,
        use_focal=args.use_focal,
        margin=args.margin,
        gamma=args.gamma,
        pos_weight=args.pos_weight,
        patience=args.patience
    )
    
    # 加载最佳模型
    print("\n" + "="*60)
    print("Evaluating on test set...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'entity_aligner_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试集评估（使用最佳阈值）
    test_results = evaluate(model, test_loader, args.device, threshold=best_threshold,
                           use_focal=args.use_focal, margin=args.margin, 
                           gamma=args.gamma, pos_weight=args.pos_weight)
    
    print("\nTest Results:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    print(f"  F1: {test_results['f1']:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    
    training_time = (time.time() - start_time) / 3600
    print(f"\n✅ Training completed in {training_time:.2f} hours")
    
    # 保存结果
    results = {
        'best_epoch': checkpoint['epoch'] + 1,
        'best_val_f1': best_val_f1,
        'best_threshold': best_threshold,
        'test_accuracy': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_f1': test_results['f1'],
        'training_time_hours': training_time,
        'config': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Best model saved to: {os.path.join(args.output_dir, 'entity_aligner_best.pt')}")
    print(f"✓ Results saved to: {results_path}")
    print("="*60)


if __name__ == '__main__':
    main()
