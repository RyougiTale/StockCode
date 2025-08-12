import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

# 导入我们自己的模块
from . import config
from .model import MultiEncoderFusionModel
from .data_utils import get_all_samples
from .dataset import MarketClassificationDataset
from .engine import train_one_epoch, evaluate

def main():
    # --- 1. 准备数据 ---
    print("--- Starting Data Preparation ---")
    all_samples = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        print("No samples were created. Exiting.")
        return
    
    print(f"Total samples created: {len(all_samples)}")
    
    # --- 2a. 计算类别权重以解决样本不平衡问题 ---
    print("--- Calculating Class Weights ---")
    labels = [s['label'] for s in all_samples]
    class_counts = np.bincount(labels)
    # 权重与类别数量成反比
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    print(f"Class counts (0, 1): {class_counts}")
    print(f"Calculated class weights: {class_weights}")
    
    # --- 2b. 创建数据集和数据加载器 ---
    full_dataset = MarketClassificationDataset(all_samples)
    
    # 【修正】严格按时间顺序划分训练集和验证集，防止数据穿越
    # 注意：get_all_samples 保证了样本是按时间顺序排列的
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 使用 torch.utils.data.Subset 来进行划分
    indices = list(range(len(full_dataset)))
    train_dataset = torch.utils.data.Subset(full_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(full_dataset, indices[train_size:])

    # 训练集需要打乱顺序 (shuffle=True)，验证集不需要
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print("--- Data Preparation Finished ---")

    # --- 3. 初始化模型、损失函数和优化器 ---
    # 获取特征数量
    feature_size = len(config.FEATURE_COLUMNS)
    
    # 为每个编码器创建配置
    encoder_config = {
        'feature_size': feature_size,
        **config.SHARED_ENCODER_CONFIG
    }
    
    model = MultiEncoderFusionModel(
        daily_config=encoder_config,
        weekly_config=encoder_config,
        monthly_config=encoder_config,
        fusion_dim=config.FUSION_DIM,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    # 使用计算出的权重来创建加权的交叉熵损失函数
    class_weights = class_weights.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # AdamW 优化器，是 Adam 的一个改进版本，通常效果更好
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    print("--- Model, Loss, and Optimizer Initialized ---")

    # --- 4. 训练循环 ---
    best_val_accuracy = 0.0
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print(f"\n--- Starting Training on {config.DEVICE} for {config.EPOCHS} epochs ---")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.DEVICE)
        
        print(f"Epoch {epoch+1:02}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存验证集上表现最好的模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  -> New best model saved to {config.MODEL_PATH} with accuracy: {val_acc:.4f}")
            
    print("--- Training Finished ---")

if __name__ == '__main__':
    main()