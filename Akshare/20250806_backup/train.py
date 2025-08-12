import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import joblib
import heapq

# 导入我们自己的模块
from . import config
from .model import MultiEncoderFusionModel
from .data_utils import get_all_samples
from .dataset import MarketClassificationDataset
from .engine import train_one_epoch, evaluate

def main():
    # --- 1. 准备数据 ---
    print("--- Starting Data Preparation ---")
    all_samples, scalers = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        print("No samples were created. Exiting.")
        return
    
    print(f"Total samples created: {len(all_samples)}")
    
    # --- 2. 创建数据集和数据加载器 ---
    full_dataset = MarketClassificationDataset(all_samples)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    indices = list(range(len(full_dataset)))
    train_dataset = torch.utils.data.Subset(full_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(full_dataset, indices[train_size : train_size + val_size])
    test_dataset = torch.utils.data.Subset(full_dataset, indices[train_size + val_size :])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 打印数据集的切分日期
    train_end_index = indices[train_size - 1]
    val_start_index = indices[train_size]
    val_end_index = indices[train_size + val_size - 1]
    test_start_index = indices[train_size + val_size]
    
    train_end_date = all_samples[train_end_index]['date']
    val_start_date = all_samples[val_start_index]['date']
    val_end_date = all_samples[val_end_index]['date']
    test_start_date = all_samples[test_start_index]['date']
    
    print(f"Train set ends on: {train_end_date.strftime('%Y-%m-%d')}")
    print(f"Validation set from: {val_start_date.strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}")
    print(f"Test set starts on: {test_start_date.strftime('%Y-%m-%d')}")
    print("--- Data Preparation Finished ---")

    # --- 2c. 打印标签分布以检查平衡性 ---
    print("--- Analyzing Label Distribution ---")
    labels = [s['label'] for s in all_samples]
    
    # 找到每个回报率最接近的中心点，作为其“硬标签”
    class_centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"].numpy()
    hard_labels = [np.argmin(np.abs(class_centers - l)) for l in labels]
    
    class_counts = np.bincount(hard_labels, minlength=config.NUM_CLASSES)
    
    print("Class Distribution (based on closest center):")
    for i, count in enumerate(class_counts):
        print(f"  - Center {class_centers[i]*100: >5.1f}%: {count} samples")
    print("------------------------------------")

    # --- 3. 初始化模型、损失函数和优化器 ---
    daily_config = {'feature_size': len(config.FEATURE_COLUMNS['daily']), **config.SHARED_ENCODER_CONFIG}
    weekly_config = {'feature_size': len(config.FEATURE_COLUMNS['weekly']), **config.SHARED_ENCODER_CONFIG}
    monthly_config = {'feature_size': len(config.FEATURE_COLUMNS['monthly']), **config.SHARED_ENCODER_CONFIG}
    
    model = MultiEncoderFusionModel(
        daily_config=daily_config,
        weekly_config=weekly_config,
        monthly_config=monthly_config,
        fusion_dim=config.FUSION_DIM,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    print("--- Model, Loss, and Optimizer Initialized ---")

    # --- 4. 训练循环 ---
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    best_models = {
        'loss': [(float('inf'), '')] * 3,
        'acc@1': [(0.0, '')] * 1,
        'acc@2': [(0.0, '')] * 1,
        'acc@3': [(0.0, '')] * 1,
    }

    print(f"\n--- Starting Training on {config.DEVICE} for {config.EPOCHS} epochs ---")
    for epoch in range(config.EPOCHS):
        train_loss, train_accs = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_accs = evaluate(model, val_loader, criterion, config.DEVICE)
        
        train_acc_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_accs.items()])
        val_acc_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_accs.items()])

        print(f"Epoch {epoch+1:02}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.6f}, Train Accs: [{train_acc_str}] | "
              f"Val Loss: {val_loss:.6f}, Val Accs: [{val_acc_str}]")
        
        # --- 保存最佳模型逻辑 ---
        def update_best_models(metric_type, value, epoch, is_loss=False):
            key = metric_type
            limit = len(best_models[key])
            
            # For loss, we want the minimum, so we compare negatively
            comparison_val = -value if is_loss else value
            current_best_vals = [(-v[0] if is_loss else v[0]) for v in best_models[key]]

            if comparison_val > min(current_best_vals):
                # Find the one to replace
                idx_to_replace = np.argmin(current_best_vals)
                old_path = best_models[key][idx_to_replace][1]
                if os.path.exists(old_path):
                    os.remove(old_path)
                
                # Add new model
                new_path = os.path.join(config.MODEL_DIR, f"model_best_{key.replace('@','_')}_{idx_to_replace+1}.pth")
                best_models[key][idx_to_replace] = (value, new_path)
                torch.save(model.state_dict(), new_path)
                joblib.dump(scalers, os.path.join(config.MODEL_DIR, "scalers.joblib"))
                print(f"  -> Saved new top-{key} model to {new_path}")

        update_best_models('loss', val_loss, epoch + 1, is_loss=True)
        for acc_key, acc_val in val_accs.items():
            update_best_models(acc_key, acc_val, epoch + 1)

    print("--- Training Finished ---")

    # --- 5. 在测试集上进行最终评估 ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    best_loss_model_path = sorted(best_models['loss'], key=lambda x: x[0])[0][1]
    if not os.path.exists(best_loss_model_path):
        print("No best model found to evaluate. Exiting.")
        return
        
    print(f"Loading best model (by loss) from {best_loss_model_path} for final evaluation...")
    model.load_state_dict(torch.load(best_loss_model_path))
    
    test_loss, test_accs = evaluate(model, test_loader, criterion, config.DEVICE)
    test_acc_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_accs.items()])
    
    print(f"Final Test Results -> Loss: {test_loss:.6f}, Accuracies: [{test_acc_str}]")
    print("--- Evaluation Finished ---")

if __name__ == '__main__':
    main()