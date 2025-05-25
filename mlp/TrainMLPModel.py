import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm

# 确保 Util 和 LSTM_Regression_Stock_Value 模块可以被导入
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设mlp文件夹在项目根目录下）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from Util.StockDataLoader import StockDataLoader
from Util.data_utils import prepare_data_for_mlp_three_class # 导入新的数据准备函数
from LSTM_Regression_Stock_Value.config import DATA_CONFIG, PREDICTION_CONFIG # 复用现有配置

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=3, dropout_prob=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        # CrossEntropyLoss 在内部应用 Softmax，所以这里不需要

    def forward(self, x):
        # MLP 通常接收扁平化的输入
        x = x.view(x.size(0), -1) # 将 (batch_size, n_steps, num_features) 展平为 (batch_size, n_steps * num_features)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

def run_mlp_classification_training(stock_code=None, save_model_dir='./models/pytorch_mlp_classification'):
    print("开始执行PyTorch MLP股市涨跌三分类模型训练流程...")
    os.makedirs(save_model_dir, exist_ok=True)
    if stock_code is None:
        stock_code = DATA_CONFIG.get('default_stock_code', '600036') # 从配置获取或使用默认值
        print(f"未提供 stock_code，使用默认值: {stock_code}")

    train_start_date = DATA_CONFIG['train_start_date']
    train_end_date = DATA_CONFIG['train_end_date']
    val_start_date = DATA_CONFIG['validation_start_date']
    val_end_date = DATA_CONFIG['validation_end_date']
    
    print(f"股票代码: {stock_code}")
    print(f"训练数据周期: {train_start_date} 到 {train_end_date}")
    print(f"验证数据周期: {val_start_date} 到 {val_end_date}")

    # 1. 加载股票数据
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    print("加载训练数据...")
    train_stock_data_df = loader.load_stock_data(stock_code, train_start_date, train_end_date)
    if train_stock_data_df is None or train_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的训练数据。训练中止。")
        return
    print(f"成功加载 {len(train_stock_data_df)} 条训练数据。")

    print("加载验证数据...")
    val_stock_data_df = loader.load_stock_data(stock_code, val_start_date, val_end_date)
    if val_stock_data_df is None or val_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的验证数据。训练中止。")
        return
    print(f"成功加载 {len(val_stock_data_df)} 条验证数据。")

    # 2. 准备数据
    prediction_window = PREDICTION_CONFIG['prediction_days']
    n_past_days = PREDICTION_CONFIG.get('n_past_days_mlp', PREDICTION_CONFIG.get('n_past_days_debug', 20)) # 可自定义或复用
    up_threshold = PREDICTION_CONFIG.get('mlp_up_threshold', 0.02)
    down_threshold = PREDICTION_CONFIG.get('mlp_down_threshold', -0.02)

    print(f"训练目标 - 用过去 {n_past_days} 天数据预测未来 {prediction_window} 天涨跌平（涨>{up_threshold*100}%, 跌<{down_threshold*100}%）。")
    
    print("准备训练数据并拟合归一化...")
    X_train, y_train, feature_scaler = prepare_data_for_mlp_three_class(
        train_stock_data_df.copy(),
        n_past_days,
        prediction_window,
        up_threshold=up_threshold,
        down_threshold=down_threshold
    )
    if X_train is None or y_train is None:
        print("训练数据准备失败（可能数据不足）。训练中止。")
        return
    print(f"训练数据准备完成。X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    unique_train_labels, counts_train_labels = np.unique(y_train.numpy(), return_counts=True)
    print(f"训练集标签分布: {dict(zip(unique_train_labels, counts_train_labels))}")

    # 保存scaler
    scaler_path = os.path.join(save_model_dir, f'{stock_code}_mlp_feature_scaler.pkl')
    joblib.dump(feature_scaler, scaler_path)
    print(f"特征归一化器已保存到: {scaler_path}")

    # 准备验证数据
    print("准备验证数据 (使用已拟合的归一化器)...")
    X_val, y_val, _ = prepare_data_for_mlp_three_class(
        val_stock_data_df.copy(),
        n_past_days,
        prediction_window,
        existing_feature_scaler=feature_scaler,
        up_threshold=up_threshold,
        down_threshold=down_threshold
    )
    if X_val is None or y_val is None:
        print("验证数据准备失败（可能数据不足）。训练中止。")
        return
    print(f"验证数据准备完成。X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    unique_val_labels, counts_val_labels = np.unique(y_val.numpy(), return_counts=True)
    print(f"验证集标签分布: {dict(zip(unique_val_labels, counts_val_labels))}")

    # 3. 创建 DataLoader
    batch_size = PREDICTION_CONFIG.get('batch_size_mlp', PREDICTION_CONFIG.get('batch_size', 64))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. 定义并实例化模型
    # MLP的input_dim是 n_past_days * num_features
    input_dim = X_train.shape[1] * X_train.shape[2] 
    hidden_dim1 = PREDICTION_CONFIG.get('mlp_hidden_dim1', 128)
    hidden_dim2 = PREDICTION_CONFIG.get('mlp_hidden_dim2', 64)
    dropout = PREDICTION_CONFIG.get('mlp_dropout', 0.3)
    output_dim = 3 # 跌 (0), 平仓 (1), 涨 (2)

    model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"MLP模型已实例化并移至设备: {device}")
    print(model)

    # 5. 定义损失函数和优化器
    # CrossEntropyLoss 用于多分类，ignore_index=1 表示忽略标签为1的样本（平仓）
    learning_rate = PREDICTION_CONFIG.get('learning_rate_mlp', PREDICTION_CONFIG.get('learning_rate', 0.001))
    criterion = nn.CrossEntropyLoss(ignore_index=1) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_f1_macro_adj = 0.0 # 基于调整后的F1分数 (只考虑类别0和2)
    best_val_model_path = os.path.join(save_model_dir, f'{stock_code}_val_best_mlp_classifier.pth')
    print(f"基于验证集最优的模型将尝试保存到: {best_val_model_path}")
    
    num_epochs = PREDICTION_CONFIG.get('num_epochs_mlp', PREDICTION_CONFIG.get('num_epochs_debug', 50))
    print(f"开始训练模型（包含验证），共 {num_epochs} 轮...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for data, targets in train_pbar:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data) # MLP的forward会自动展平
            loss = criterion(outputs, targets)
            
            # 只计算有效样本的损失（如果需要手动过滤）
            # valid_indices = targets != 1
            # if valid_indices.sum() > 0:
            #     loss = criterion(outputs[valid_indices], targets[valid_indices])
            # else:
            #     loss = torch.tensor(0.0, device=device, requires_grad=True) # 如果批次中所有样本都被忽略

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_pbar.set_postfix({'Train Batch Loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] 完成, 训练平均损失: {avg_epoch_loss:.6f}')
        
        # Validation
        model.eval()
        val_loss_epoch = 0.0
        all_preds_epoch_val = []
        all_targets_epoch_val = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for val_data, val_targets in val_pbar:
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                val_outputs = model(val_data)
                
                loss = criterion(val_outputs, val_targets) # ignore_index 会处理
                val_loss_epoch += loss.item()
                
                # 获取预测类别 (argmax)
                _, predicted_classes = torch.max(val_outputs, 1)
                
                all_preds_epoch_val.extend(predicted_classes.cpu().numpy())
                all_targets_epoch_val.extend(val_targets.cpu().numpy())
                val_pbar.set_postfix({'Val Batch Loss': f'{loss.item():.4f}'})

        avg_val_loss_epoch = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0
        
        targets_np_val = np.array(all_targets_epoch_val)
        preds_np_val = np.array(all_preds_epoch_val)

        # 计算指标时，我们主要关注类别 0 (跌) 和 2 (涨)
        # 过滤掉类别 1 (平仓) 的样本进行特定评估
        valid_indices_val = (targets_np_val == 0) | (targets_np_val == 2)
        if np.sum(valid_indices_val) > 0:
            targets_filtered_val = targets_np_val[valid_indices_val]
            preds_filtered_val = preds_np_val[valid_indices_val]

            # 调整准确率：只在有效类别上计算
            val_acc_adj = np.mean(targets_filtered_val == preds_filtered_val) if len(targets_filtered_val) > 0 else 0.0
            
            # F1, Precision, Recall (宏平均，只针对类别0和2)
            # sklearn的classification_report可以直接处理，或者手动计算
            # labels=[0, 2] 确保只报告这两个类别的指标
            # target_names=['跌(0)', '涨(2)']
            report = classification_report(targets_filtered_val, preds_filtered_val, labels=[0, 2], 
                                           target_names=['跌', '涨'], output_dict=True, zero_division=0)
            
            val_f1_macro_adj = report['macro avg']['f1-score']
            val_precision_macro_adj = report['macro avg']['precision']
            val_recall_macro_adj = report['macro avg']['recall']

            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss_epoch:.6f}')
            tqdm.write(f'  Adjusted Metrics (labels 0 and 2 only):')
            tqdm.write(f'    Adj Acc: {val_acc_adj:.4f}, Adj F1 (macro): {val_f1_macro_adj:.4f}, Adj P (macro): {val_precision_macro_adj:.4f}, Adj R (macro): {val_recall_macro_adj:.4f}')
            
            # 打印针对类别 0 和 2 的详细分类报告
            report_0_2_str = classification_report(targets_filtered_val, preds_filtered_val, labels=[0, 2], target_names=['跌(0)', '涨(2)'], zero_division=0)
            tqdm.write(f"  Detailed Classification Report (labels 0 and 2):\n{report_0_2_str}")

            if val_f1_macro_adj > best_val_f1_macro_adj:
                best_val_f1_macro_adj = val_f1_macro_adj
                torch.save(model.state_dict(), best_val_model_path)
                tqdm.write(f'*** Epoch {epoch+1}: New best validation model saved (Val Adj F1 (0,2): {best_val_f1_macro_adj:.4f}) ***')
        
        else: # 如果验证集中没有类别0或2的样本
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss_epoch:.6f}. No samples of class 0 or 2 in validation set for adjusted metrics.')

        # 打印包含所有三个类别的完整混淆矩阵和分类报告
        tqdm.write(f'  Full Metrics (all labels 0, 1, 2):')
        full_conf_matrix = confusion_matrix(targets_np_val, preds_np_val, labels=[0, 1, 2])
        tqdm.write(f"  Full Confusion Matrix (labels 0,1,2):\n{full_conf_matrix}")
        full_class_report_str = classification_report(targets_np_val, preds_np_val, labels=[0, 1, 2], target_names=['跌(0)', '平仓(1)', '涨(2)'], zero_division=0)
        tqdm.write(f"  Full Classification Report (labels 0,1,2):\n{full_class_report_str}")


    print("模型训练与验证完成。")
    if os.path.exists(best_val_model_path) and best_val_f1_macro_adj > 0:
        print(f"基于验证集最优的模型已保存在: {best_val_model_path} (最高验证 Adj F1 (0,2): {best_val_f1_macro_adj:.4f})")
    else:
        print("未能找到或保存更优的模型（基于验证集调整后F1分数）。")
    print("PyTorch MLP股市涨跌三分类模型训练流程结束。")


if __name__ == "__main__":
    # 允许通过命令行参数指定股票代码
    stock_to_train = DATA_CONFIG.get('default_stock_code', '600036')
    if len(sys.argv) > 1:
        stock_to_train = sys.argv[1]
        print(f"从命令行参数获取股票代码: {stock_to_train}")
    
    run_mlp_classification_training(stock_code=stock_to_train)