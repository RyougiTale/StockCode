import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# 本地模块导入
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../LSTM_Regression_Stock_Value')))
from Util.StockDataLoader import StockDataLoader
from Util.data_utils import prepare_data_for_pytorch_classification
from LSTM_Regression_Stock_Value.lstm_model import LSTMClassifier
from LSTM_Regression_Stock_Value.config import DATA_CONFIG, PREDICTION_CONFIG


def run_classification_training(stock_code=None, save_model_dir='./models/pytorch_lstm_classification'):
    print("开始执行PyTorch LSTM股市涨跌分类模型训练流程...")
    os.makedirs(save_model_dir, exist_ok=True)
    if stock_code is None:
        raise ValueError("stock_code is required")
    train_start_date = DATA_CONFIG['train_start_date']
    train_end_date = DATA_CONFIG['train_end_date']
    val_start_date = DATA_CONFIG['validation_start_date']
    val_end_date = DATA_CONFIG['validation_end_date']
    print(f"股票代码: {stock_code}")
    print(f"训练数据周期: {train_start_date} 到 {train_end_date}")
    print(f"验证数据周期: {val_start_date} 到 {val_end_date}")
    # 1. 加载股票数据
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    print(f"加载训练数据...")
    train_stock_data_df = loader.load_stock_data(stock_code, train_start_date, train_end_date)
    if train_stock_data_df is None or train_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的训练数据。训练中止。")
        return
    print(f"成功加载 {len(train_stock_data_df)} 条训练数据。")
    print(f"加载验证数据...")
    val_stock_data_df = loader.load_stock_data(stock_code, val_start_date, val_end_date)
    if val_stock_data_df is None or val_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的验证数据。训练中止。")
        return
    print(f"成功加载 {len(val_stock_data_df)} 条验证数据。")
    # 2. 准备数据
    prediction_window = PREDICTION_CONFIG['prediction_days']
    n_past_days = PREDICTION_CONFIG['n_past_days_debug']
    print(f"训练目标 - 用过去 {n_past_days} 天数据预测未来 {prediction_window} 天涨跌。")
    # 处理训练数据并拟合Scaler
    print("准备训练数据并拟合归一化...")
    X_train, y_train, feature_scaler = prepare_data_for_pytorch_classification(
        train_stock_data_df.copy(),
        n_past_days,
        prediction_window
    )
    if X_train is None or y_train is None:
        print("训练数据准备失败（可能数据不足）。训练中止。")
        return
    print(f"训练数据准备完成。X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print('训练集涨天数:', int(y_train.sum()), '跌天数:', len(y_train)-int(y_train.sum()))
    # 保存scaler
    print(f"特征归一化器已基于训练数据拟合并保存到: {save_model_dir}")
    # 处理验证数据，使用已拟合的Scaler
    print("准备验证数据 (使用已拟合的归一化器)...")
    X_val, y_val, _ = prepare_data_for_pytorch_classification(
        val_stock_data_df.copy(),
        n_past_days,
        prediction_window,
        existing_feature_scaler=feature_scaler
    )
    if X_val is None or y_val is None:
        print("验证数据准备失败（可能数据不足或与训练数据特征不一致）。训练中止。")
        return
    joblib.dump(feature_scaler, os.path.join(save_model_dir, f'{stock_code}_feature_scaler.pkl'))
    print(f"验证数据准备完成。X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print('验证集涨天数:', int(y_val.sum()), '跌天数:', len(y_val)-int(y_val.sum()))
    # 3. 分割数据（已通过分别加载和处理训练/验证数据完成）
    print(f"训练集大小 (来自独立时段): {len(X_train)}")
    print(f"验证集大小 (来自独立时段): {len(X_val)}")
    # 4. 创建 DataLoader
    batch_size = PREDICTION_CONFIG['batch_size']
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # 5. 定义并实例化模型
    if X_train.shape[2] != X_val.shape[2]:
        print(f"错误：训练集特征维度 ({X_train.shape[2]}) 与验证集特征维度 ({X_val.shape[2]}) 不匹配！")
        return
    input_dim = X_train.shape[2]
    # print(input_dim)
    hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
    layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
    dropout = PREDICTION_CONFIG['lstm_dropout']
    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_size=1, dropout_prob=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型已实例化并移至设备: {device}")
    # print(model)
    # 6. 定义损失函数和优化器
    learning_rate = PREDICTION_CONFIG['learning_rate']
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    best_val_model_path = os.path.join(save_model_dir, f'{stock_code}_val_best_lstm_classifier.pth')
    print(f"基于验证集最优的模型将尝试保存到: {best_val_model_path}")
    num_epochs = PREDICTION_CONFIG['num_epochs_debug']
    print(f"开始训练模型（包含验证），共 {num_epochs} 轮...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for batch_idx, (data, targets) in enumerate(batch_pbar):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            # print(outputs, targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_pbar.set_postfix({'Train Batch Loss': f'{loss.item():.4f}'})
        avg_epoch_loss = epoch_loss / len(train_loader)
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] 完成, 训练平均损失: {avg_epoch_loss:.6f}')
        # --- 验证阶段 ---
        model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for val_data, val_targets in val_pbar:
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                val_outputs = model(val_data)
                loss = criterion(val_outputs, val_targets)
                val_loss_epoch += loss.item()
                preds = (val_outputs > 0.5).float()
                # print(preds)
                correct += (preds == val_targets).sum().item()
                total += val_targets.size(0)
                val_pbar.set_postfix({'Val Batch Loss': f'{loss.item():.4f}'})
        avg_val_loss_epoch = val_loss_epoch / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] - 验证平均损失: {avg_val_loss_epoch:.6f}，准确率: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_val_model_path)
            tqdm.write(f'*** Epoch {epoch+1}: 新的最佳验证模型已保存 (验证准确率: {best_val_acc:.4f}) ***')
    print("模型训练与验证完成。")
    if os.path.exists(best_val_model_path) and best_val_acc > 0:
        print(f"基于验证集最优的模型已保存在: {best_val_model_path} (最高验证准确率: {best_val_acc:.4f})")
    else:
        print("未能找到或保存更优的模型（基于验证准确率）。")
    print("PyTorch LSTM股市涨跌分类模型训练流程结束。")

def evaluate_single_features(stock_code=None, save_model_dir='./models/pytorch_lstm_classification'):
    print("\n===== 单特征有效性评估 =====")
    os.makedirs(save_model_dir, exist_ok=True)
    if stock_code is None:
        raise ValueError("stock_code is required")
    train_start_date = DATA_CONFIG['train_start_date']
    train_end_date = DATA_CONFIG['train_end_date']
    val_start_date = DATA_CONFIG['validation_start_date']
    val_end_date = DATA_CONFIG['validation_end_date']
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    train_stock_data_df = loader.load_stock_data(stock_code, train_start_date, train_end_date)
    val_stock_data_df = loader.load_stock_data(stock_code, val_start_date, val_end_date)
    prediction_window = PREDICTION_CONFIG['prediction_days']
    n_past_days = PREDICTION_CONFIG['n_past_days_debug']
    feature_list = ['close', 'high', 'low', 'open', 'volume']
    results = {}
    for feat in feature_list:
        print(f'\n==== 只用特征: {feat} ====')
        X_train, y_train, feature_scaler = prepare_data_for_pytorch_classification(
            train_stock_data_df.copy(),
            n_past_days,
            prediction_window,
            feature_columns=[feat]
        )
        X_val, y_val, _ = prepare_data_for_pytorch_classification(
            val_stock_data_df.copy(),
            n_past_days,
            prediction_window,
            existing_feature_scaler=feature_scaler,
            feature_columns=[feat]
        )
        if X_train is None or y_train is None or X_val is None or y_val is None:
            print(f"特征 {feat} 数据准备失败，跳过。")
            results[feat] = None
            continue
        batch_size = PREDICTION_CONFIG['batch_size']
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        input_dim = X_train.shape[2]
        print(X_train.shape)
        hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
        layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
        dropout = PREDICTION_CONFIG['lstm_dropout']
        model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_size=1, dropout_prob=dropout)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        learning_rate = PREDICTION_CONFIG['learning_rate']
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        num_epochs = 3  # 单特征评估可适当减少轮数加快速度
        best_val_acc = 0.0
        best_f1 = 0.0 
        best_precision = 0.0
        best_recall = 0.0

        for epoch in range(num_epochs):
            model.train()
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            model.eval()
            correct = 0
            total = 0
            all_preds_epoch = []
            all_targets_epoch = []
            with torch.no_grad():
                for val_data, val_targets in val_loader:
                    val_data, val_targets = val_data.to(device), val_targets.to(device)
                    val_outputs = model(val_data)
                    preds = (val_outputs > 0.5).float()
                    correct += (preds == val_targets).sum().item()
                    total += val_targets.size(0)
                    all_preds_epoch.extend(preds.cpu().numpy())
                    all_targets_epoch.extend(val_targets.cpu().numpy())
            
            current_epoch_val_acc = correct / total if total > 0 else 0
            current_epoch_f1 = 0
            current_epoch_precision = 0
            current_epoch_recall = 0

            if total > 0:
                targets_np = np.array(all_targets_epoch).ravel()
                preds_np = np.array(all_preds_epoch).ravel()
                current_epoch_f1 = f1_score(targets_np, preds_np, zero_division=0)
                current_epoch_precision = precision_score(targets_np, preds_np, zero_division=0)
                current_epoch_recall = recall_score(targets_np, preds_np, zero_division=0)

            if current_epoch_val_acc > best_val_acc:
                best_val_acc = current_epoch_val_acc
                best_f1 = current_epoch_f1
                best_precision = current_epoch_precision
                best_recall = current_epoch_recall
            elif epoch == num_epochs - 1 and best_val_acc == 0.0 and total > 0 :
                best_val_acc = current_epoch_val_acc 
                best_f1 = current_epoch_f1
                best_precision = current_epoch_precision
                best_recall = current_epoch_recall
        
        print(f"特征 {feat} - 最佳验证 Acc: {best_val_acc:.4f}, F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
        results[feat] = {'acc': best_val_acc, 'f1': best_f1, 'precision': best_precision, 'recall': best_recall}
        
        if y_val is not None:
            num_val_samples = y_val.shape[0] if isinstance(y_val, torch.Tensor) else len(y_val)
            if num_val_samples > 0:
                num_positive_val_samples = (torch.sum(y_val).item() if isinstance(y_val, torch.Tensor) else np.sum(y_val))
                print(f"特征 {feat} - 验证集样本数: {num_val_samples}, 验证集正样本数: {int(num_positive_val_samples)}")
            else:
                print(f"特征 {feat} - 验证集数据为空。")
        else:
            print(f"特征 {feat} - 验证集数据(y_val)未生成。")
            
    print("\n===== 单特征评估结果汇总 =====")
    for feat, metrics in results.items():
        if metrics is not None:
            print(f"特征 {feat}: Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        else:
            print(f"特征 {feat}: 无数据")

if __name__ == "__main__":
    run_classification_training(DATA_CONFIG['default_stock_code'])
    # evaluate_single_features(DATA_CONFIG['default_stock_code'])
