import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd # 主要用于处理日期或少量数据操作
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib # 用于保存scaler
from tqdm import tqdm

# 本地模块导入
from Util.StockDataLoader import StockDataLoader
from Util.data_utils import prepare_data_for_pytorch_regression
from .lstm_model import LSTMRegressor
from .config import DATA_CONFIG, PREDICTION_CONFIG # 假设模型参数也在PREDICTION_CONFIG或新的MODEL_CONFIG中

def run_training(stock_code=None,
                 save_model_dir='./models/pytorch_lstm'):
    """
    执行完整的模型训练流程，使用config.py中定义的训练和验证日期。
    """
    print("开始执行PyTorch LSTM模型训练流程...")


    os.makedirs(save_model_dir, exist_ok=True)
    
    if stock_code is None:
        raise ValueError("stock_code is required")
    
    train_start_date = DATA_CONFIG['train_start_date']
    print("train_start_date: ", train_start_date)
    train_end_date = DATA_CONFIG['train_end_date']
    print("train_end_date: ", train_end_date)
    val_start_date = DATA_CONFIG['validation_start_date']
    print("val_start_date: ", val_start_date)
    val_end_date = DATA_CONFIG['validation_end_date']
    print("val_end_date: ", val_end_date)

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
    
    print(f"训练目标 - 用过去 {n_past_days} 天数据预测未来 {prediction_window} 天。")
    
    # 处理训练数据并拟合Scalers
    print("准备训练数据并拟合归一化...")
    X_train, y_train, feature_scaler, target_scaler = prepare_data_for_pytorch_regression(
        train_stock_data_df.copy(),
        n_past_days,
        prediction_window
        # 第一次调用，不传入 existing_scalers，函数内部会fit新的
    )
    if X_train is None or y_train is None: # 检查 prepare_data 是否成功
        print("训练数据准备失败（可能数据不足）。训练中止。")
        return
    print(f"训练数据准备完成。X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 保存scalers (在训练数据上拟合的)
    joblib.dump(feature_scaler, os.path.join(save_model_dir, f'{stock_code}_feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(save_model_dir, f'{stock_code}_target_scaler.pkl'))
    print(f"特征和目标归一化器已基于训练数据拟合并保存到: {save_model_dir}")

    # 处理验证数据，使用已拟合的Scalers
    print("准备验证数据 (使用已拟合的归一化器)...")
    # 假设 prepare_data_for_pytorch_regression 将被修改以接受 existing_scalers
    X_val, y_val, _, _ = prepare_data_for_pytorch_regression(
        val_stock_data_df.copy(),
        n_past_days,
        prediction_window,
        existing_feature_scaler=feature_scaler, # 传入已拟合的scaler
        existing_target_scaler=target_scaler  # 传入已拟合的scaler
    )
    if X_val is None or y_val is None: # 检查 prepare_data 是否成功
        print("验证数据准备失败（可能数据不足或与训练数据特征不一致）。训练中止。")
        return
    print(f"验证数据准备完成。X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # 3. 分割数据 (此步骤已通过分别加载和处理训练/验证数据完成，旧的split_ratio逻辑移除)
    print(f"训练集大小 (来自独立时段): {len(X_train)}")
    print(f"验证集大小 (来自独立时段): {len(X_val)}")


    # 4. 创建 DataLoader
    batch_size = PREDICTION_CONFIG['batch_size']
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = TensorDataset(X_val, y_val) # 使用 X_val, y_val
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # 使用 val_loader 进行验证

    # 5. 定义并实例化模型
    if X_train.shape[2] != X_val.shape[2]: # 增加维度检查
        print(f"错误：训练集特征维度 ({X_train.shape[2]}) 与验证集特征维度 ({X_val.shape[2]}) 不匹配！")
        print("这可能发生在scaler使用不当或数据预处理不一致时。请检查prepare_data_for_pytorch_regression的实现。")
        return

    input_dim = X_train.shape[2]
    hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
    layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
    output_dim = 1
    dropout = PREDICTION_CONFIG['lstm_dropout']

    model = LSTMRegressor(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=dropout)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型已实例化并移至设备: {device}")
    print(model)

    # 6. 定义损失函数和优化器
    learning_rate = PREDICTION_CONFIG['learning_rate']
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化最佳训练损失和验证损失以及模型保存路径
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    
    best_train_model_path = os.path.join(save_model_dir, f'{stock_code}_train_best_lstm_model.pth')
    best_val_model_path = os.path.join(save_model_dir, f'{stock_code}_val_best_lstm_model.pth')
    
    print(f"基于训练集最优的模型将尝试保存到: {best_train_model_path}")
    print(f"基于验证集最优的模型将尝试保存到: {best_val_model_path}")

    # 7. 训练模型
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
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_pbar.set_postfix({'Train Batch Loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] 完成, 训练平均损失: {avg_epoch_loss:.6f}')

        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
            torch.save(model.state_dict(), best_train_model_path)
            tqdm.write(f'--- Epoch {epoch+1}: 新的最佳训练模型已保存 (训练损失: {best_train_loss:.6f}) ---')

        # --- 开始验证阶段 ---
        model.eval()
        val_loss_epoch = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False) # 使用 val_loader
        with torch.no_grad():
            for val_data, val_targets in val_pbar:
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                val_outputs = model(val_data)
                loss = criterion(val_outputs, val_targets)
                val_loss_epoch += loss.item()
                val_pbar.set_postfix({'Val Batch Loss': f'{loss.item():.4f}'})
        
        avg_val_loss_epoch = val_loss_epoch / len(val_loader) # 使用 val_loader
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] - 验证平均损失: {avg_val_loss_epoch:.6f}')

        if avg_val_loss_epoch < best_val_loss:
            best_val_loss = avg_val_loss_epoch
            torch.save(model.state_dict(), best_val_model_path)
            tqdm.write(f'*** Epoch {epoch+1}: 新的最佳验证模型已保存 (验证损失: {best_val_loss:.6f}) ***')
        # --- 结束验证阶段 ---

    print("模型训练与验证完成。")

    # 8. 提示最佳模型保存信息
    if os.path.exists(best_train_model_path) and best_train_loss != float('inf'):
        print(f"基于训练集最优的模型已保存在: {best_train_model_path} (最低训练损失: {best_train_loss:.6f})")
    else:
        print("未能找到或保存更优的模型（基于训练损失）。")

    if os.path.exists(best_val_model_path) and best_val_loss != float('inf'):
        print(f"基于验证集最优的模型已保存在: {best_val_model_path} (最低验证损失: {best_val_loss:.6f})")
    else:
        print("未能找到或保存更优的模型（基于验证损失）。")
    
    print("PyTorch LSTM模型训练流程（含验证）结束。")

if __name__ == "__main__":
    run_training(DATA_CONFIG['default_stock_code'])