import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Util.StockDataLoader import StockDataLoader
from Util.data_utils import prepare_data_for_pytorch_classification
from LSTM_Regression_Stock_Value.lstm_model import LSTMClassifier
from LSTM_Regression_Stock_Value.config import DATA_CONFIG, PREDICTION_CONFIG

def calculate_model_performance(model, data_loader, device, criterion=None):
    """计算模型在给定数据加载器上的性能"""
    model.eval()
    all_targets = []
    all_preds = []
    total_loss = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    targets_np = np.array(all_targets).ravel()
    preds_np = np.array(all_preds).ravel()
    
    acc = accuracy_score(targets_np, preds_np)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    
    avg_loss = total_loss / len(data_loader) if criterion and len(data_loader) > 0 else None
    return acc, f1, avg_loss

def run_permutation_importance(stock_code=None, model_path=None, save_model_dir='./models/pytorch_lstm_classification'):
    print(f"开始对股票 {stock_code} 的模型进行 Permutation Importance 分析...")
    if stock_code is None:
        raise ValueError("stock_code is required")
    if model_path is None:
        model_path = os.path.join(save_model_dir, f'{stock_code}_val_best_lstm_classifier.pth')

    if not os.path.exists(model_path):
        print(f"错误: 预训练模型未找到于 {model_path}。请先运行 TrainModel.py 中的 run_classification_training。")
        return

    # --- 1. 加载数据和预处理器 ---
    val_start_date = DATA_CONFIG['validation_start_date']
    val_end_date = DATA_CONFIG['validation_end_date']
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    val_stock_data_df = loader.load_stock_data(stock_code, val_start_date, val_end_date)

    if val_stock_data_df is None or val_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的验证数据。分析中止。")
        return

    prediction_window = PREDICTION_CONFIG['prediction_days']
    n_past_days = PREDICTION_CONFIG['n_past_days_debug'] # 与训练时保持一致
    
    # 加载训练时使用的 scaler
    scaler_path = os.path.join(save_model_dir, f'{stock_code}_feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"错误: 特征归一化器未找到于 {scaler_path}。请确保已成功运行训练。")
        return
    feature_scaler = joblib.load(scaler_path)

    # 默认特征列 (与 run_classification_training 一致)
    # 在 prepare_data_for_pytorch_classification 中，如果 feature_columns=None，则会使用包含 ma 和 rsi 的默认列
    # 我们需要知道这些列名以便进行打乱
    # 假设 prepare_data_for_pytorch_classification 内部会处理好特征工程
    # 我们需要的是最终进入模型的特征列名，这通常是 feature_scaler.feature_names_in_ (如果scaler是sklearn的)
    # 或者，我们可以从 prepare_data_for_pytorch_classification 的输出来推断
    
    # 为了获取正确的特征列名和顺序，我们先调用一次 prepare_data
    # 注意：这里的 feature_columns 应该与训练时一致，即 None，让函数内部决定
    X_val_orig, y_val_orig, _ = prepare_data_for_pytorch_classification(
        val_stock_data_df.copy(),
        n_past_days,
        prediction_window,
        existing_feature_scaler=feature_scaler,
        feature_columns=None # 使用训练时的默认特征
    )

    if X_val_orig is None or y_val_orig is None:
        print("验证数据准备失败。分析中止。")
        return

    # 获取实际使用的特征名称 (假设 prepare_data_for_pytorch_classification 使用了所有传入的原始列，并可能添加了技术指标)
    # 对于 Permutation Importance，我们需要知道输入到 LSTM 模型之前的特征维度顺序
    # X_val_orig.shape[2] 是特征数量
    # 假设 feature_scaler.get_feature_names_out() (sklearn >= 0.24) 或 feature_scaler.feature_names_in_ (旧版)
    # 或者，如果 prepare_data_for_pytorch_classification 返回了特征名列表会更好
    # 暂时硬编码，需要与 prepare_data_for_pytorch_classification 内部逻辑同步
    # 默认特征列在 prepare_data_for_pytorch_classification 中是:
    # ['close', 'high', 'low', 'open', 'volume', 'ma5', 'ma10', 'rsi14']
    # 这些是原始数据中的列，经过 scaler 变换后输入模型
    # 我们打乱的是 scaler 之前的原始数据中的列
    
    # 修正：我们应该打乱的是 X_val_orig (即归一化后的数据) 中的特征维度
    # X_val_orig 的 shape 是 (n_samples, n_past_days, n_features)
    # 我们要打乱的是 n_features 这个维度中的某一列
    
    # 获取特征名称列表，这需要与 prepare_data_for_pytorch_classification 内部的 feature_columns 一致
    # 如果 prepare_data_for_pytorch_classification 中 feature_columns=None, 它会使用默认列表
    temp_df_for_cols = val_stock_data_df.copy()
    temp_df_for_cols['ma5'] = temp_df_for_cols['close'].rolling(window=5).mean()
    temp_df_for_cols['ma10'] = temp_df_for_cols['close'].rolling(window=10).mean()
    temp_df_for_cols['rsi14'] = 100 - 100 / (1 + temp_df_for_cols['close'].pct_change().rolling(14).mean())
    default_feature_columns = ['close', 'high', 'low', 'open', 'volume', 'ma5', 'ma10', 'rsi14']
    
    # 确保这些列在 temp_df_for_cols 中都存在，并且顺序与 scaler 训练时一致
    # 实际上，feature_scaler 是基于这些列的 values 训练的，顺序很重要
    # 如果 feature_scaler 是 Sklearn Scaler, 可以用 feature_scaler.feature_names_in_
    try:
        actual_feature_names = list(feature_scaler.feature_names_in_)
    except AttributeError:
        print("警告: 使用的 scaler 可能不是标准的 Sklearn Scaler，无法自动获取 feature_names_in_。将使用默认特征列表。")
        actual_feature_names = default_feature_columns
        if X_val_orig.shape[2] != len(actual_feature_names):
            print(f"错误: X_val_orig 特征维度 {X_val_orig.shape[2]} 与预期特征数 {len(actual_feature_names)} 不符。")
            return


    print(f"模型使用的特征 ({len(actual_feature_names)}): {actual_feature_names}")

    val_dataset_orig = TensorDataset(X_val_orig, y_val_orig)
    val_loader_orig = DataLoader(val_dataset_orig, batch_size=PREDICTION_CONFIG['batch_size'], shuffle=False)

    # --- 2. 加载预训练模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_val_orig.shape[2] # 特征数量
    model = LSTMClassifier(input_dim, PREDICTION_CONFIG['lstm_hidden_dim'], PREDICTION_CONFIG['lstm_layer_dim'], 1, PREDICTION_CONFIG['lstm_dropout'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已从 {model_path} 加载并移至 {device}")

    # --- 3. 计算基线性能 ---
    baseline_acc, baseline_f1, _ = calculate_model_performance(model, val_loader_orig, device)
    print(f"基线性能 - 验证集 Acc: {baseline_acc:.4f}, F1: {baseline_f1:.4f}")

    # --- 4. 执行 Permutation Importance ---
    importances = {}
    
    # 我们需要原始的、未处理成序列的、但已经过归一化的特征数据，以便按特征列打乱
    # X_val_orig 是 (n_samples, n_timesteps, n_features)
    # 我们要打乱的是 n_features 维度
    
    print("\n开始计算特征重要性 (Permutation Importance)...")
    for i, feature_name in enumerate(actual_feature_names):
        print(f"  打乱特征: {feature_name} (索引 {i})...")
        
        X_val_permuted_tensor = X_val_orig.clone() # 创建副本进行修改
        
        # 对时间序列数据中的特定特征进行打乱
        # X_val_permuted_tensor 的形状: (num_samples, sequence_length, num_features)
        # 我们要打乱的是第 i 个特征在所有样本和所有时间步上的值
        
        # 方案1: 对每个样本的每个时间步的该特征值进行独立打乱 (可能破坏时序内部结构) - 不太对
        # 方案2: 对整个特征列 (跨样本，跨时间步) 进行打乱 - 更像是标准做法
        # 我们打乱的是特征维度 i 上的所有值
        
        # 获取第 i 个特征在所有样本和时间步上的数据
        feature_column_data = X_val_permuted_tensor[:, :, i].clone() # Shape: (num_samples, sequence_length)
        
        # 打乱这个特征的数据
        # 将其展平，打乱，然后重塑回原来的形状
        original_shape = feature_column_data.shape
        flattened_feature = feature_column_data.reshape(-1)
        permutation_indices = torch.randperm(flattened_feature.size(0))
        permuted_flattened_feature = flattened_feature[permutation_indices]
        
        # 将打乱后的特征放回 X_val_permuted_tensor
        X_val_permuted_tensor[:, :, i] = permuted_flattened_feature.reshape(original_shape)

        permuted_dataset = TensorDataset(X_val_permuted_tensor, y_val_orig)
        permuted_loader = DataLoader(permuted_dataset, batch_size=PREDICTION_CONFIG['batch_size'], shuffle=False)
        
        permuted_acc, permuted_f1, _ = calculate_model_performance(model, permuted_loader, device)
        
        # 使用 F1 分数下降作为重要性度量
        importance_score = baseline_f1 - permuted_f1
        importances[feature_name] = importance_score
        print(f"    打乱后 Acc: {permuted_acc:.4f}, F1: {permuted_f1:.4f}. F1 重要性: {importance_score:.4f}")

    # --- 5. 展示结果 ---
    print("\n===== Permutation Importance 结果 (基于F1分数下降) =====")
    sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    for feature_name, score in sorted_importances:
        print(f"特征: {feature_name:<15} | 重要性 (F1下降): {score:.4f}")

if __name__ == "__main__":
    stock_to_analyze = DATA_CONFIG.get('default_stock_code', '600036') # 从config获取或使用默认
    
    # 确保模型和scaler已存在
    model_file = os.path.join('./models/pytorch_lstm_classification', f'{stock_to_analyze}_val_best_lstm_classifier.pth')
    scaler_file = os.path.join('./models/pytorch_lstm_classification', f'{stock_to_analyze}_feature_scaler.pkl')

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        print(f"模型 ({model_file}) 或特征缩放器 ({scaler_file}) 不存在。")
        print("请先运行 'python -m Classification.TrainModel' 中的 'run_classification_training' 来训练并保存模型。")
        # 可以选择在这里调用训练，或者提示用户手动运行
        # from Classification.TrainModel import run_classification_training
        # print("尝试自动运行训练...")
        # run_classification_training(stock_code=stock_to_analyze)
        # print("训练完成，请重新运行特征重要性分析。")
    else:
        run_permutation_importance(stock_code=stock_to_analyze)