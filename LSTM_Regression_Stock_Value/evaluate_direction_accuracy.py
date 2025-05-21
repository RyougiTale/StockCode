import os
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# 相对导入，因为这个新脚本也在 LSTM_Regression_Stock_Value 包内
from .config import DATA_CONFIG, PREDICTION_CONFIG # 假设模型参数等在config中
from .lstm_model import LSTMRegressor

# 从Util导入共享模块
from Util.StockDataLoader import StockDataLoader
from Util.data_utils import prepare_data_for_pytorch_regression

# 尝试从config.py获取模型保存目录，如果不存在则使用默认值
try:
    from .config import MODEL_SAVE_DIR
except ImportError:
    MODEL_SAVE_DIR = './models/pytorch_lstm' # 与Train_LSTM_Model.py中的默认值一致

def get_base_price_for_prediction(X_sample_scaled, feature_scaler, close_feature_index=3):
    """
    从归一化的X样本中提取并反归一化最后一个时间步的收盘价作为基准价格。
    假设收盘价是X的特征之一。
    Args:
        X_sample_scaled (torch.Tensor): 单个归一化的X样本，形状为 (sequence_length, num_features)
        feature_scaler (sklearn.preprocessing.Scaler): 用于特征的scaler
        close_feature_index (int): 收盘价在特征中的索引 (0-indexed).
                                   根据 prepare_data_for_pytorch_regression, 特征通常是
                                   ['open', 'high', 'low', 'close', 'volume', ...], close通常是第3个索引 (即index 3)
                                   如果特征顺序不同，需要调整此索引。
    Returns:
        float: 反归一化后的基准收盘价
    """
    # 获取最后一个时间步的所有特征
    last_step_features_scaled = X_sample_scaled[-1, :].cpu().numpy().reshape(1, -1) # (1, num_features)
    
    # 创建一个与scaler期望的输入形状相同的虚拟数组
    # scaler.transform期望的是 (n_samples, n_features)
    # scaler.inverse_transform也期望 (n_samples, n_features)
    # 我们只反归一化收盘价，但scaler是针对所有特征训练的
    # 为了正确反归一化单个特征，通常需要将其放回原始的多特征结构中（用虚拟值填充其他特征）
    # 或者，如果scaler支持针对特定列操作，或者我们有单独的收盘价scaler（不常见）
    
    # 简单的方法：如果feature_scaler是MinMaxScaler，并且我们知道min_和scale_
    # close_scaled_value = last_step_features_scaled[0, close_feature_index]
    # close_original_value = (close_scaled_value * feature_scaler.data_range_[close_feature_index]) + feature_scaler.min_[close_feature_index]
    # 这是针对MinMaxScaler的特定计算，更通用的方法是使用inverse_transform

    # 通用方法：反归一化整个特征向量，然后提取收盘价
    # 注意：这假设 feature_scaler.inverse_transform 可以正确处理我们这样构造的输入
    # 实际上，feature_scaler通常在fit时看到的是 (num_samples, num_features)
    # inverse_transform也期望 (num_samples, num_features)
    original_features_shape = last_step_features_scaled.shape # (1, num_features)
    
    # 为了使用scaler的inverse_transform，我们需要一个包含所有特征的向量
    # 我们已经有了 last_step_features_scaled
    unscaled_features = feature_scaler.inverse_transform(last_step_features_scaled)
    base_price = unscaled_features[0, close_feature_index]
    return base_price


def evaluate_model_direction_accuracy(
    stock_code, 
    test_start_date, 
    test_end_date, 
    model_filename, # 例如 '600036_val_best_lstm_model.pth'
    feature_scaler_filename, # 例如 '600036_feature_scaler.pkl'
    target_scaler_filename, # 例如 '600036_target_scaler.pkl'
    close_feature_index=3, # 收盘价在X特征中的索引
    target_is_single_value=True # 假设y (目标) 是单个值 (如未来第N天收盘价)
):
    """
    评估模型预测股票涨跌方向的准确率。
    """
    print(f"开始评估股票 {stock_code} 的涨跌方向预测准确率...")
    print(f"测试数据周期: {test_start_date} 到 {test_end_date}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型和scalers
    model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    feature_scaler_path = os.path.join(MODEL_SAVE_DIR, feature_scaler_filename)
    target_scaler_path = os.path.join(MODEL_SAVE_DIR, target_scaler_filename)

    if not all(map(os.path.exists, [model_path, feature_scaler_path, target_scaler_path])):
        print("错误：模型、特征缩放器或目标缩放器文件未找到。请确保已成功训练并保存。")
        print(f"检查路径: {model_path}, {feature_scaler_path}, {target_scaler_path}")
        return

    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    
    # 动态获取模型的input_dim, hidden_dim等参数有点麻烦，
    # 通常这些参数在训练时确定并可能保存在某处，或在加载模型结构时不需要显式传入
    # LSTMRegressor的定义需要这些参数。我们假设config中的参数与保存的模型一致。
    input_dim = feature_scaler.n_features_in_ # 从scaler获取输入特征数量
    hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
    layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
    # output_dim 应该与y的形状匹配。如果y是单个值，output_dim=1
    # 如果y是序列，output_dim=prediction_window (如果模型预测整个序列)
    # 假设模型输出与target_scaler处理的目标一致
    if target_is_single_value:
        output_dim = 1 
    else:
        # 如果目标是一个序列，需要确定其维度
        # 假设target_scaler.transform(y.reshape(-1,1))，那么output_dim还是1，但模型结构可能不同
        # 或者如果模型直接输出 (batch, seq_len_out, features_out)
        # 为简单起见，我们假设目标是单个值，因此output_dim=1
        output_dim = 1 # 需要根据实际模型输出调整
        if hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ > 1:
            # 这暗示目标可能是多维的，或者scaler被fit到了多维数据上
            # 但通常回归任务的目标scaler是针对单列的
            pass
        elif hasattr(target_scaler, 'scale_') and len(target_scaler.scale_) > 1:
             print(f"警告: target_scaler似乎是为多维目标fit的 (scale_ len: {len(target_scaler.scale_)}). output_dim可能需要调整。")
             # output_dim = len(target_scaler.scale_) # 这可能不正确，取决于模型如何设计

    dropout = PREDICTION_CONFIG['lstm_dropout']
    
    model = LSTMRegressor(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型及Scalers加载完成。模型移至: {device}")

    # 2. 加载并准备测试数据
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    print(f"加载测试原始数据...")
    test_stock_data_df = loader.load_stock_data(stock_code, test_start_date, test_end_date)
    if test_stock_data_df is None or test_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 在 {test_start_date} 到 {test_end_date} 的测试数据。评估中止。")
        return
    print(f"成功加载 {len(test_stock_data_df)} 条测试原始数据。")

    n_past_days = PREDICTION_CONFIG['n_past_days_debug'] # 使用与训练时相同的n_past_days_debug
    prediction_window = PREDICTION_CONFIG['prediction_days'] # 目标是预测这么多天之后的情况

    print(f"准备测试数据 (使用已加载的归一化器)...")
    # 使用 existing_scalers，不fit新的
    X_test_scaled, y_test_scaled, _, _ = prepare_data_for_pytorch_regression(
        test_stock_data_df.copy(),
        n_past_days,
        prediction_window,
        existing_feature_scaler=feature_scaler,
        existing_target_scaler=target_scaler
    )

    if X_test_scaled is None or y_test_scaled is None or X_test_scaled.shape[0] == 0:
        print("测试数据准备失败（可能数据不足或与训练数据特征不一致）。评估中止。")
        return
    print(f"测试数据准备完成。X_test_scaled shape: {X_test_scaled.shape}, y_test_scaled shape: {y_test_scaled.shape}")

    # 3. 进行预测并反归一化
    actual_target_prices = target_scaler.inverse_transform(y_test_scaled.cpu().numpy()) # (num_samples, 1)

    predicted_target_prices_scaled_list = []
    with torch.no_grad():
        test_pbar = tqdm(range(X_test_scaled.shape[0]), desc="模型预测中")
        for i in test_pbar:
            sample_X = X_test_scaled[i, :, :].unsqueeze(0).to(device) # (1, seq_len, num_features)
            prediction_scaled = model(sample_X) # (1, output_dim)
            predicted_target_prices_scaled_list.append(prediction_scaled.cpu().numpy())

    predicted_target_prices_scaled = np.concatenate(predicted_target_prices_scaled_list, axis=0) # (num_samples, output_dim)
    predicted_target_prices = target_scaler.inverse_transform(predicted_target_prices_scaled) # (num_samples, 1)

    # 4. 获取基准价格 (每个预测窗口开始前的最后一个已知价格)
    base_prices = []
    # X_test_scaled 的每个样本 X_test_scaled[i] 对应一个预测 y_test_scaled[i]
    # 我们需要 X_test_scaled[i] 的最后一个时间步的收盘价作为基准
    print("提取基准价格...")
    for i in tqdm(range(X_test_scaled.shape[0]), desc="提取基准价格"):
        # X_test_scaled[i] 的形状是 (n_past_days, num_features)
        base_price = get_base_price_for_prediction(X_test_scaled[i], feature_scaler, close_feature_index)
        base_prices.append(base_price)
    base_prices = np.array(base_prices).reshape(-1, 1) # (num_samples, 1)

    if not (len(actual_target_prices) == len(predicted_target_prices) == len(base_prices)):
        print("错误：实际价格、预测价格和基准价格的样本数量不匹配。")
        print(f"Actuals: {len(actual_target_prices)}, Predicted: {len(predicted_target_prices)}, Bases: {len(base_prices)}")
        return

    # 5. 计算涨跌方向并比较
    actual_diff = actual_target_prices - base_prices
    predicted_diff = predicted_target_prices - base_prices

    # np.sign: 1 for positive, -1 for negative, 0 for zero
    actual_signs = np.sign(actual_diff)
    predicted_signs = np.sign(predicted_diff)
    
    # 过滤掉实际涨跌为0的情况，因为方向不明确 (或者根据需求定义)
    # valid_indices = actual_signs != 0
    # actual_signs = actual_signs[valid_indices]
    # predicted_signs = predicted_signs[valid_indices]
    # if len(actual_signs) == 0:
    #     print("没有有效的非零实际涨跌样本可供比较。")
    #     return

    correct_direction_predictions = np.sum(actual_signs == predicted_signs)
    total_comparisons = len(actual_signs)
    
    accuracy = 0
    if total_comparisons > 0:
        accuracy = correct_direction_predictions / total_comparisons
        print(f"总比较样本数: {total_comparisons}")
        print(f"方向预测正确数: {correct_direction_predictions}")
        print(f"涨跌方向预测准确率: {accuracy:.2%}")
    else:
        print("没有样本可供比较涨跌方向。")

    # 进一步分析：
    # 实际涨，预测也涨
    true_positives = np.sum((actual_signs > 0) & (predicted_signs > 0))
    # 实际跌，预测也跌
    true_negatives = np.sum((actual_signs < 0) & (predicted_signs < 0))
    # 实际平，预测也平 (如果符号包含0)
    true_zeros = np.sum((actual_signs == 0) & (predicted_signs == 0))
    
    print(f"  其中，实际为涨且预测为涨: {true_positives}")
    print(f"  其中，实际为跌且预测为跌: {true_negatives}")
    if np.any(actual_signs == 0) or np.any(predicted_signs == 0):
        print(f"  其中，实际为平且预测为平: {true_zeros}")
    
    # 用户要求增加的统计
    actual_up_predicted_down = np.sum((actual_signs > 0) & (predicted_signs < 0))
    actual_down_predicted_up = np.sum((actual_signs < 0) & (predicted_signs > 0))
    print(f"  其中，实际为涨且预测为跌: {actual_up_predicted_down}")
    print(f"  其中，实际为跌且预测为涨: {actual_down_predicted_up}")

    return accuracy


if __name__ == "__main__":
    stock_code_to_test = DATA_CONFIG['default_stock_code']
    
    # === 配置测试参数 ===
    # 您可能需要根据实际情况调整这些文件名和日期
    # 确保这些文件是您希望评估的训练运行所产生的
    model_to_evaluate_filename = f"{stock_code_to_test}_val_best_lstm_model.pth" 
    # model_to_evaluate_filename = f"{stock_code_to_test}_train_best_lstm_model.pth" # 或者用训练集上最好的模型

    feature_scaler_to_use_filename = f"{stock_code_to_test}_feature_scaler.pkl"
    target_scaler_to_use_filename = f"{stock_code_to_test}_target_scaler.pkl"

    # 定义测试集的时间范围 (从config中读取)
    test_period_start_date = DATA_CONFIG['eval_start_date']
    test_period_end_date = DATA_CONFIG['eval_end_date']

    # 检查 PREDICTION_CONFIG 中是否有 'features' 列表以确定 close_feature_index
    # 默认特征顺序: ['open', 'high', 'low', 'close', 'volume'] (来自旧的data_utils)
    # 如果您的 prepare_data_for_pytorch_regression 使用了不同的特征或顺序，需要调整
    # 例如，如果 PREDICTION_CONFIG['feature_columns'] = ['close', 'open', 'volume']，那么 close_feature_index = 0
    feature_columns_used = PREDICTION_CONFIG['feature_columns']
    try:
        # 假设 'close' 是用于计算基准的价格列
        # 注意：这里的 'close' 是指原始DataFrame中的列名，用于确定其在特征工程后的X中的索引
        # 如果 'close' 不是特征之一，或者基准应该是其他值，需要修改逻辑
        close_col_name_for_base = 'close' 
        if close_col_name_for_base in feature_columns_used:
            idx_close_in_features = feature_columns_used.index(close_col_name_for_base)
        else:
            print(f"警告: 基准列 '{close_col_name_for_base}' 不在特征列表 {feature_columns_used} 中。将使用默认索引3。")
            idx_close_in_features = 3 # 默认 'close' 是第4个特征 (index 3)
    except ValueError:
        print(f"警告: 基准列 '{close_col_name_for_base}' 在特征列表 {feature_columns_used} 中未找到。将使用默认索引3。")
        idx_close_in_features = 3
    
    print(f"使用的股票代码: {stock_code_to_test}")
    print(f"评估模型文件: {model_to_evaluate_filename}")
    print(f"特征Scaler文件: {feature_scaler_to_use_filename}")
    print(f"目标Scaler文件: {target_scaler_to_use_filename}")
    print(f"测试开始日期: {test_period_start_date}")
    print(f"测试结束日期: {test_period_end_date}")
    print(f"收盘价在特征中的索引 (用于基准价): {idx_close_in_features}")

    evaluate_model_direction_accuracy(
        stock_code=stock_code_to_test,
        test_start_date=test_period_start_date,
        test_end_date=test_period_end_date,
        model_filename=model_to_evaluate_filename,
        feature_scaler_filename=feature_scaler_to_use_filename,
        target_scaler_filename=target_scaler_to_use_filename,
        close_feature_index=idx_close_in_features
    )