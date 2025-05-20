import pandas as pd

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def prepare_data_for_pytorch_regression(data_df, n_steps, pred_window,
                                        existing_feature_scaler=None,
                                        existing_target_scaler=None):
    """
    准备时间序列数据用于PyTorch回归模型。
    :param data_df: 输入的DataFrame，需要包含 'close' 列以及其他特征列。
    :param n_steps: 用作输入序列的时间步长 (例如，过去30天的数据)。
    :param pred_window: 预测未来多少天后的收盘价。
    :param existing_feature_scaler: (可选) 预先拟合的特征归一化器。
    :param existing_target_scaler: (可选) 预先拟合的目标归一化器。
    :return: X (特征张量), y (目标张量), feature_scaler, target_scaler。如果数据不足，返回 None, None, None, None。
    """
    # 1. 创建目标变量: 预测 pred_window 天后的收盘价
    data_df_copy = data_df.copy() # 操作副本以避免 SettingWithCopyWarning
    data_df_copy['target'] = data_df_copy['close'].shift(-pred_window)
    data_df_copy.dropna(inplace=True) # 移除因shift产生的NaN值

    if len(data_df_copy) < n_steps + 1: # 需要至少 n_steps 用于输入，1 用于当前的目标点
        print(f"警告: 数据不足 (处理后 {len(data_df_copy)} 行) 以创建至少一个长度为 {n_steps} 的序列。")
        return None, None, None, None
    
    # 2. 选择特征列
    feature_columns = ['close', 'high', 'low', 'open', 'volume'] # 可以根据需要调整
    # 确保在尝试访问列之前，DataFrame不为空且包含这些列
    if not all(col in data_df_copy.columns for col in feature_columns + ['target']):
        print(f"警告: DataFrame缺少必要的列。需要: {feature_columns + ['target']}。可用: {data_df_copy.columns.tolist()}")
        return None, None, None, None
        
    features = data_df_copy[feature_columns].values
    target = data_df_copy[['target']].values

    # 3. 数据归一化
    if existing_feature_scaler:
        feature_scaler = existing_feature_scaler
        scaled_features = feature_scaler.transform(features)
    else:
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_features = feature_scaler.fit_transform(features)
    
    if existing_target_scaler:
        target_scaler = existing_target_scaler
        scaled_target = target_scaler.transform(target)
    else:
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_target = target_scaler.fit_transform(target)
    
    # 4. 创建时间序列样本
    X_list, y_list = [], []
    # 循环的上限应该是 len(scaled_features) 或 len(scaled_target)，它们应该是一样的
    # 确保 i 不会超出 scaled_target 的范围，因为 y_list.append(scaled_target[i, 0])
    # 实际上，由于target是基于features[i]时间点的值，所以循环到len(scaled_features)即可
    # 而scaled_target的长度应该与scaled_features相同
    
    # 确保在创建序列之前，我们有足够的数据点
    # scaled_features 的行数必须至少为 n_steps 才能创建第一个序列
    # 循环从 n_steps 开始，到 len(scaled_features) -1 结束（因为索引是 i-n_steps 到 i-1 作为X, i 作为y）
    # 所以，len(scaled_features) 必须至少是 n_steps。
    # 如果 len(scaled_features) == n_steps, 循环 for i in range(n_steps, n_steps) 不会执行。
    # 应该是 for i in range(n_steps -1, len(scaled_features) -1) 如果y是 target[i+1]
    # 或者 for i in range(n_steps, len(scaled_features)+1) 如果y是 target[i-1]
    # 当前逻辑: X是 [i-n_steps:i], y是 [i,0] (scaled_target的第i行)
    # 这意味着当 i = n_steps 时, X是 [0:n_steps], y是 scaled_target[n_steps,0]
    # 循环的最后一次 i = len(scaled_features)-1, X是 [len-1-n_steps : len-1], y是 scaled_target[len-1,0]
    # 所以，scaled_features 和 scaled_target 的长度必须至少是 n_steps + 1 才能让循环至少跑一次（当i=n_steps时）
    # 但实际上，只要 len(scaled_features) > n_steps 即可。
    # 如果 len(scaled_features) == n_steps, range(n_steps, n_steps) 为空。
    # 如果 len(scaled_features) == n_steps + 1, range(n_steps, n_steps+1) -> i = n_steps.
    
    if len(scaled_features) < n_steps: # 严格来说，如果等于n_steps，也无法形成序列y[i]
        print(f"警告: 归一化后的特征数据不足 ({len(scaled_features)} 行) 以创建长度为 {n_steps} 的序列。")
        return None, None, feature_scaler, target_scaler # 返回scaler，即使没有数据

    for i in range(n_steps, len(scaled_features)):
        X_list.append(scaled_features[i-n_steps:i, :])
        y_list.append(scaled_target[i, 0])
    
    if not X_list: # 如果没有生成任何序列
        print(f"警告: 未能从数据中创建任何时间序列样本。")
        return None, None, feature_scaler, target_scaler

    # 转换为PyTorch张量
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    
    return X_tensor, y_tensor, feature_scaler, target_scaler