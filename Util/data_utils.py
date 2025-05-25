import pandas as pd

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def prepare_data_for_pytorch_regression(data_df, n_steps, pred_window,
                                        existing_feature_scaler=None,
                                        existing_target_scaler=None,
                                        feature_columns=None):
    """
    准备时间序列数据用于PyTorch回归模型。
    :param data_df: 输入的DataFrame，需要包含 'close' 列以及其他特征列。
    :param n_steps: 用作输入序列的时间步长 (例如，过去30天的数据)。
    :param pred_window: 预测未来多少天后的收盘价。
    :param existing_feature_scaler: (可选) 预先拟合的特征归一化器。
    :param existing_target_scaler: (可选) 预先拟合的目标归一化器。
    :param feature_columns: (可选) 特征列名列表，默认为基础行情数据。
    :return: X (特征张量), y (目标张量), feature_scaler, target_scaler。如果数据不足，返回 None, None, None, None。
    """
    data_df_copy = data_df.copy()

    # 1. 数据质量验证
    if 'close' in data_df_copy.columns:
        invalid_prices = data_df_copy['close'] <= 0
        if invalid_prices.any():
            print(f"警告: 发现 {invalid_prices.sum()} 条无效的收盘价（<=0）")
            data_df_copy.loc[invalid_prices, 'close'] = np.nan
    
    if 'volume' in data_df_copy.columns:
        invalid_volume = data_df_copy['volume'] < 0
        if invalid_volume.any():
            print(f"警告: 发现 {invalid_volume.sum()} 条无效的成交量（<0）")
            data_df_copy.loc[invalid_volume, 'volume'] = np.nan

    # 2. 设置默认特征列（如果未指定）
    if feature_columns is None:
        feature_columns = ['close', 'high', 'low', 'open', 'volume']
    
    # 3. 确保所需特征列存在
    if not all(col in data_df_copy.columns for col in feature_columns):
        print(f"警告: DataFrame缺少必要的列。需要: {feature_columns}。可用: {data_df_copy.columns.tolist()}")
        return None, None, None, None

    # 4. 创建目标变量: 预测 pred_window 天后的收盘价
    data_df_copy['target'] = data_df_copy['close'].shift(-pred_window)

    # 5. 移除所有包含NaN的行
    # 这会移除：
    # a) 原始数据中的无效值
    # b) shift操作产生的NaN（最后pred_window天）
    data_df_copy.dropna(inplace=True)

    if len(data_df_copy) < n_steps + 1:
        print(f"警告: 数据不足 (处理后 {len(data_df_copy)} 行) 以创建至少一个长度为 {n_steps} 的序列。")
        return None, None, None, None

    # 6. 提取特征和目标
    features = data_df_copy[feature_columns].values
    target = data_df_copy[['target']].values

    # 7. 数据归一化 (使用0-1范围)
    if existing_feature_scaler:
        feature_scaler = existing_feature_scaler
        scaled_features = feature_scaler.transform(features)
    else:
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(features)
    
    if existing_target_scaler:
        target_scaler = existing_target_scaler
        scaled_target = target_scaler.transform(target)
    else:
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_target = target_scaler.fit_transform(target)

    # 8. 创建时间序列样本
    X_list, y_list = [], []
    
    if len(scaled_features) < n_steps:
        print(f"警告: 归一化后的特征数据不足 ({len(scaled_features)} 行) 以创建长度为 {n_steps} 的序列。")
        return None, None, feature_scaler, target_scaler

    # 创建时间序列样本
    for i in range(n_steps, len(scaled_features)):
        X_list.append(scaled_features[i-n_steps:i, :])
        y_list.append(scaled_target[i, 0])
    
    if not X_list:
        print(f"警告: 未能从数据中创建任何时间序列样本。")
        return None, None, feature_scaler, target_scaler

    # 9. 转换为PyTorch张量
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    
    return X_tensor, y_tensor, feature_scaler, target_scaler

def prepare_data_for_pytorch_classification(data_df, n_steps, pred_window, 
                                          existing_feature_scaler=None, 
                                          feature_columns=None,
                                          price_increase_threshold=0.02):
    """
    准备时间序列数据用于PyTorch分类模型。
    目标为未来pred_window天后收盘价是否上涨（1）或下跌（0）。
    :param data_df: 输入的DataFrame，需要包含 'close' 列以及其他特征列。
    :param n_steps: 用作输入序列的时间步长。
    :param pred_window: 预测未来多少天后的涨跌。
    :param existing_feature_scaler: (可选) 预先拟合的特征归一化器。
    :param feature_columns: (可选) 特征列名列表，默认包含基础行情和常用技术指标。
    :param price_increase_threshold: 定义上涨的价格增长阈值，默认0.02（2%）
    :return: X (特征张量), y (标签张量), feature_scaler。如果数据不足，返回 None, None, None。
    """
    data_df_copy = data_df.copy()
    
    # 1. 数据质量验证
    if 'close' in data_df_copy.columns:
        invalid_prices = data_df_copy['close'] <= 0
        if invalid_prices.any():
            print(f"警告: 发现 {invalid_prices.sum()} 条无效的收盘价（<=0）")
            data_df_copy.loc[invalid_prices, 'close'] = np.nan
    
    if 'volume' in data_df_copy.columns:
        invalid_volume = data_df_copy['volume'] < 0
        if invalid_volume.any():
            print(f"警告: 发现 {invalid_volume.sum()} 条无效的成交量（<0）")
            data_df_copy.loc[invalid_volume, 'volume'] = np.nan

    # 2. 计算技术指标
    data_df_copy['ma5'] = data_df_copy['close'].rolling(window=5).mean()
    data_df_copy['ma10'] = data_df_copy['close'].rolling(window=10).mean()
    data_df_copy['rsi14'] = 100 - 100 / (1 + data_df_copy['close'].pct_change().rolling(14).mean())
    
    # 3. 设置默认特征列（如果未指定）
    if feature_columns is None:
        feature_columns = ['close', 'high', 'low', 'open', 'volume', 'ma5', 'ma10', 'rsi14']
        # feature_columns = ['close', 'high', 'low', 'open', 'volume']
    
    # 确保所需特征列存在
    if not all(col in data_df_copy.columns for col in feature_columns):
        print(f"警告: DataFrame缺少必要的列。需要: {feature_columns}。可用: {data_df_copy.columns.tolist()}")
        return None, None, None
    
    # 4. 计算未来最高价 (用于后续生成标签)
    # 修正：对于第i行，计算从第i+1行到第i+pred_window行的'close'价格中的最大值
    data_df_copy['future_max'] = data_df_copy['close'].rolling(window=pred_window, min_periods=1).max().shift(-pred_window)
    
    # 5. 移除所有包含NaN的行 (在生成最终标签之前)
    # 这会移除：
    # a) 原始数据中的无效值
    # b) 技术指标计算产生的NaN（序列开始的部分）
    # c) future_max计算产生的NaN（序列末尾的部分）
    data_df_copy.dropna(inplace=True) # 标签将在此操作之后从清理过的数据帧计算
    
    if len(data_df_copy) < n_steps + 1: # 长度检查也相应提前
        print(f"警告: 数据不足 (处理后 {len(data_df_copy)} 行) 以创建至少一个长度为 {n_steps} 的序列。")
        return None, None, None
    
    # 6. 从清理后的数据生成标签
    # 确保 'future_max' 和 'close' 列在 data_df_copy 中仍然存在且有效
    labels = (data_df_copy['future_max'] > data_df_copy['close'] * (1 + price_increase_threshold)).astype(int).values.reshape(-1, 1)
        
    # 7. 提取特征 (从同样清理过的数据)
    # 确保 feature_columns 中的所有列在 data_df_copy 中仍然存在且有效
    features = data_df_copy[feature_columns].values
    
    # 8. 归一化特征 (使用0-1范围)
    if existing_feature_scaler:
        feature_scaler = existing_feature_scaler
        scaled_features = feature_scaler.transform(features)
    else:
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(features)
    
    # 9. 创建时间序列样本
    X_list, y_list = [], []
    # scaled_features 和 labels 的行数此时应该是一致的
    # 因为它们都是从执行了 dropna() 后的 data_df_copy 派生出来的
    if len(scaled_features) < n_steps: 
        print(f"警告: 归一化后的特征数据不足 ({len(scaled_features)} 行) 以创建长度为 {n_steps} 的序列。")
        return None, None, feature_scaler

    for i in range(n_steps, len(scaled_features)):
        X_list.append(scaled_features[i-n_steps:i, :])
        y_list.append(labels[i, 0])

    if not X_list:
        print(f"警告: 未能从数据中创建任何时间序列样本。")
        return None, None, feature_scaler

    # 10. 转换为PyTorch张量
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    
    return X_tensor, y_tensor, feature_scaler
def prepare_data_for_mlp_three_class(data_df, n_steps, pred_window,
                                     existing_feature_scaler=None,
                                     feature_columns=None,
                                     up_threshold=0.02,
                                     down_threshold=-0.02):
    """
    准备时间序列数据用于PyTorch MLP三分类模型。
    目标: 0 for 跌, 1 for 平仓 (将在损失计算中被忽略), 2 for 涨.
    :param data_df: 输入的DataFrame，需要包含 'close' 列以及其他特征列。
    :param n_steps: 用作输入序列的时间步长 (回看天数)。
    :param pred_window: 预测未来多少天后的涨跌。
    :param existing_feature_scaler: (可选) 预先拟合的特征归一化器。
    :param feature_columns: (可选) 特征列名列表，默认包含基础行情和常用技术指标。
    :param up_threshold: 定义上涨的价格增长阈值 (例如 0.02 表示 +2%)。
    :param down_threshold: 定义下跌的价格下跌阈值 (例如 -0.02 表示 -2%)。
    :return: X (特征张量), y (标签张量, dtype=torch.long), feature_scaler。如果数据不足，返回 None, None, None。
    """
    data_df_copy = data_df.copy()

    # 1. 数据质量验证 和 早期NaN处理
    required_cols = ['close', 'high', 'low', 'open', 'volume']
    for col in required_cols:
        if col not in data_df_copy.columns:
            print(f"警告: DataFrame缺少基础列: {col}。")
            return None, None, None
            
    if 'close' in data_df_copy.columns:
        invalid_prices = data_df_copy['close'] <= 0
        if invalid_prices.any():
            print(f"警告: 发现 {invalid_prices.sum()} 条无效的收盘价（<=0），将被设为NaN。")
            data_df_copy.loc[invalid_prices, 'close'] = np.nan
    
    if 'volume' in data_df_copy.columns:
        invalid_volume = data_df_copy['volume'] < 0
        if invalid_volume.any():
            print(f"警告: 发现 {invalid_volume.sum()} 条无效的成交量（<0），将被设为NaN。")
            data_df_copy.loc[invalid_volume, 'volume'] = np.nan
    
    data_df_copy.dropna(subset=required_cols, inplace=True) # 移除基础数据中仍存在的NaN

    if data_df_copy.empty:
        print("警告: 基础数据清理后为空。")
        return None, None, None

    # 2. 计算技术指标
    data_df_copy['ma5'] = data_df_copy['close'].rolling(window=5, min_periods=1).mean()
    data_df_copy['ma10'] = data_df_copy['close'].rolling(window=10, min_periods=1).mean()
    
    # Robust RSI calculation
    delta = data_df_copy['close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    data_df_copy['rsi14'] = 100 - (100 / (1 + rs))
    # 处理 avg_loss == 0 的情况
    data_df_copy.loc[avg_loss == 0, 'rsi14'] = 100 # 如果 avg_gain > 0, RSI is 100
    data_df_copy.loc[(avg_gain == 0) & (avg_loss == 0), 'rsi14'] = 50 # 如果 gain 和 loss 都是0, RSI is neutral (e.g. 50)
    data_df_copy['rsi14'] = data_df_copy['rsi14'].fillna(50) # 填充RSI计算开始时可能产生的NaN

    # 3. 设置默认特征列
    if feature_columns is None:
        feature_columns = ['close', 'high', 'low', 'open', 'volume', 'ma5', 'ma10', 'rsi14']
    
    if not all(col in data_df_copy.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in data_df_copy.columns]
        print(f"警告: DataFrame缺少必要的特征列: {missing_cols}。可用: {data_df_copy.columns.tolist()}")
        return None, None, None

    # 4. 计算未来 pred_window 天后的收盘价 (用于生成标签)
    data_df_copy['future_target_close'] = data_df_copy['close'].shift(-pred_window)

    # 5. 移除所有包含NaN的行 (技术指标计算、future_target_close计算等产生的NaN)
    columns_to_check_for_nan = feature_columns + ['future_target_close']
    data_df_copy.dropna(subset=columns_to_check_for_nan, inplace=True)

    if len(data_df_copy) < n_steps : 
        print(f"警告: 数据不足 (处理后 {len(data_df_copy)} 行) 以创建至少一个长度为 {n_steps} 的序列。")
        return None, None, None

    # 6. 从清理后的数据生成标签 (三分类: 0=跌, 1=平仓, 2=涨)
    current_close = data_df_copy['close'].values
    future_close = data_df_copy['future_target_close'].values
    
    percentage_change = (future_close - current_close) / current_close
    
    labels_np = np.full(len(data_df_copy), 1, dtype=int) # 默认平仓 (label 1)
    labels_np[percentage_change < down_threshold] = 0  # 跌 (label 0)
    labels_np[percentage_change > up_threshold] = 2   # 涨 (label 2)
    
    # 7. 提取特征
    features_np = data_df_copy[feature_columns].values
    
    # 8. 归一化特征
    if existing_feature_scaler:
        feature_scaler = existing_feature_scaler
        scaled_features = feature_scaler.transform(features_np)
    else:
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(features_np)
    
    # 9. 创建时间序列样本
    # X_list 的每个元素是 (n_steps, num_features)
    # y_list 的每个元素是对应的标签
    # 遵循现有 prepare_data_for_pytorch_classification 的索引逻辑:
    # X_list.append(scaled_features[i-n_steps:i, :]) # 特征序列从 i-n_steps 到 i-1
    # y_list.append(labels_np[i]) # 标签对应于 scaled_features[i] 之后发生的事件
    
    X_list, y_list = [], []
    if len(scaled_features) <= n_steps: # 需要至少 n_steps+1 个点来形成一个序列和它之后的标签
        print(f"警告: 归一化后的特征数据不足 ({len(scaled_features)} 行) 以创建序列并获取其后的标签。至少需要 {n_steps+1} 行。")
        return None, None, feature_scaler

    # 循环从 n_steps 开始，确保 labels_np[i] 是有效的
    # scaled_features 和 labels_np 长度相同
    for i in range(n_steps, len(scaled_features)):
        X_list.append(scaled_features[i-n_steps:i, :])
        y_list.append(labels_np[i]) # labels_np[i] 是与 scaled_features[i] 相关联的未来事件的标签

    if not X_list:
        print(f"警告: 未能从数据中创建任何时间序列样本。")
        return None, None, feature_scaler

    # 10. 转换为PyTorch张量
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    # y_tensor 用于 CrossEntropyLoss，应为 Long 类型，形状为 (N)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.long) 
    
    return X_tensor, y_tensor, feature_scaler