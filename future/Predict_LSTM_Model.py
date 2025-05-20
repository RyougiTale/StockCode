import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset # 添加导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib # 用于加载scaler
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



# 本地模块导入
from StockDataLoader import StockDataLoader
from lstm_model import LSTMRegressor
from config import DATA_CONFIG, PREDICTION_CONFIG

def generate_full_predictions( # 函数更名并移除绘图相关参数
    trained_model,
    original_df, # 完整的原始数据帧
    feature_scaler,
    target_scaler,
    n_past_days, # 形成一个序列所需的过去天数
    prediction_window, # 模型预测未来多少天
    device,
    stock_code, # 保留stock_code用于可能的打印信息
    batch_size=64 # 新增batch_size参数
):
    """
    在整个数据集上为给定模型生成预测值和对应日期。
    不再直接绘图或保存。
    """
    print(f"\n正在为模型针对 {stock_code} 的完整数据集生成预测...")
    trained_model.eval()
    
    data_for_pred = original_df.copy()
    
    # 1. 特征选择和归一化 (使用已加载和拟合的scaler)
    feature_columns = ['close', 'high', 'low', 'open', 'volume']
    if not all(col in data_for_pred.columns for col in feature_columns):
        print(f"错误: 原始数据 {stock_code} 中缺少必要的特征列: {feature_columns}")
        return None, None # 返回 None 表示绘图或预测失败
            
    features_full = data_for_pred[feature_columns].values
    try:
        scaled_features_full = feature_scaler.transform(features_full)
    except Exception as e:
        print(f"错误: 使用feature_scaler转换 {stock_code} 完整数据集特征时出错: {e}")
        print(f"Feature_scaler期望的特征数量: {feature_scaler.n_features_in_ if hasattr(feature_scaler, 'n_features_in_') else '未知'}")
        print(f"提供的特征形状: {features_full.shape}")
        return None, None # 返回 None 表示绘图或预测失败

    # 2. 创建输入序列 X_full 和对应的输入序列结束日期
    X_full_list = []
    input_sequence_end_dates = [] 
    
    if len(scaled_features_full) < n_past_days:
        print(f"数据长度 ({len(scaled_features_full)}) 小于 n_past_days ({n_past_days})，无法为 {stock_code} 生成序列。")
        return None, None # 返回 None 表示绘图或预测失败

    for i in range(n_past_days, len(scaled_features_full) + 1):
        X_full_list.append(scaled_features_full[i-n_past_days:i, :])
        # 确保日期列存在且名为 'date'，并且是 datetime 类型
        if 'date' not in data_for_pred.columns or not pd.api.types.is_datetime64_any_dtype(data_for_pred['date']):
            print(f"错误: {stock_code} 的 data_for_pred DataFrame 中缺少 'date' 列或其不是datetime类型。")
            # 可以选择创建一个从0开始的索引作为日期替代，或者直接返回
            # data_for_pred['date'] = pd.to_datetime(data_for_pred.index, unit='D') # 示例替代
            return None, None # 返回 None 表示绘图或预测失败
        input_sequence_end_dates.append(data_for_pred['date'].iloc[i-1]) # 序列的最后一天
            
    if not X_full_list:
        print(f"未能为 {stock_code} 的完整数据集生成任何输入序列。")
        return None, None # 返回 None 表示绘图或预测失败
            
    X_full_tensor = torch.tensor(np.array(X_full_list), dtype=torch.float32) # 先不移到device，按批次移动
    
    # 3. 模型预测 (批处理)
    full_predictions_normalized = []
    with torch.no_grad():
        for i in range(0, len(X_full_tensor), batch_size):
            batch_X = X_full_tensor[i:i+batch_size].to(device)
            outputs_batch = trained_model(batch_X)
            full_predictions_normalized.extend(outputs_batch.cpu().numpy())
            
    # 4. 反归一化
    full_predictions_np = np.array(full_predictions_normalized)
    if full_predictions_np.ndim == 1:
        full_predictions_np = full_predictions_np.reshape(-1, 1)
    
    actual_price_predictions_full = target_scaler.inverse_transform(full_predictions_np)
    
    # 5. 计算预测对应的未来日期
    # 确保 input_sequence_end_dates 中的日期是 pandas Timestamp 对象
    predicted_dates_for_plot = [
        (d + pd.Timedelta(days=prediction_window)) if pd.api.types.is_datetime64_any_dtype(d) else pd.to_datetime(d) + pd.Timedelta(days=prediction_window)
        for d in input_sequence_end_dates
    ]
    
    # 6. 移除可视化部分，只返回数据
    # print(f"预测生成完毕，准备返回价格和日期。") # 可选的调试信息
    
    # 返回预测的价格数组和对应的日期列表
    return actual_price_predictions_full.flatten(), predicted_dates_for_plot
    # return actual_price_predictions_full.flatten(), input_sequence_end_dates


def run_prediction_evaluation(stock_code=None, # 将从DATA_CONFIG获取默认值
                              model_dir='./models/pytorch_lstm',
                              results_save_dir='./results/pytorch_lstm'):
    """
    执行模型加载、预测、评估和可视化的完整流程。
    评估数据的时间范围将从 DATA_CONFIG 中的 eval_start_date 和 eval_end_date 读取。
    """
    if stock_code is None:
        raise ValueError("stock_code is required")
        # stock_code = DATA_CONFIG['default_stock_code']


    print(f"开始为股票 {stock_code} 执行PyTorch LSTM模型预测与评估流程...")
    os.makedirs(results_save_dir, exist_ok=True)

    # 从 DATA_CONFIG 获取评估数据的日期范围
    eval_start_date = DATA_CONFIG['eval_start_date']
    eval_end_date = DATA_CONFIG['eval_end_date']

    if not eval_start_date or not eval_end_date:
        print("错误: DATA_CONFIG 中未定义 'eval_start_date' 或 'eval_end_date'。评估中止。")
        return
    
    print(f"评估数据（测试集）将从 {eval_start_date} 加载到 {eval_end_date}。")

    # 1. 定义模型和Scaler路径
    model_train_best_path = os.path.join(model_dir, f'{stock_code}_train_best_lstm_model.pth')
    model_val_best_path = os.path.join(model_dir, f'{stock_code}_val_best_lstm_model.pth')
    feature_scaler_path = os.path.join(model_dir, f'{stock_code}_feature_scaler.pkl')
    target_scaler_path = os.path.join(model_dir, f'{stock_code}_target_scaler.pkl')

    if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
        print(f"错误: Scaler文件未找到于 {model_dir}。请先运行训练流程。")
        return
    if not os.path.exists(model_train_best_path):
        print(f"错误: 基于训练集最优的模型 {model_train_best_path} 未找到。请先运行训练流程。")
        return
    if not os.path.exists(model_val_best_path):
        print(f"错误: 基于验证集最优的模型 {model_val_best_path} 未找到。请先运行训练流程。")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    print("Feature scaler 和 Target scaler 已加载。")

    # 使用加载的 feature_scaler 确定 input_dim
    try:
        input_dim_model = feature_scaler.n_features_in_
    except AttributeError:
        print("警告: feature_scaler 没有 n_features_in_ 属性。可能scaler未正确拟合或版本问题。")
        print("将尝试使用默认特征数 5 作为 input_dim。")
        # 尝试从 scaler 的其他属性推断，或者使用默认值
        if hasattr(feature_scaler, 'scale_'):
            print(feature_scaler.scale_)
            input_dim_model = len(feature_scaler.scale_)
        else:
            input_dim_model = 5 # 回退到默认值
    
    print(f"模型输入维度 (input_dim) 设置为: {input_dim_model}")
    
    # 获取模型结构参数 (hidden_dim, layer_dim, output_dim, dropout)
    hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
    layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
    output_dim = 1
    dropout = PREDICTION_CONFIG['lstm_dropout']

    # 加载基于训练集最优的模型
    model_train_best = LSTMRegressor(input_dim_model, hidden_dim, layer_dim, output_dim, dropout_prob=dropout)
    model_train_best.load_state_dict(torch.load(model_train_best_path, map_location=device))
    model_train_best.to(device)
    model_train_best.eval()
    print(f"模型 {model_train_best_path} (训练集最优) 已加载到设备: {device}")

    # 加载基于验证集最优的模型
    model_val_best = LSTMRegressor(input_dim_model, hidden_dim, layer_dim, output_dim, dropout_prob=dropout)
    model_val_best.load_state_dict(torch.load(model_val_best_path, map_location=device))
    model_val_best.to(device)
    model_val_best.eval()
    print(f"模型 {model_val_best_path} (验证集最优) 已加载到设备: {device}")

    # 2. 加载和准备评估数据 (测试集)
    print(f"加载评估数据 (测试集) {stock_code} 从 {eval_start_date} 到 {eval_end_date}...")
    eval_loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    eval_stock_data_df = eval_loader.load_stock_data(stock_code, eval_start_date, eval_end_date)
    if eval_stock_data_df is None or eval_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的评估数据。预测中止。")
        return
    
        print(f"未能加载股票 {stock_code} 的评估数据。预测中止。")
        return

    # 这些参数将传递给核心预测与绘图函数
    prediction_window = PREDICTION_CONFIG['prediction_days']
    n_past_days = PREDICTION_CONFIG['n_past_days_debug'] # 应与训练时一致
    inference_batch_size = PREDICTION_CONFIG.get('inference_batch_size', PREDICTION_CONFIG.get('batch_size', 64)) # 优先使用inference_batch_size，否则回退到训练的batch_size

    # 3. 为两个模型生成预测
    # print("\n--- 为基于训练集最优的模型生成预测 ---")
    # predicted_prices_train_best, predicted_dates_train_best = generate_full_predictions(
    #     model_train_best,
    #     eval_stock_data_df.copy(),
    #     feature_scaler,
    #     target_scaler,
    #     n_past_days,
    #     prediction_window,
    #     device,
    #     stock_code,
    #     batch_size=inference_batch_size
    # )

    print("\n--- 为基于验证集最优的模型生成预测 ---")
    predicted_prices_val_best, predicted_dates_val_best = generate_full_predictions(
        model_val_best,
        eval_stock_data_df.copy(),
        feature_scaler,
        target_scaler,
        n_past_days,
        prediction_window,
        device,
        stock_code,
        batch_size=inference_batch_size
    )

    # 4. 整合绘图：实际价格 + 两条预测线
    plt.figure(figsize=(18, 9))
    # 绘制实际历史收盘价
    if 'date' in eval_stock_data_df.columns and pd.api.types.is_datetime64_any_dtype(eval_stock_data_df['date']):
        plt.plot(eval_stock_data_df['date'], eval_stock_data_df['close'], label='实际历史收盘价', color='black', alpha=0.7, linewidth=1.5)
    else:
        plt.plot(eval_stock_data_df.index, eval_stock_data_df['close'], label='实际历史收盘价', color='black', alpha=0.7, linewidth=1.5)

    # 绘制训练集最优模型的预测
    # if predicted_prices_train_best is not None and predicted_dates_train_best is not None:
    #     plt.plot(predicted_dates_train_best, predicted_prices_train_best,
    #              label=f'预测 (训练集最优模型)', color='blue', linestyle='--', marker='o', markersize=3, alpha=0.8)

    # 绘制验证集最优模型的预测
    if predicted_prices_val_best is not None and predicted_dates_val_best is not None:
        plt.plot(predicted_dates_val_best, predicted_prices_val_best,
                 label=f'预测 (验证集最优模型)', color='red', linestyle=':', marker='x', markersize=3, alpha=0.8)
    
    title = f'{stock_code} - 模型预测对比 (未来{prediction_window}天评估期)'
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('股价')
    plt.legend()
    plt.grid(True)
    
    # 保存和显示图表
    plot_filename = os.path.join(results_save_dir, f"{stock_code}_eval_predictions_train_vs_val_{prediction_window}d.png")
    plt.savefig(plot_filename)
    print(f"\n评估对比图已保存到: {plot_filename}")
    plt.show()
    plt.close()

    # 5. 计算并报告两个模型的RMSE
    def calculate_rmse_for_predictions(predictions, dates, actual_df, stock_code_str, model_type_str):
        if predictions is not None and dates is not None and len(predictions) > 0:
            if 'date' not in actual_df.columns or not pd.api.types.is_datetime64_any_dtype(actual_df['date']):
                print(f"错误: {model_type_str} - 评估数据 {stock_code_str} 中缺少 'date' 列或其不是datetime类型，无法计算RMSE。")
                return
            
            actual_df_for_rmse = actual_df.set_index('date')
            valid_preds_rmse = []
            valid_actuals_rmse = []
            for p_date, p_price in zip(dates, predictions):
                current_pred_date_ts = pd.Timestamp(p_date)
                if current_pred_date_ts in actual_df_for_rmse.index:
                    actual_price = actual_df_for_rmse.loc[current_pred_date_ts, 'close']
                    valid_preds_rmse.append(p_price)
                    valid_actuals_rmse.append(actual_price)
            
            if len(valid_actuals_rmse) > 0 and len(valid_preds_rmse) == len(valid_actuals_rmse):
                rmse = np.sqrt(np.mean((np.array(valid_preds_rmse) - np.array(valid_actuals_rmse))**2))
                print(f"评估集RMSE ({model_type_str}模型) for {stock_code_str}: {rmse:.4f}")
            elif len(valid_actuals_rmse) == 0:
                 print(f"没有找到与预测日期对齐的实际数据点为 {stock_code_str} ({model_type_str}模型) 计算RMSE。")
            else:
                print(f"警告: 为 {stock_code_str} ({model_type_str}模型) 计算RMSE时，对齐的预测与实际数据点数量不匹配。")
        else:
            print(f"未能获取 {model_type_str} 模型的预测价格用于 {stock_code_str} 的RMSE计算，或预测序列为空。")

    print("\n--- RMSE 计算 ---")
    # calculate_rmse_for_predictions(predicted_prices_train_best, predicted_dates_train_best, eval_stock_data_df, stock_code, "训练集最优")
    calculate_rmse_for_predictions(predicted_prices_val_best, predicted_dates_val_best, eval_stock_data_df, stock_code, "验证集最优")
    
    print(f"\nPyTorch LSTM模型预测与评估流程 for {stock_code} 结束。")


if __name__ == "__main__":
    stock_to_predict = '600036'
    # 假设模型已在 './models/pytorch_lstm' 训练并保存
    # 评估时，我们可以选择与训练时不同的日期范围，或者使用训练时的测试集部分对应的日期范围
    # eval_start_date 和 eval_end_date 将从 DATA_CONFIG 内部读取
    # 因此调用时不再需要传递日期参数

    run_prediction_evaluation(stock_code=stock_to_predict)