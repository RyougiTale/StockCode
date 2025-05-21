import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# 调整sys.path以允许从父目录导入模块
# 获取当前脚本文件所在的目录 (trading_strategies)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (项目根目录)
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from StockDataLoader import StockDataLoader
from lstm_model import LSTMRegressor
from Predict_LSTM_Model import generate_full_predictions # 用于获取预测
from config import DATA_CONFIG, PREDICTION_CONFIG

def run_backtest(
    stock_code=None, # 将从DATA_CONFIG获取默认值
    initial_capital=1000000.0,
    model_type='val_best', # 'val_best' 或 'train_best'
    model_dir='models/pytorch_lstm', # 路径相对于项目根目录 (CWD)
    results_save_dir='results/pytorch_lstm', # 路径相对于项目根目录 (CWD)
    buy_threshold_factor=1.02, # 预测价比当前价高2%则考虑买入 (示例调整)
    sell_threshold_factor=0.98 # 预测价比当前价低2%则考虑卖出 (示例调整)
):
    """
    执行基于LSTM模型的简单交易策略回测。
    回测数据的时间范围将从 DATA_CONFIG 中的 eval_start_date 和 eval_end_date 读取。
    """
    if stock_code is None:
        stock_code = DATA_CONFIG.get('default_stock_code', '600036')

    print(f"--- 开始对股票 {stock_code} 执行交易策略回测 ---")
    print(f"初始资金: {initial_capital:,.2f}")
    
    # 从 DATA_CONFIG 获取回测数据的日期范围 (即测试集)
    backtest_start_date = DATA_CONFIG.get('eval_start_date')
    backtest_end_date = DATA_CONFIG.get('eval_end_date')

    if not backtest_start_date or not backtest_end_date:
        print("错误: DATA_CONFIG 中未定义 'eval_start_date' 或 'eval_end_date'。回测中止。")
        return

    print(f"回测时段 (测试集): {backtest_start_date} 到 {backtest_end_date}")
    print(f"使用模型类型: {model_type}")

    os.makedirs(results_save_dir, exist_ok=True)

    # --- 1. 加载模型和Scalers ---
    if model_type == 'val_best':
        model_filename = f'{stock_code}_val_best_lstm_model.pth'
    elif model_type == 'train_best':
        model_filename = f'{stock_code}_train_best_lstm_model.pth'
    else:
        print(f"错误: 无效的模型类型 '{model_type}'。请选择 'val_best' 或 'train_best'。")
        return

    model_path = os.path.join(model_dir, model_filename)
    feature_scaler_path = os.path.join(model_dir, f'{stock_code}_feature_scaler.pkl')
    target_scaler_path = os.path.join(model_dir, f'{stock_code}_target_scaler.pkl')

    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 未找到。")
        return
    if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
        print(f"错误: Scaler文件未找到于 {model_dir}。")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    print("Scalers已加载。")

    # 使用加载的 feature_scaler 确定 input_dim
    try:
        input_dim_model = feature_scaler.n_features_in_
    except AttributeError:
        print("警告: feature_scaler 没有 n_features_in_ 属性。可能scaler未正确拟合或版本问题。")
        print("将尝试使用默认特征数 5 作为 input_dim。")
        if hasattr(feature_scaler, 'scale_'): # 备用方案
            input_dim_model = len(feature_scaler.scale_)
        else:
            input_dim_model = 5 #最终回退
    print(f"模型输入维度 (input_dim) 设置为: {input_dim_model}")

    # 获取模型结构参数 (hidden_dim, layer_dim, output_dim, dropout)
    hidden_dim = PREDICTION_CONFIG.get('lstm_hidden_dim', 128)
    layer_dim = PREDICTION_CONFIG.get('lstm_layer_dim', 2)
    output_dim = 1 # 回归任务
    dropout = PREDICTION_CONFIG.get('lstm_dropout', 0.2)

    model = LSTMRegressor(input_dim_model, hidden_dim, layer_dim, output_dim, dropout_prob=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型 {model_path} 已加载。")

    # --- 2. 加载回测数据 (测试集) ---
    print(f"加载回测数据 (测试集) {stock_code} 从 {backtest_start_date} 到 {backtest_end_date}...")
    backtest_data_loader_dir = os.path.join(parent_dir, DATA_CONFIG.get('data_dir', 'data'))
    backtest_loader = StockDataLoader(data_dir=backtest_data_loader_dir)
    eval_stock_data_df = backtest_loader.load_stock_data(stock_code, backtest_start_date, backtest_end_date)

    if eval_stock_data_df is None or eval_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的回测数据。回测中止。")
        return
    if 'date' not in eval_stock_data_df.columns or not pd.api.types.is_datetime64_any_dtype(eval_stock_data_df['date']):
        print("错误: 回测数据缺少 'date' 列或其不是datetime类型。")
        # 尝试转换，如果失败则退出
        try:
            eval_stock_data_df['date'] = pd.to_datetime(eval_stock_data_df['date'])
        except Exception as e:
            print(f"转换日期列失败: {e}")
            return
    eval_stock_data_df = eval_stock_data_df.sort_values(by='date').reset_index(drop=True)
    print(f"成功加载 {len(eval_stock_data_df)} 条回测数据。")


    # --- 3. 获取模型预测 ---
    # 注意: generate_full_predictions 需要的 n_past_days 和 prediction_window 来自 PREDICTION_CONFIG
    n_past_days = PREDICTION_CONFIG.get('n_past_days_debug', 60)
    prediction_window = PREDICTION_CONFIG.get('prediction_days', 90)

    # generate_full_predictions 返回的 predicted_dates 是 input_sequence_end_dates + prediction_window
    # 我们需要将这些预测与 eval_stock_data_df 的日期对齐以进行每日决策
    print("正在生成模型预测...")
    # 传递给generate_full_predictions的DataFrame应该是包含足够历史数据以形成第一个序列的
    # 因此，加载数据时可能需要比eval_start_date更早一点的数据，或者generate_full_predictions内部能处理
    # 为简单起见，假设generate_full_predictions能处理好基于eval_stock_data_df的预测序列生成
    predicted_prices, predicted_dates = generate_full_predictions(
        model,
        eval_stock_data_df.copy(), # 传入整个回测期间的数据
        feature_scaler,
        target_scaler,
        n_past_days,
        prediction_window,
        device,
        stock_code
    )

    if predicted_prices is None or predicted_dates is None:
        print("未能生成模型预测。回测中止。")
        return
        
    predictions_df = pd.DataFrame({'predicted_date': pd.to_datetime(predicted_dates), 'predicted_close': predicted_prices})
    
    # --- 4. 执行回测循环 ---
    cash = initial_capital
    shares_held = 0.0
    portfolio_value_history = [] # 记录每日组合总价值

    # 为了决策，我们需要在每个交易日 eval_stock_data_df[i] 查找模型对未来的预测
    # 模型的预测是针对 predicted_date 的价格
    # 策略：在 current_date，我们看模型对 current_date + prediction_window 的预测价格
    
    print("开始回测循环...")
    for i in range(len(eval_stock_data_df)):
        current_date = eval_stock_data_df['date'].iloc[i]
        current_actual_close = eval_stock_data_df['close'].iloc[i]

        # 找到模型对未来 (current_date + prediction_window) 的预测
        # target_future_date_for_prediction = current_date + pd.Timedelta(days=prediction_window)
        # current_prediction_for_future = predictions_df[predictions_df['predicted_date'] == target_future_date_for_prediction]
        
        # 简化策略：我们使用在 input_sequence_end_date (即 current_date - prediction_window + N_PAST_DAYS -1 附近)
        # 做出的对 current_date 的预测。
        # 不，generate_full_predictions 返回的 predicted_dates[k] 是 input_sequence_end_dates[k] + prediction_window 的日期
        # 而 predicted_prices[k] 是对该日期的预测。
        # 所以，我们需要的是在 current_date，我们用什么预测来决策？
        # 假设策略是：在 current_date，我们看模型对 current_date 之后第 prediction_window 天的预测值。
        
        # 找到与当前日期 current_date 对应的、基于过去数据做出的对未来的预测
        # input_sequence_for_today_ends_at = current_date - pd.Timedelta(days=1) # 假设用昨天收盘数据预测
        # 我们需要找到 predictions_df 中哪个 predicted_date 是由接近 current_date 的数据生成的
        # generate_full_predictions 的 input_sequence_end_dates 是关键
        # 让我们重新思考：
        # 在回测的第 i 天 (current_date, current_actual_close):
        # 我们需要一个对 current_date 之后某个时间点 (如 current_date + X 天) 的预测。
        # generate_full_predictions 的输出 (predicted_prices, predicted_dates) 中，
        # predicted_dates[k] 是 input_sequence_end_dates[k] + prediction_window。
        # predicted_prices[k] 是对 predicted_dates[k] 的预测。
        
        # 策略：在 current_date，我们查找对 current_date + prediction_window 的预测。
        # 这个预测是基于 current_date 之前的数据做出的。
        target_prediction_date = current_date + pd.Timedelta(days=prediction_window)
        
        # 从 predictions_df 中找到对 target_prediction_date 的预测
        # 注意：predictions_df 中的 'predicted_date' 可能不完全精确匹配 target_prediction_date（如果日历不连续）
        # 我们需要找到最接近的可用预测。
        # 为了简化，我们假设可以直接查找。如果 predictions_df 是稀疏的，这里需要更鲁棒的查找。
        
        # 查找在 current_date 时可获得的、对未来的预测
        # 假设我们总是使用最新的、针对 "current_date 之后第 prediction_window 天" 的预测
        # 这个预测的 "生成日期" (即 input_sequence_end_date) 应该是 current_date 或其之前。
        
        # 简化：假设在 current_date，我们能拿到一个对未来的预测 P_future。
        # 这个 P_future 是模型对 current_date 之后第 prediction_window 天的价格的预测。
        # 我们需要从 predictions_df 中提取这个 P_future。
        # predictions_df 中的 'predicted_date' 是预测的目标日期。
        # 'predicted_price' 是对该目标日期的预测价。
        # 策略是在 current_date 做决策，所以我们需要一个基于 current_date 之前信息对未来的预测。
        
        # 找到在 predictions_df 中，predicted_date 等于 target_prediction_date 的那条记录
        relevant_prediction_row = predictions_df[predictions_df['predicted_date'] == target_prediction_date]

        if not relevant_prediction_row.empty:
            predicted_future_close = relevant_prediction_row['predicted_close'].iloc[0]
            
            # 交易决策 (简单示例)
            # 买入逻辑
            if cash > 0 and predicted_future_close > current_actual_close * buy_threshold_factor:
                shares_to_buy = cash / current_actual_close
                shares_held += shares_to_buy
                cash = 0.0
                print(f"{current_date.strftime('%Y-%m-%d')}: 买入 {shares_to_buy:.2f} 股 at {current_actual_close:.2f}。预测未来价: {predicted_future_close:.2f}。剩余现金: {cash:.2f}")
            # 卖出逻辑
            elif shares_held > 0 and predicted_future_close < current_actual_close * sell_threshold_factor:
                cash_before_sell = cash # 记录卖出前现金，虽然当前逻辑下卖出前现金通常是0（如果是从持股状态卖出）
                cash += shares_held * current_actual_close
                print(f"{current_date.strftime('%Y-%m-%d')}: 卖出 {shares_held:.2f} 股 at {current_actual_close:.2f}。预测未来价: {predicted_future_close:.2f}。现金变为: {cash:.2f}")
                shares_held = 0.0
        
        # 计算当日组合价值
        current_portfolio_value = cash + shares_held * current_actual_close
        portfolio_value_history.append(current_portfolio_value)

    # --- 5. 计算并打印回测结果 ---
    final_portfolio_value = portfolio_value_history[-1] if portfolio_value_history else initial_capital
    total_return = final_portfolio_value - initial_capital
    return_rate = (total_return / initial_capital) * 100

    print("\n--- 回测结果 ---")
    print(f"回测结束日期: {eval_stock_data_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"最终组合价值: {final_portfolio_value:,.2f}")
    print(f"总收益/亏损: {total_return:,.2f}")
    print(f"收益率: {return_rate:.2f}%")
    
    # (可选) 绘制资产曲线
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12,6))
    # plt.plot(eval_stock_data_df['date'], portfolio_value_history, label='Portfolio Value')
    # plt.title(f'Strategy Backtest for {stock_code} ({model_type} model)')
    # plt.xlabel('Date')
    # plt.ylabel('Portfolio Value')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return final_portfolio_value, total_return, return_rate


if __name__ == "__main__":
    # --- 配置回测参数 ---
    stock_to_backtest = '600036' 
    # backtest_start 和 backtest_end 将从 DATA_CONFIG 内部读取
    # 因此调用时不再需要传递日期参数
    
    initial_funds = 1000000.0
    model_to_use = 'val_best'

    print(f"将对股票 {stock_to_backtest} 使用 {model_to_use} 模型进行回测。")
    # 回测期间的打印已移至 run_backtest 函数内部

    run_backtest(
        stock_code=stock_to_backtest,
        initial_capital=initial_funds,
        model_type=model_to_use
    )