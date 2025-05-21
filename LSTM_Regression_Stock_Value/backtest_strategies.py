import os
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

# 相对导入
from .config import DATA_CONFIG, PREDICTION_CONFIG
from .lstm_model import LSTMRegressor

# 从Util导入
from Util.StockDataLoader import StockDataLoader
# from Util.data_utils import prepare_data_for_pytorch_regression # 可能不需要直接用这个

# 尝试从config.py获取模型保存目录，如果不存在则使用默认值
try:
    from .config import MODEL_SAVE_DIR
except ImportError:
    MODEL_SAVE_DIR = './models/pytorch_lstm'

# --- 全局参数 ---
INITIAL_CAPITAL = 1000000.0
TRADE_AMOUNT_PER_BUY = 100000.0
# TRANSACTION_COST_RATE = 0.001 # 示例：0.1% 的交易成本（双边）

# --- 辅助函数 ---
def get_model_prediction_signal(model, current_data_sequence_scaled, target_scaler, base_price):
    """
    使用模型进行预测，并返回预测价格和涨跌信号。
    Args:
        model: 已加载的PyTorch模型。
        current_data_sequence_scaled (torch.Tensor): 当前归一化的输入序列 (1, seq_len, num_features)。
        target_scaler: 已加载的目标scaler。
        base_price (float): 用于比较涨跌的基准价格 (通常是当前价格 Close[t])。
    Returns:
        tuple: (predicted_price, signal)
               signal: 1 (涨), -1 (跌), 0 (平或无法判断)
    """
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(current_data_sequence_scaled)
        predicted_price_array = target_scaler.inverse_transform(prediction_scaled.cpu().numpy())
        predicted_price = predicted_price_array[0, 0]

    if predicted_price > base_price:
        signal = 1  # 预测涨
    elif predicted_price < base_price:
        signal = -1 # 预测跌
    else:
        signal = 0  # 预测平
    return predicted_price, signal

# --- 策略1：固定持有期 ---
def run_strategy_1(
    stock_code,
    backtest_start_date,
    backtest_end_date,
    model,
    feature_scaler,
    target_scaler,
    device
):
    print(f"\n--- 开始执行策略1：固定持有期 ({PREDICTION_CONFIG['prediction_days']}天) ---")
    capital = INITIAL_CAPITAL
    shares_held = 0
    pending_sell_orders = [] # 存储 (sell_date_index, shares_to_sell, purchase_price_for_pnl)

    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    # 需要加载比回测开始日期更早的数据，以构建第一个输入序列
    # 例如，早 n_past_days + buffer
    data_load_start_date = (pd.to_datetime(backtest_start_date) - pd.Timedelta(days=PREDICTION_CONFIG['n_past_days_debug'] + 60)).strftime('%Y-%m-%d')
    df = loader.load_stock_data(stock_code, data_load_start_date, backtest_end_date)
    if df is None or df.empty:
        print("策略1: 无法加载数据。")
        return capital, capital - INITIAL_CAPITAL
    
    df = df[(df['date'] >= pd.to_datetime(backtest_start_date)) & (df['date'] <= pd.to_datetime(backtest_end_date))].reset_index(drop=True)
    if len(df) <= PREDICTION_CONFIG['n_past_days_debug']:
        print("策略1: 回测期数据不足。")
        return capital, capital - INITIAL_CAPITAL

    print(f"回测数据从 {df['date'].iloc[0].strftime('%Y-%m-%d')} 到 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    n_past = PREDICTION_CONFIG['n_past_days_debug']
    pred_days = PREDICTION_CONFIG['prediction_days']
    feature_cols = PREDICTION_CONFIG['feature_columns']

    for i in tqdm(range(n_past, len(df)), desc="策略1回测中"):
        current_date = df['date'].iloc[i-1] # 当前决策基于 i-1 收盘的数据
        transaction_price_today = df['open'].iloc[i] # 交易发生在 i 开盘
        
        if pd.isna(transaction_price_today): # 跳过没有开盘价的日子
            # 处理待卖出订单（如果今天本应卖出但无法交易）
            new_pending_sell_orders = []
            for sell_idx, shares, buy_price in pending_sell_orders:
                if i >= sell_idx: # 如果错过了卖出日
                    print(f"{current_date.strftime('%Y-%m-%d')}: 原定卖出 {shares} 股，但当日无开盘价，顺延。")
                    new_pending_sell_orders.append((i + 1, shares, buy_price)) # 尝试下一天卖出
                else:
                    new_pending_sell_orders.append((sell_idx, shares, buy_price))
            pending_sell_orders = new_pending_sell_orders
            continue

        # 1. 处理卖出订单
        sold_today = False
        temp_pending_sell_orders = []
        for sell_idx, shares, buy_price in pending_sell_orders:
            if i == sell_idx:
                sell_value = shares * transaction_price_today
                capital += sell_value
                pnl_this_trade = sell_value - (shares * buy_price)
                print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略1] 卖出 {shares} 股 @ {transaction_price_today:.2f}, 价值 {sell_value:.2f}, 此笔盈亏 {pnl_this_trade:.2f}. 当前资金: {capital:.2f}")
                shares_held -= shares # 应该等于0，因为是固定持有期后全卖
                sold_today = True
            else:
                temp_pending_sell_orders.append((sell_idx, shares, buy_price))
        pending_sell_orders = temp_pending_sell_orders
        
        if shares_held > 0 and not sold_today: # 如果还有持仓但不是因为固定周期卖出（不应发生）
             pass # 正常持有

        # 2. 生成预测信号并决定是否买入 (仅当当前无持仓时才考虑买入)
        # 2. 生成预测信号并决定是否买入 (允许加仓)
        # 准备输入数据
        input_data_raw = df[feature_cols].iloc[i-n_past:i].values
        if input_data_raw.shape[0] < n_past: continue # 数据不足形成序列

        input_data_scaled = feature_scaler.transform(input_data_raw)
        X_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        base_price_for_signal = df['close'].iloc[i-1] # 基于i-1的收盘价做决策

        _, signal = get_model_prediction_signal(model, X_tensor, target_scaler, base_price_for_signal)

        if signal == 1: # 预测涨
            if transaction_price_today <= 0: # 避免股价为0或负导致错误
                print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略1] 预测涨，但当日股价无效 ({transaction_price_today:.2f})，无法买入。")
            else:
                shares_to_buy = math.floor(TRADE_AMOUNT_PER_BUY / transaction_price_today)
                if shares_to_buy > 0 and capital >= shares_to_buy * transaction_price_today:
                    cost = shares_to_buy * transaction_price_today
                    capital -= cost
                    shares_held += shares_to_buy # 允许加仓，总持股数增加
                    
                    sell_date_index = i + pred_days
                    # 每笔买入都有独立的卖出计划
                    if sell_date_index < len(df): #确保卖出日在数据范围内
                        pending_sell_orders.append((sell_date_index, shares_to_buy, transaction_price_today))
                        print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略1] 预测涨。买入 {shares_to_buy} 股 @ {transaction_price_today:.2f}, 花费 {cost:.2f}. 当前资金: {capital:.2f}. 总持仓: {shares_held}. 预计于索引 {sell_date_index} 卖出此笔.")
                    else: # 如果卖出日在数据范围外，则在最后一天卖出
                        pending_sell_orders.append((len(df) - 1, shares_to_buy, transaction_price_today))
                        print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略1] 预测涨。买入 {shares_to_buy} 股 @ {transaction_price_today:.2f}, 花费 {cost:.2f}. 总持仓: {shares_held}. 卖出日超出范围，此笔将在期末卖出.")
                elif shares_to_buy == 0:
                    print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略1] 预测涨，但资金 ({capital:.2f}) 不足以购买1股 @ {transaction_price_today:.2f} 或股价过高。")
                elif capital < shares_to_buy * transaction_price_today:
                     print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略1] 预测涨，但资金不足。需要 {shares_to_buy * transaction_price_today:.2f}, 现有 {capital:.2f}")


    # 回测期结束，清算所有剩余持仓 (主要处理卖出日在数据范围外的情况)
    if shares_held > 0 and pending_sell_orders:
        last_available_price = df['open'].iloc[-1] if not pd.isna(df['open'].iloc[-1]) else df['close'].iloc[-1]
        if pd.isna(last_available_price): # 如果最后一天也没有价格
            print(f"警告: 回测期末最后一天价格缺失，无法清算 {shares_held} 股。")
        else:
            for sell_idx, shares, buy_price in pending_sell_orders: # 此时pending_sell_orders里应该是未到期或顺延到最后的
                 sell_value = shares * last_available_price
                 capital += sell_value
                 pnl_this_trade = sell_value - (shares * buy_price)
                 print(f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}: [策略1] 期末清算卖出 {shares} 股 @ {last_available_price:.2f}, 价值 {sell_value:.2f}, 此笔盈亏 {pnl_this_trade:.2f}. 当前资金: {capital:.2f}")
            shares_held = 0 
            pending_sell_orders = []


    total_pnl = capital - INITIAL_CAPITAL
    print(f"--- 策略1结束 ---")
    print(f"初始资金: {INITIAL_CAPITAL:.2f}")
    print(f"最终资金: {capital:.2f}")
    print(f"总盈亏: {total_pnl:.2f}")
    return capital, total_pnl


# --- 策略2：信号驱动卖出 ---
def run_strategy_2(
    stock_code,
    backtest_start_date,
    backtest_end_date,
    model,
    feature_scaler,
    target_scaler,
    device
):
    print(f"\n--- 开始执行策略2：信号驱动卖出 ---")
    capital = INITIAL_CAPITAL
    shares_held = 0
    avg_buy_price = 0 # 用于计算盈亏

    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    data_load_start_date = (pd.to_datetime(backtest_start_date) - pd.Timedelta(days=PREDICTION_CONFIG['n_past_days_debug'] + 60)).strftime('%Y-%m-%d')
    df = loader.load_stock_data(stock_code, data_load_start_date, backtest_end_date)
    if df is None or df.empty:
        print("策略2: 无法加载数据。")
        return capital, capital - INITIAL_CAPITAL

    df = df[(df['date'] >= pd.to_datetime(backtest_start_date)) & (df['date'] <= pd.to_datetime(backtest_end_date))].reset_index(drop=True)
    if len(df) <= PREDICTION_CONFIG['n_past_days_debug']:
        print("策略2: 回测期数据不足。")
        return capital, capital - INITIAL_CAPITAL
        
    print(f"回测数据从 {df['date'].iloc[0].strftime('%Y-%m-%d')} 到 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    n_past = PREDICTION_CONFIG['n_past_days_debug']
    feature_cols = PREDICTION_CONFIG['feature_columns']

    for i in tqdm(range(n_past, len(df)), desc="策略2回测中"):
        current_date = df['date'].iloc[i-1]
        transaction_price_today = df['open'].iloc[i]

        if pd.isna(transaction_price_today): continue

        # 准备输入数据
        input_data_raw = df[feature_cols].iloc[i-n_past:i].values
        if input_data_raw.shape[0] < n_past: continue

        input_data_scaled = feature_scaler.transform(input_data_raw)
        X_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        base_price_for_signal = df['close'].iloc[i-1]
        _, signal = get_model_prediction_signal(model, X_tensor, target_scaler, base_price_for_signal)

        if signal == 1: # 预测涨
            # 允许加仓
            if transaction_price_today <= 0: # 避免股价为0或负导致错误
                print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略2] 预测涨，但当日股价无效 ({transaction_price_today:.2f})，无法买入。")
            else:
                shares_to_buy_now = math.floor(TRADE_AMOUNT_PER_BUY / transaction_price_today)
                if shares_to_buy_now > 0 and capital >= shares_to_buy_now * transaction_price_today:
                    cost_now = shares_to_buy_now * transaction_price_today
                    capital -= cost_now
                    
                    if shares_held == 0: # 首次建仓
                        shares_held = shares_to_buy_now
                        avg_buy_price = transaction_price_today
                    else: # 加仓
                        current_total_cost = shares_held * avg_buy_price
                        new_total_shares = shares_held + shares_to_buy_now
                        avg_buy_price = (current_total_cost + cost_now) / new_total_shares
                        shares_held = new_total_shares
                        
                    print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略2] 预测涨。买入/加仓 {shares_to_buy_now} 股 @ {transaction_price_today:.2f}, 花费 {cost_now:.2f}. 总持仓: {shares_held}, 平均成本: {avg_buy_price:.2f}. 当前资金: {capital:.2f}")
                elif shares_to_buy_now == 0:
                    print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略2] 预测涨，但资金 ({capital:.2f}) 不足以购买1股 @ {transaction_price_today:.2f} 或股价过高。")
                elif capital < shares_to_buy_now * transaction_price_today:
                     print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略2] 预测涨，但资金不足。需要 {shares_to_buy_now * transaction_price_today:.2f}, 现有 {capital:.2f}")
        
        elif signal == -1: # 预测跌
            if shares_held > 0:
                sell_value = shares_held * transaction_price_today
                capital += sell_value
                pnl_this_trade = sell_value - (shares_held * avg_buy_price)
                print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略2] 预测跌。卖出 {shares_held} 股 @ {transaction_price_today:.2f}, 价值 {sell_value:.2f}, 此笔盈亏 {pnl_this_trade:.2f}. 当前资金: {capital:.2f}")
                shares_held = 0
                avg_buy_price = 0

    # 回测期结束，清算所有剩余持仓
    if shares_held > 0:
        last_available_price = df['open'].iloc[-1] if not pd.isna(df['open'].iloc[-1]) else df['close'].iloc[-1]
        if pd.isna(last_available_price):
             print(f"警告: 回测期末最后一天价格缺失，无法清算 {shares_held} 股。")
        else:
            sell_value = shares_held * last_available_price
            capital += sell_value
            pnl_this_trade = sell_value - (shares_held * avg_buy_price)
            print(f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}: [策略2] 期末清算卖出 {shares_held} 股 @ {last_available_price:.2f}, 价值 {sell_value:.2f}, 此笔盈亏 {pnl_this_trade:.2f}. 当前资金: {capital:.2f}")
            shares_held = 0

    total_pnl = capital - INITIAL_CAPITAL
    print(f"--- 策略2结束 ---")
    print(f"初始资金: {INITIAL_CAPITAL:.2f}")
    print(f"最终资金: {capital:.2f}")
    print(f"总盈亏: {total_pnl:.2f}")
    return capital, total_pnl


# --- 策略3：买入并持有 ---
def run_strategy_buy_and_hold(
    stock_code,
    backtest_start_date,
    backtest_end_date
):
    print(f"\n--- 开始执行策略3：买入并持有 ---")
    capital = INITIAL_CAPITAL
    shares_bought = 0
    buy_price_at_start = 0.0 # 初始化

    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    # 只需要回测期的数据
    df_full_period = loader.load_stock_data(stock_code, backtest_start_date, backtest_end_date)
    
    if df_full_period is None or df_full_period.empty:
        print(f"策略3: 无法加载股票 {stock_code} 在 {backtest_start_date} 到 {backtest_end_date} 的数据。")
        return capital, capital - INITIAL_CAPITAL

    # 筛选确保日期在请求范围内，并去除开盘价无效的行，然后排序
    df = df_full_period[
        (df_full_period['date'] >= pd.to_datetime(backtest_start_date)) &
        (df_full_period['date'] <= pd.to_datetime(backtest_end_date)) &
        (df_full_period['open'].notna()) &
        (df_full_period['open'] > 0) # 确保开盘价有效用于交易
    ].copy()
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print(f"策略3: 在 {backtest_start_date} 到 {backtest_end_date} 期间没有有效的交易数据（开盘价有效）。")
        return INITIAL_CAPITAL, 0.0

    n_past = PREDICTION_CONFIG['n_past_days_debug']
    if len(df) <= n_past:
        print(f"策略3: 回测期有效数据天数 ({len(df)}) 不足以覆盖模型所需的前置天数 ({n_past})，无法进行公平的买入并持有回测。")
        return INITIAL_CAPITAL, 0.0

    # 买入操作：在模型策略可以开始交易的第一个有效交易日（即原始回测开始后的第 n_past 天）的开盘价买入
    actual_buy_day_index = n_past
    buy_date = df['date'].iloc[actual_buy_day_index]
    buy_price_at_start = df['open'].iloc[actual_buy_day_index]

    if pd.isna(buy_price_at_start) or buy_price_at_start <= 0: # 再次检查，尽管前面已筛选过open > 0
        print(f"策略3: 在公平起始日 ({buy_date.strftime('%Y-%m-%d')}) 的开盘价无效 ({buy_price_at_start})，无法执行买入。")
        return INITIAL_CAPITAL, 0.0
    
    shares_bought = math.floor(capital / buy_price_at_start)
    cost_of_purchase = shares_bought * buy_price_at_start
    capital_after_buy = capital - cost_of_purchase # 剩余现金

    if shares_bought > 0:
        print(f"{buy_date.strftime('%Y-%m-%d')}: [策略3] 以开盘价 {buy_price_at_start:.2f} 买入 {shares_bought} 股，花费 {cost_of_purchase:.2f}。剩余现金: {capital_after_buy:.2f}")
    else:
        print(f"{buy_date.strftime('%Y-%m-%d')}: [策略3] 资金 {INITIAL_CAPITAL:.2f} 不足以在开盘价 {buy_price_at_start:.2f} 买入任何股票。")
        return INITIAL_CAPITAL, 0.0

    # 卖出操作：在回测期的最后一个有效交易日的开盘价卖出
    sell_date = df['date'].iloc[-1]
    sell_price_at_end = df['open'].iloc[-1] # 使用最后一个有效日的开盘价

    value_at_end = shares_bought * sell_price_at_end
    final_capital = capital_after_buy + value_at_end # 剩余现金加上卖出股票所得
    
    print(f"{sell_date.strftime('%Y-%m-%d')}: [策略3] 以开盘价 {sell_price_at_end:.2f} 卖出 {shares_bought} 股，获得 {value_at_end:.2f}。")

    total_pnl = final_capital - INITIAL_CAPITAL
    print(f"--- 策略3结束 ---")
    print(f"初始资金: {INITIAL_CAPITAL:.2f}")
    print(f"最终资金: {final_capital:.2f}")
    print(f"总盈亏: {total_pnl:.2f}")
    return final_capital, total_pnl


# --- 策略4：基于预期收益率阈值的固定持有期策略 ---
def run_strategy_4(
    stock_code,
    backtest_start_date,
    backtest_end_date,
    model,
    feature_scaler,
    target_scaler,
    device,
    buy_threshold_return=0.05 # 默认5%的预期回报率作为买入阈值
):
    print(f"\n--- 开始执行策略4：基于预期收益率阈值 ({buy_threshold_return*100:.1f}%)，固定持有期 ({PREDICTION_CONFIG['prediction_days']}天) ---")
    capital = INITIAL_CAPITAL
    shares_held = 0
    pending_sell_orders = [] # 存储 (sell_date_index, shares_to_sell, purchase_price_for_pnl)

    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    data_load_start_date = (pd.to_datetime(backtest_start_date) - pd.Timedelta(days=PREDICTION_CONFIG['n_past_days_debug'] + 60)).strftime('%Y-%m-%d')
    df = loader.load_stock_data(stock_code, data_load_start_date, backtest_end_date)
    if df is None or df.empty:
        print("策略4: 无法加载数据。")
        return capital, capital - INITIAL_CAPITAL
    
    df = df[(df['date'] >= pd.to_datetime(backtest_start_date)) & (df['date'] <= pd.to_datetime(backtest_end_date))].reset_index(drop=True)
    if len(df) <= PREDICTION_CONFIG['n_past_days_debug']:
        print("策略4: 回测期数据不足。")
        return capital, capital - INITIAL_CAPITAL

    print(f"回测数据从 {df['date'].iloc[0].strftime('%Y-%m-%d')} 到 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    n_past = PREDICTION_CONFIG['n_past_days_debug']
    pred_days = PREDICTION_CONFIG['prediction_days'] # 持有期与预测期一致
    feature_cols = PREDICTION_CONFIG['feature_columns']

    for i in tqdm(range(n_past, len(df)), desc="策略4回测中"):
        current_decision_date = df['date'].iloc[i-1] # 决策基于i-1的数据
        transaction_price_today = df['open'].iloc[i] # 交易发生在i的开盘
        
        if pd.isna(transaction_price_today):
            new_pending_sell_orders = []
            for sell_idx, shares, buy_price in pending_sell_orders:
                if i >= sell_idx:
                    print(f"{current_decision_date.strftime('%Y-%m-%d')}: [策略4] 原定卖出 {shares} 股，但当日无开盘价，顺延。")
                    new_pending_sell_orders.append((i + 1, shares, buy_price))
                else:
                    new_pending_sell_orders.append((sell_idx, shares, buy_price))
            pending_sell_orders = new_pending_sell_orders
            continue

        # 1. 处理到期的卖出订单
        temp_pending_sell_orders = []
        for sell_idx, shares, buy_price in pending_sell_orders:
            if i == sell_idx:
                sell_value = shares * transaction_price_today
                capital += sell_value
                pnl_this_trade = sell_value - (shares * buy_price)
                shares_held -= shares
                print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略4] 固定持有期到期卖出 {shares} 股 @ {transaction_price_today:.2f}, 价值 {sell_value:.2f}, 此笔盈亏 {pnl_this_trade:.2f}. 当前资金: {capital:.2f}, 总持仓: {shares_held}")
            else:
                temp_pending_sell_orders.append((sell_idx, shares, buy_price))
        pending_sell_orders = temp_pending_sell_orders

        # 2. 生成预测并决定是否买入 (允许加仓)
        input_data_raw = df[feature_cols].iloc[i-n_past:i].values
        if input_data_raw.shape[0] < n_past: continue

        input_data_scaled = feature_scaler.transform(input_data_raw)
        X_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        base_price_for_signal = df['close'].iloc[i-1]
        predicted_future_price, _ = get_model_prediction_signal(model, X_tensor, target_scaler, base_price_for_signal) # signal在这里不直接用

        if base_price_for_signal > 0: # 避免除以零
            expected_return = (predicted_future_price - base_price_for_signal) / base_price_for_signal
            
            if expected_return > buy_threshold_return:
                if transaction_price_today <= 0:
                    print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略4] 预测收益率达标 ({expected_return:.2%})，但当日股价无效 ({transaction_price_today:.2f})，无法买入。")
                else:
                    shares_to_buy = math.floor(TRADE_AMOUNT_PER_BUY / transaction_price_today)
                    if shares_to_buy > 0 and capital >= shares_to_buy * transaction_price_today:
                        cost = shares_to_buy * transaction_price_today
                        capital -= cost
                        shares_held += shares_to_buy
                        
                        sell_date_index = i + pred_days
                        if sell_date_index < len(df):
                            pending_sell_orders.append((sell_date_index, shares_to_buy, transaction_price_today))
                            print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略4] 预测收益率 {expected_return:.2%} > {buy_threshold_return:.2%}. 买入 {shares_to_buy} 股 @ {transaction_price_today:.2f}, 花费 {cost:.2f}. 当前资金: {capital:.2f}. 总持仓: {shares_held}. 此笔预计于索引 {sell_date_index} 卖出.")
                        else:
                            pending_sell_orders.append((len(df) - 1, shares_to_buy, transaction_price_today))
                            print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略4] 预测收益率 {expected_return:.2%} > {buy_threshold_return:.2%}. 买入 {shares_to_buy} 股 @ {transaction_price_today:.2f}, 花费 {cost:.2f}. 总持仓: {shares_held}. 卖出日超出范围，此笔将在期末卖出.")
                    elif shares_to_buy == 0:
                        print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略4] 预测收益率达标，但资金 ({capital:.2f}) 不足以购买1股 @ {transaction_price_today:.2f} 或股价过高。")
                    elif capital < shares_to_buy * transaction_price_today:
                         print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略4] 预测收益率达标，但资金不足。需要 {shares_to_buy * transaction_price_today:.2f}, 现有 {capital:.2f}")
        else: # base_price_for_signal is 0 or negative
            print(f"{df['date'].iloc[i].strftime('%Y-%m-%d')}: [策略4] 基准价格无效 ({base_price_for_signal:.2f})，无法计算预期收益率。")


    # 回测期结束，清算所有剩余持仓
    if shares_held > 0 and pending_sell_orders:
        last_available_price = df['open'].iloc[-1] if not pd.isna(df['open'].iloc[-1]) else df['close'].iloc[-1]
        if pd.isna(last_available_price) or last_available_price <=0:
            print(f"警告: [策略4] 回测期末最后一天价格无效 ({last_available_price})，无法清算 {shares_held} 股。")
        else:
            for sell_idx, shares, buy_price in pending_sell_orders:
                 sell_value = shares * last_available_price
                 capital += sell_value
                 pnl_this_trade = sell_value - (shares * buy_price)
                 print(f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}: [策略4] 期末清算卖出 {shares} 股 @ {last_available_price:.2f}, 价值 {sell_value:.2f}, 此笔盈亏 {pnl_this_trade:.2f}. 当前资金: {capital:.2f}")
            shares_held = 0
            pending_sell_orders = []

    total_pnl = capital - INITIAL_CAPITAL
    print(f"--- 策略4结束 ---")
    print(f"初始资金: {INITIAL_CAPITAL:.2f}")
    print(f"最终资金: {capital:.2f}")
    print(f"总盈亏: {total_pnl:.2f}")
    return capital, total_pnl


if __name__ == "__main__":
    stock_code_to_test = DATA_CONFIG['default_stock_code']
    
    # --- 配置 ---
    model_filename = f"{stock_code_to_test}_val_best_lstm_model.pth"
    feature_scaler_filename = f"{stock_code_to_test}_feature_scaler.pkl"
    target_scaler_filename = f"{stock_code_to_test}_target_scaler.pkl"

    # 回测时段 (可以与评估时段相同，或另选)
    backtest_start = DATA_CONFIG['eval_start_date']
    backtest_end = DATA_CONFIG['eval_end_date'] 

    print(f"开始回测股票: {stock_code_to_test}")
    print(f"模型文件: {model_filename}")
    print(f"回测时段: {backtest_start} 到 {backtest_end}")
    print(f"初始资金: {INITIAL_CAPITAL}, 每次买入金额: {TRADE_AMOUNT_PER_BUY}")

    # --- 加载模型和Scalers ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    feature_scaler_path = os.path.join(MODEL_SAVE_DIR, feature_scaler_filename)
    target_scaler_path = os.path.join(MODEL_SAVE_DIR, target_scaler_filename)

    if not all(map(os.path.exists, [model_path, feature_scaler_path, target_scaler_path])):
        print("错误：模型或Scaler文件未找到。请确保路径正确且文件存在。")
        exit()

    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    
    input_dim = feature_scaler.n_features_in_
    hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
    layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
    output_dim = 1 # 假设预测单个值
    dropout = PREDICTION_CONFIG['lstm_dropout']
    
    model = LSTMRegressor(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型和Scalers加载完毕。使用设备: {device}")

    # --- 运行策略 ---
    # run_strategy_1(
    #     stock_code_to_test,
    #     backtest_start,
    #     backtest_end,
    #     model,
    #     feature_scaler,
    #     target_scaler,
    #     device
    # )
    
    # run_strategy_2(
    #     stock_code_to_test,
    #     backtest_start,
    #     backtest_end,
    #     model,
    #     feature_scaler,
    #     target_scaler,
    #     device
    # )

    run_strategy_buy_and_hold(
        stock_code_to_test,
        backtest_start,
        backtest_end
    )
    
    run_strategy_4(
        stock_code_to_test,
        backtest_start,
        backtest_end,
        model,
        feature_scaler,
        target_scaler,
        device,
        buy_threshold_return=0.03 # 可以选择在这里覆盖默认阈值
    )