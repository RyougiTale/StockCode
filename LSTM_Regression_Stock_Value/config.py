# 股票长周期预测系统配置文件
import pandas as pd

_N_PAST_DAYS_FOR_SHIFT = 30 # 这个值应与 PREDICTION_CONFIG 中的 n_past_days_debug 同步

def _shift_date_approx_calendar(base_date_str, n_trading_days_to_shift_back):
    """
    根据交易日大致回溯日历日期。
    例如，每5个交易日近似为7个日历天。
    """
    calendar_days_to_shift = int(n_trading_days_to_shift_back * 7 / 5)
    base_timestamp = pd.Timestamp(base_date_str)
    shifted_timestamp = base_timestamp - pd.Timedelta(days=calendar_days_to_shift)
    return shifted_timestamp.strftime('%Y-%m-%d')

# 数据相关配置
DATA_CONFIG = {
    'data_dir': './data',                    # 数据存储目录
    'default_stock_code': '600036',         # 默认股票代码(浦发银行)

    # 训练集时段
    'train_start_date': _shift_date_approx_calendar('2008-01-01', _N_PAST_DAYS_FOR_SHIFT),
    'train_end_date': '2021-06-30',

    # 验证集时段 (紧跟训练集之后)
    'validation_start_date': _shift_date_approx_calendar('2023-07-01', _N_PAST_DAYS_FOR_SHIFT),
    'validation_end_date': '2024-07-01',

    # 评估/回测集时段 (紧跟验证集之后)
    # 'eval_start_date': _shift_date_approx_calendar('2024-07-01', _N_PAST_DAYS_FOR_SHIFT),
    'eval_start_date': _shift_date_approx_calendar('2023-07-01', _N_PAST_DAYS_FOR_SHIFT),
    'eval_end_date': '2025-05-21',
}

# 预测相关配置
PREDICTION_CONFIG = {
    'n_past_days_debug': _N_PAST_DAYS_FOR_SHIFT, # 输入序列长度（过去多少天的数据）
    'prediction_days': 3,                  # 预测未来多少天
    'lstm_hidden_dim': 256,
    'lstm_layer_dim': 3,
    'lstm_dropout': 0.2,
    'learning_rate': 0.0001,                 # 学习率
    'num_epochs_debug': 10,                 # 训练轮数
    'batch_size': 128,                       # 批量大小
    'inference_batch_size': 64,
    'random_seed': 42,                      # 随机种子
    'feature_columns': ['open', 'high', 'low', 'close', 'volume'], # 使用的特征列
    'target_column': 'close',                  # 预测的目标列名
    
    # 以下配置项可以移除或注释掉，因为日期范围已明确指定
    # 'train_test_split': 0.8,
    # 'test_size': 0.2,
    # 'time_based_split': True,

    # 保留其他可能存在的配置
    'long_cycle_days': 120,
    'very_long_cycle_days': 180,
    'mid_cycle_days': 60,
    'short_cycle_days': 20,
}

# 模型相关配置 (保持不变)
MODEL_CONFIG = {
    'model_type': 'random_forest',          # 模型类型：random_forest, xgboost, lstm
    'model_params': {
        'random_forest': {
            'n_estimators': 100,            # 树的数量
            'max_depth': 10,                # 最大深度
            'random_state': 42              # 随机种子
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'lstm': { # 这些参数似乎更适合旧的Keras风格，PyTorch在PREDICTION_CONFIG中定义
            'units': 50, # 可考虑移除或与PREDICTION_CONFIG中的lstm参数统一
            'batch_size': 32, # 与PREDICTION_CONFIG中的batch_size冲突
            'epochs': 50, # 与PREDICTION_CONFIG中的num_epochs_debug冲突
            'dropout': 0.2 # 与PREDICTION_CONFIG中的lstm_dropout冲突
        }
    },
    'saved_model_path': './models/'         # 模型保存路径
}

# 特征工程配置 (保持不变)
FEATURE_CONFIG = {
    'use_ma': True,                         # 是否使用移动平均线
    'use_macd': True,                       # 是否使用MACD
    'use_rsi': True,                        # 是否使用RSI
    'use_seasonal': True,                   # 是否使用季节性特征
    'ma_windows': [5, 10, 20, 60, 120],     # MA窗口大小
    'rsi_window': 14,                       # RSI窗口大小
    'macd_params': {                        # MACD参数
        'fast': 12,
        'slow': 26,
        'signal': 9
    }
}

# 可视化配置 (保持不变)
VISUALIZATION_CONFIG = {
    'figure_size': (14, 7),                 # 图表大小
    'save_path': './',                      # 图表保存路径
    'dpi': 300,                             # 图表DPI
    'up_color': 'red',                      # 上涨颜色
    'down_color': 'green',                  # 下跌颜色
    'line_color': 'blue'                    # 线条颜色
}