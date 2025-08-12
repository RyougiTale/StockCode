import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# 利用项目根目录下的工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_util import read_history_by_code

try:
    from . import config
    from .rolling_scaler import RollingWindowScaler
    from .logger_config import get_logger, log_performance
except ImportError:
    # 如果相对导入失败，尝试直接导入
    import config
    from rolling_scaler import RollingWindowScaler
    from logger_config import get_logger, log_performance

# 获取日志记录器
logger = get_logger(__name__)

def resample_to_period(df, period='W-FRI'):
    """将日K数据降采样为周K或月K，修复异常值计算问题。"""
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    logic = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'turnover': 'sum',
        'amplitude': lambda x: ((x.max() - x.min()) / x.iloc[0] * 100) if (not x.empty and x.iloc[0] != 0) else 0,
        'pct_chg': lambda x: ((x.iloc[-1] / x.iloc[0] - 1) * 100) if (not x.empty and x.iloc[0] != 0) else 0,
        'chg_amount': lambda x: (x.iloc[-1] - x.iloc[0]) if not x.empty else 0,  # 修复：应该是价格变化，不是求和
        'turnover_rate': lambda x: x.mean()  # 修复：应该是平均值，不是求和
    }
    
    resampled_df = df.resample(period).apply(logic).dropna()
    
    # 异常值处理：限制幅度值的范围
    if 'amplitude' in resampled_df.columns:
        resampled_df['amplitude'] = np.clip(resampled_df['amplitude'], 0, 50)  # 限制振幅在50%以内
    
    if 'pct_chg' in resampled_df.columns:
        resampled_df['pct_chg'] = np.clip(resampled_df['pct_chg'], -50, 50)  # 限制涨跌幅在±50%以内
    
    if 'turnover_rate' in resampled_df.columns:
        resampled_df['turnover_rate'] = np.clip(resampled_df['turnover_rate'], 0, 100)  # 限制换手率在100%以内
    
    return resampled_df.reset_index()

def calculate_features(df, period):
    """根据不同的时间尺度计算相应的技术指标"""
    indicators_to_calc = config.TECH_INDICATORS.get(period, [])
    
    if 'SMA20' in indicators_to_calc:
        df['SMA20'] = df['close'].rolling(window=20).mean()
    if 'SMA60' in indicators_to_calc:
        df['SMA60'] = df['close'].rolling(window=60).mean()
        
    return df

def calculate_future_metrics(price_series):
    """
    计算未来N天窗口内的四个核心指标。
    Args:
        price_series (pd.Series): 未来N天的价格序列。
    Returns:
        dict: 包含四个指标的字典，如果数据不足则返回None。
    """
    if len(price_series) < 2:
        return None

    final_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    cumulative_max = price_series.cummax()
    drawdown = (price_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    daily_returns = price_series.pct_change().dropna()
    volatility = daily_returns.std()

    return {
        "final_return": final_return,
        "max_drawdown": max_drawdown,
        "volatility": volatility
    }

# (classify_market_pattern 函数将被移除)

@log_performance("数据样本创建")
def create_samples_for_code(code):
    """为单只股票/指数创建所有样本（新版：软标签）"""
    logger.info(f"开始处理股票 {code} 的数据...")
    daily_df = read_history_by_code(code)
    if daily_df is None or daily_df.empty:
        return [], {}

    # --- 步骤 1: 保存原始收盘价并计算标签 ---
    original_close = daily_df['close'].copy()
    look_forward_days = config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"]
    daily_df['label'] = original_close.shift(-look_forward_days) / original_close - 1
    
    # --- 步骤 2: 计算输入特征 (分时间尺度) ---
    daily_df = calculate_features(daily_df, 'daily')
    
    # --- 步骤 3: 准备多时间尺度数据 ---
    weekly_df = resample_to_period(daily_df.copy(), 'W-FRI')
    weekly_df = calculate_features(weekly_df, 'weekly')
    
    monthly_df = resample_to_period(daily_df.copy(), 'ME')
    monthly_df = calculate_features(monthly_df, 'monthly')

    # --- 步骤 4: 数据清洗和归一化 ---
    # 强制将所有特征列转换为数值类型，无效值变为NaN
    all_feature_cols = set(config.FEATURE_COLUMNS['daily'] + config.FEATURE_COLUMNS['weekly'] + config.FEATURE_COLUMNS['monthly'])
    for col in all_feature_cols:
        if col in daily_df.columns:
            daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
        if col in weekly_df.columns:
            weekly_df[col] = pd.to_numeric(weekly_df[col], errors='coerce')
        if col in monthly_df.columns:
            monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')
    
    if config.DEBUG_MODE:
        logger.debug("=== 归一化前数据统计 ===")
        logger.debug(f"日线数据长度: {len(daily_df)}")
        logger.debug(f"日线数据预览:\n{daily_df.head()}")
        logger.debug(f"日线数据尾部:\n{daily_df.tail()}")
        logger.debug(f"日线NaN统计:\n{daily_df[config.FEATURE_COLUMNS['daily']].isnull().sum()}")
        
        logger.debug(f"周线数据长度: {len(weekly_df)}")
        logger.debug(f"周线数据预览:\n{weekly_df.head()}")
        logger.debug(f"周线NaN统计:\n{weekly_df[config.FEATURE_COLUMNS['weekly']].isnull().sum()}")
        
        logger.debug(f"月线数据长度: {len(monthly_df)}")
        logger.debug(f"月线数据预览:\n{monthly_df.head()}")
        logger.debug(f"月线NaN统计:\n{monthly_df[config.FEATURE_COLUMNS['monthly']].isnull().sum()}")
    else:
        logger.info(f"数据长度 - 日线: {len(daily_df)}, 周线: {len(weekly_df)}, 月线: {len(monthly_df)}")

    # --- 数据归一化前的异常值检查 ---
    if config.ENABLE_DATA_VALIDATION:
        logger.debug("检查归一化前的数据异常...")
        for period, df, feature_cols in [('daily', daily_df, config.FEATURE_COLUMNS['daily']),
                                         ('weekly', weekly_df, config.FEATURE_COLUMNS['weekly']),
                                         ('monthly', monthly_df, config.FEATURE_COLUMNS['monthly'])]:
            if config.DEBUG_MODE:
                logger.debug(f"{period}数据异常检查:")
            anomaly_summary = {}
            for col in feature_cols:
                if col in df.columns:
                    col_data = df[col]
                    # 确保数据是数值类型
                    try:
                        col_data = pd.to_numeric(col_data, errors='coerce')
                        inf_count = np.isinf(col_data).sum()
                        nan_count = np.isnan(col_data).sum()
                        zero_count = (col_data == 0).sum()
                        min_val, max_val = col_data.min(), col_data.max()
                        
                        if config.DEBUG_MODE:
                            logger.debug(f"  {col}: inf={inf_count}, nan={nan_count}, zeros={zero_count}, range=[{min_val:.6f}, {max_val:.6f}]")
                        
                        if inf_count > 0 or nan_count > 0:
                            anomaly_summary[col] = {'inf': inf_count, 'nan': nan_count}
                        
                        # 替换无穷大值
                        if inf_count > 0:
                            logger.warning(f"替换 {col} 中的 {inf_count} 个无穷大值")
                            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    except Exception as e:
                        logger.error(f"处理列 {col} 时出错: {e}")
            
            if anomaly_summary and not config.DEBUG_MODE:
                logger.warning(f"{period}数据异常统计: {anomaly_summary}")
    
    # 删除包含 NaN 的行
    daily_df.dropna(subset=config.FEATURE_COLUMNS['daily'], inplace=True)
    weekly_df.dropna(subset=config.FEATURE_COLUMNS['weekly'], inplace=True)
    monthly_df.dropna(subset=config.FEATURE_COLUMNS['monthly'], inplace=True)
    
    logger.info(f"数据清理后: 日线={len(daily_df)}, 周线={len(weekly_df)}, 月线={len(monthly_df)}")

    # 使用滚动窗口归一化替代MinMaxScaler
    logger.info("开始滚动窗口归一化...")
    
    # 日线数据归一化（窗口252天，约1年）
    daily_scaler = RollingWindowScaler(window_size=252, method='zscore', min_periods=60)
    daily_df = daily_scaler.fit_transform(daily_df, config.FEATURE_COLUMNS['daily'])
    
    # 周线数据归一化（窗口52周，约1年）
    weekly_scaler = RollingWindowScaler(window_size=52, method='zscore', min_periods=12)
    weekly_df = weekly_scaler.fit_transform(weekly_df, config.FEATURE_COLUMNS['weekly'])
    
    # 月线数据归一化（窗口24个月，约2年）
    monthly_scaler = RollingWindowScaler(window_size=24, method='zscore', min_periods=6)
    monthly_df = monthly_scaler.fit_transform(monthly_df, config.FEATURE_COLUMNS['monthly'])
    
    logger.info("滚动窗口归一化完成")
    
    # --- 归一化后的数据检查 ---
    if config.DEBUG_MODE:
        logger.debug("=== 归一化后数据检查 ===")
        for period, df, feature_cols in [('daily', daily_df, config.FEATURE_COLUMNS['daily']),
                                         ('weekly', weekly_df, config.FEATURE_COLUMNS['weekly']),
                                         ('monthly', monthly_df, config.FEATURE_COLUMNS['monthly'])]:
            logger.debug(f"{period}归一化数据:")
            for col in feature_cols:
                if col in df.columns:
                    col_data = df[col]
                    nan_count = np.isnan(col_data).sum()
                    min_val, max_val = col_data.min(), col_data.max()
                    logger.debug(f"  {col}: nan={nan_count}, range=[{min_val:.6f}, {max_val:.6f}]")
    else:
        # 生产环境只检查是否有异常
        total_nan = 0
        for period, df, feature_cols in [('daily', daily_df, config.FEATURE_COLUMNS['daily']),
                                         ('weekly', weekly_df, config.FEATURE_COLUMNS['weekly']),
                                         ('monthly', monthly_df, config.FEATURE_COLUMNS['monthly'])]:
            period_nan = sum(np.isnan(df[col]).sum() for col in feature_cols if col in df.columns)
            total_nan += period_nan
        
        if total_nan > 0:
            logger.warning(f"归一化后发现 {total_nan} 个NaN值")
        else:
            logger.info("归一化完成，数据质量良好")

    if config.DEBUG_MODE:
        logger.debug("=== 归一化后数据预览 ===")
        logger.debug(f"日线数据预览:\n{daily_df.head()}")
        logger.debug(f"周线数据预览:\n{weekly_df.head()}")
        logger.debug(f"月线数据预览:\n{monthly_df.head()}")

    # --- 步骤 5: 创建样本 ---
    samples = []
    daily_df.dropna(subset=['label'] + config.FEATURE_COLUMNS['daily'], inplace=True)

    for i in range(len(daily_df) - 1, config.DAILY_SEQ_LEN - 1, -1):
        current_date = daily_df.iloc[i]['date']
        
        daily_end_idx = i + 1
        daily_start_idx = daily_end_idx - config.DAILY_SEQ_LEN
        daily_slice = daily_df.iloc[daily_start_idx:daily_end_idx]
        
        weekly_slice = weekly_df[weekly_df['date'] <= current_date].tail(config.WEEKLY_SEQ_LEN)
        monthly_slice = monthly_df[monthly_df['date'] <= current_date].tail(config.MONTHLY_SEQ_LEN)
        
        if (len(daily_slice) == config.DAILY_SEQ_LEN and
            len(weekly_slice) == config.WEEKLY_SEQ_LEN and
            len(monthly_slice) == config.MONTHLY_SEQ_LEN):
            
            daily_data = daily_slice[config.FEATURE_COLUMNS['daily']].values
            weekly_data = weekly_slice[config.FEATURE_COLUMNS['weekly']].values
            monthly_data = monthly_slice[config.FEATURE_COLUMNS['monthly']].values
            label = daily_df.iloc[i]['label']
            # 必须从原始收盘价中提取future_prices，而不是从归一化后的daily_df['close']
            future_prices = original_close.iloc[i : i + look_forward_days].values

            if np.isnan(daily_data).any() or np.isnan(weekly_data).any() or np.isnan(monthly_data).any() or pd.isna(label):
                continue
            
            sample = {
                'date': current_date,
                'daily': daily_data,
                'weekly': weekly_data,
                'monthly': monthly_data,
                'label': label,
                'future_prices': future_prices,
                'stock_code': code  # 添加股票代码信息，用于相对化指标计算
            }
            samples.append(sample)
    
    # 返回样本和拟合好的scalers
    scalers = {'daily': daily_scaler, 'weekly': weekly_scaler, 'monthly': monthly_scaler}
    return samples[::-1], scalers

def get_all_available_stock_codes():
    """动态获取所有可用的股票代码，用于预训练"""
    try:
        # 这里需要根据你的数据源获取所有股票代码
        # 如果有现成的股票列表文件或数据库，可以在这里实现
        logger.info("尝试获取所有可用股票代码...")
        
        # 方法1: 如果有股票代码文件
        stock_list_file = config.PRETRAINING_CONFIG.get("stock_list_file")
        if stock_list_file and os.path.exists(stock_list_file):
            with open(stock_list_file, 'r', encoding='utf-8') as f:
                codes = [line.strip() for line in f if line.strip()]
            logger.info(f"从文件 {stock_list_file} 加载了 {len(codes)} 只股票")
            return codes
            
        # 方法2: 动态扫描数据库或数据目录（这里是示例，需要根据实际数据源调整）
        # 例如，如果股票数据存储在特定目录下，可以扫描文件名
        # data_dir = "/path/to/stock/data"
        # if os.path.exists(data_dir):
        #     files = os.listdir(data_dir)
        #     codes = [f.split('_')[0] for f in files if f.endswith('.csv')]
        #     return sorted(set(codes))
        
        # 方法3: 使用已知的常用股票代码作为fallback
        fallback_codes = [
            "000001", "000002", "000858", "002415", "002594", "300059", 
            "600519", "600036", "603259", "000063", "000166", "002230",
            "300750", "600900", "601318", "000725", "002304", "300253"
        ]
        logger.warning(f"无法动态获取股票列表，使用默认的 {len(fallback_codes)} 只股票")
        return fallback_codes
        
    except Exception as e:
        logger.error(f"获取股票代码失败: {e}")
        return []

def filter_stock_codes(codes):
    """根据配置筛选股票代码"""
    if not codes:
        return []
        
    original_count = len(codes)
    logger.info(f"开始筛选股票，初始数量: {original_count}")
    
    # 排除特定模式的股票（如ST股票）
    exclude_patterns = config.PRETRAINING_CONFIG.get("exclude_stocks", [])
    filtered_codes = []
    
    for code in codes:
        should_exclude = False
        
        for pattern in exclude_patterns:
            if pattern.startswith("*") and pattern.endswith("*"):
                # 包含模式，如 "*ST*"
                if pattern[1:-1] in code:
                    should_exclude = True
                    break
            elif pattern.startswith("*"):
                # 后缀模式，如 "*ST"  
                if code.endswith(pattern[1:]):
                    should_exclude = True
                    break
            elif pattern.endswith("*"):
                # 前缀模式，如 "ST*"
                if code.startswith(pattern[:-1]):
                    should_exclude = True
                    break
            else:
                # 精确匹配
                if code == pattern:
                    should_exclude = True
                    break
        
        if not should_exclude:
            filtered_codes.append(code)
    
    # 应用数量限制
    max_stocks = config.PRETRAINING_CONFIG.get("max_stocks")
    if max_stocks and len(filtered_codes) > max_stocks:
        logger.info(f"限制股票数量从 {len(filtered_codes)} 到 {max_stocks}")
        filtered_codes = filtered_codes[:max_stocks]
    
    excluded_count = original_count - len(filtered_codes)
    logger.info(f"股票筛选完成: 保留 {len(filtered_codes)} 只，排除 {excluded_count} 只")
    
    return filtered_codes

@log_performance("获取预训练股票代码")
def get_stock_codes_for_training():
    """根据当前训练阶段获取合适的股票代码"""
    if config.TRAINING_PHASE == "pretraining":
        if config.PRETRAINING_CONFIG.get("use_all_available", False):
            # 动态获取所有可用股票
            all_codes = get_all_available_stock_codes()
            filtered_codes = filter_stock_codes(all_codes)
            
            if not filtered_codes:
                logger.error("未能获取任何有效的股票代码用于预训练")
                return []
                
            logger.info(f"预训练将使用 {len(filtered_codes)} 只股票")
            return filtered_codes
        else:
            # 使用config中预定义的股票列表
            return config.STOCK_CODES or []
            
    elif config.TRAINING_PHASE == "finetuning":
        # 微调阶段使用单只股票
        return config.STOCK_CODES or []
    
    return []

@log_performance("验证股票数据质量")
def validate_stock_data_quality(code, samples):
    """验证单只股票的数据质量，返回是否符合要求"""
    min_data_points = config.PRETRAINING_CONFIG.get("min_data_points", 1000)
    
    if len(samples) < min_data_points:
        if config.DEBUG_MODE:
            logger.debug(f"股票 {code} 数据点不足: {len(samples)} < {min_data_points}")
        return False
    
    # 检查数据完整性
    nan_count = sum(1 for sample in samples if 
                   np.isnan(sample['daily']).any() or 
                   np.isnan(sample['weekly']).any() or 
                   np.isnan(sample['monthly']).any())
    
    nan_ratio = nan_count / len(samples)
    if nan_ratio > 0.1:  # 如果超过10%的样本包含NaN，认为质量不佳
        logger.warning(f"股票 {code} 数据质量不佳，NaN比例: {nan_ratio:.2%}")
        return False
        
    return True

def get_all_samples(stock_codes=None):
    """获取所有股票代码的样本，支持大规模预训练数据加载"""
    
    # 如果未提供stock_codes，根据训练阶段自动获取
    if stock_codes is None:
        stock_codes = get_stock_codes_for_training()
        
    if not stock_codes:
        logger.error("没有可用的股票代码")
        return [], {}
    
    logger.info(f"开始获取 {len(stock_codes)} 只股票的样本数据...")
    all_samples = []
    all_scalers = {}
    successful_codes = []
    failed_codes = []
    
    # 对于大量股票，显示进度
    from tqdm import tqdm
    
    for code in tqdm(stock_codes, desc="处理股票数据"):
        try:
            samples, scalers = create_samples_for_code(code)
            
            if samples:
                # 验证数据质量
                if validate_stock_data_quality(code, samples):
                    all_samples.extend(samples)
                    successful_codes.append(code)
                    
                    # 保存第一个成功股票的scalers作为参考
                    if not all_scalers:
                        all_scalers = scalers
                        logger.info(f"使用股票 {code} 的scalers作为参考")
                else:
                    logger.info(f"股票 {code} 数据质量不符合要求，跳过")
                    failed_codes.append((code, "数据质量不佳"))
            else:
                logger.warning(f"股票 {code} 无法创建有效样本")
                failed_codes.append((code, "无法创建样本"))
                
        except Exception as e:
            logger.error(f"处理股票 {code} 时出错: {e}")
            failed_codes.append((code, f"处理出错: {e}"))
            continue
    
    # 统计结果
    logger.info(f"数据加载完成:")
    logger.info(f"  成功: {len(successful_codes)} 只股票, 总样本数: {len(all_samples)}")
    logger.info(f"  失败: {len(failed_codes)} 只股票")
    
    if config.DEBUG_MODE and failed_codes:
        logger.debug("失败股票详情:")
        for code, reason in failed_codes[:10]:  # 只显示前10个
            logger.debug(f"  {code}: {reason}")
        if len(failed_codes) > 10:
            logger.debug(f"  ... 还有 {len(failed_codes) - 10} 只股票失败")
    
    # 验证最终结果
    if not all_samples:
        logger.error("未能获取任何有效样本")
        return [], {}
        
    if len(successful_codes) < len(stock_codes) * 0.5:
        logger.warning(f"成功率较低: {len(successful_codes)}/{len(stock_codes)} = {len(successful_codes)/len(stock_codes):.1%}")
    
    return all_samples, all_scalers