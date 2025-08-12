import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from db_manager import read_table, save_dataframe
import re

def _is_etf(code: str) -> bool:
    """根据代码格式判断是否为ETF"""
    return bool(re.match(r'^(51|58|15)\d{4}$', code))

def read_history_stock_by_code(code: str, start_date: str = "19700101", end_date: str = None):
    """
    获取单只【股票】的K线数据（日线），带数据库缓存功能。
    """
    if end_date is None:
        end_date_dt = datetime.now()
        # 如果当前时间早于15:00，则将截止日期设为昨天
        if end_date_dt.time() < datetime.strptime("15:00", "%H:%M").time():
            end_date_dt -= timedelta(days=1)
        end_date = end_date_dt.strftime('%Y%m%d')

    # 1. 读取该股票在数据库中的【全部】历史数据
    all_local_data = read_table('stock_daily_kline', where_clause=f"stock_code = '{code}'")
    if not all_local_data.empty:
        # 确保date列永远是Timestamp类型
        all_local_data['date'] = pd.to_datetime(all_local_data['date'])

    # 2. 判断是否需要从网络更新
    last_local_date = all_local_data['date'].max() if not all_local_data.empty else None
    needs_update = True
    if last_local_date:
        now = datetime.now()
        today = now.date()
        
        # 获取最近的已完成交易日
        def get_last_completed_trading_day():
            # 如果当前是交易日但还未收盘（15:30前），则最近完成的交易日是前一个工作日
            if today.weekday() < 5 and now.time() < datetime.strptime("15:30", "%H:%M").time():
                # 从昨天开始往前找最近的工作日
                check_date = today - timedelta(days=1)
            else:
                # 从今天开始往前找最近的工作日
                check_date = today
            
            while check_date.weekday() >= 5:  # 0=Monday, 6=Sunday
                check_date -= timedelta(days=1)
            return check_date
        
        last_completed_trading_day = get_last_completed_trading_day()
        
        # 如果数据已包含最近的已完成交易日，则无需更新
        if last_local_date.date() >= last_completed_trading_day:
            needs_update = False

    if not needs_update:
        print(f"股票数据已是最新 (更新到 {last_local_date.strftime('%Y-%m-%d')})，从本地数据库加载 {code}。")
    else:
        # 3. 执行网络更新
        fetch_start_date = (last_local_date + timedelta(days=1)).strftime('%Y%m%d') if last_local_date else "19700101"
        print(f"本地股票数据需要更新, 从网络获取 {code} 从 {fetch_start_date} 到 {end_date} 的数据...")
        try:
            online_data = ak.stock_zh_a_hist(symbol=code, start_date=fetch_start_date, end_date=end_date, period="daily", adjust="hfq")
            if not online_data.empty:
                online_data['stock_code'] = code
                column_mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'turnover', '振幅': 'amplitude', '涨跌幅': 'pct_chg', '涨跌额': 'chg_amount', '换手率': 'turnover_rate'}
                online_data.rename(columns=column_mapping, inplace=True)
                db_columns = ['stock_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
                df_to_save = online_data[db_columns]
                save_dataframe(df_to_save, 'stock_daily_kline', if_exists='append')
                all_local_data = pd.concat([all_local_data, df_to_save]).drop_duplicates(subset=['stock_code', 'date']).reset_index(drop=True)
            else:
                print(f"从网络未获取到股票 {code} 的新数据。")
        except Exception as e:
            print(f"!!! 获取股票 {code} 历史数据失败: {e} !!!")

    if not all_local_data.empty:
        all_local_data['date'] = pd.to_datetime(all_local_data['date'])
        return all_local_data[(all_local_data['date'] >= pd.to_datetime(start_date)) & (all_local_data['date'] <= pd.to_datetime(end_date))].copy()
    
    return pd.DataFrame()

def read_history_etf_by_code(code: str, start_date: str = "19700101", end_date: str = None):
    """
    获取单只【ETF】的K线数据（日线），带数据库缓存功能。
    """
    if end_date is None:
        end_date_dt = datetime.now()
        # 如果当前时间早于15:00，则将截止日期设为昨天
        if end_date_dt.time() < datetime.strptime("15:00", "%H:%M").time():
            end_date_dt -= timedelta(days=1)
        end_date = end_date_dt.strftime('%Y%m%d')

    all_local_data = read_table('etf_daily_kline', where_clause=f"stock_code = '{code}'")
    if not all_local_data.empty:
        all_local_data['date'] = pd.to_datetime(all_local_data['date'])

    last_local_date = all_local_data['date'].max() if not all_local_data.empty else None
    needs_update = True
    if last_local_date:
        now = datetime.now()
        today = now.date()
        
        # 获取最近的已完成交易日
        def get_last_completed_trading_day():
            # 如果当前是交易日但还未收盘（15:30前），则最近完成的交易日是前一个工作日
            if today.weekday() < 5 and now.time() < datetime.strptime("15:30", "%H:%M").time():
                # 从昨天开始往前找最近的工作日
                check_date = today - timedelta(days=1)
            else:
                # 从今天开始往前找最近的工作日
                check_date = today
            
            while check_date.weekday() >= 5:  # 0=Monday, 6=Sunday
                check_date -= timedelta(days=1)
            return check_date
        
        last_completed_trading_day = get_last_completed_trading_day()
        
        # 如果数据已包含最近的已完成交易日，则无需更新
        if last_local_date.date() >= last_completed_trading_day:
            needs_update = False

    if not needs_update:
        print(f"ETF数据已是最新 (更新到 {last_local_date.strftime('%Y-%m-%d')})，从本地数据库加载 {code}。")
    else:
        fetch_start_date = (last_local_date + timedelta(days=1)).strftime('%Y%m%d') if last_local_date else "19700101"
        print(f"本地ETF数据需要更新, 从网络获取 {code} 从 {fetch_start_date} 到 {end_date} 的数据...")
        try:
            online_data = ak.fund_etf_hist_em(symbol=code, start_date=fetch_start_date, end_date=end_date, period="daily", adjust="hfq")
            if not online_data.empty:
                online_data['stock_code'] = code
                column_mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'turnover', '振幅': 'amplitude', '涨跌幅': 'pct_chg', '涨跌额': 'chg_amount', '换手率': 'turnover_rate'}
                online_data.rename(columns=column_mapping, inplace=True)
                db_columns = ['stock_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
                df_to_save = online_data[db_columns]
                save_dataframe(df_to_save, 'etf_daily_kline', if_exists='append')
                all_local_data = pd.concat([all_local_data, df_to_save]).drop_duplicates(subset=['stock_code', 'date']).reset_index(drop=True)
            else:
                print(f"从网络未获取到ETF {code} 的新数据。")
        except Exception as e:
            print(f"!!! 获取ETF {code} 历史数据失败: {e} !!!")

    if not all_local_data.empty:
        all_local_data['date'] = pd.to_datetime(all_local_data['date'])
        return all_local_data[(all_local_data['date'] >= pd.to_datetime(start_date)) & (all_local_data['date'] <= pd.to_datetime(end_date))].copy()
    
    return pd.DataFrame()

def read_history_by_code(code: str, start_date: str = "19700101", end_date: str = None):
    """
    统一的K线数据读取入口，自动判断股票或ETF。
    """
    if _is_etf(code):
        print(f"代码 {code} 被识别为 ETF。")
        return read_history_etf_by_code(code, start_date, end_date)
    else:
        print(f"代码 {code} 被识别为 股票。")
        return read_history_stock_by_code(code, start_date, end_date)
