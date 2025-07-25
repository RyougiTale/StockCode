import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from Akshare.db_manager import read_table, save_dataframe

def read_cur_stock():
    stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
    print(stock_zh_a_spot_em_df)

def read_history_stock_by_code(code: str, start_date: str = "19700101", end_date: str = None):
    """
    获取单只股票的K线数据（日线），带数据库缓存功能。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

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
        # 如果数据已是今天，则无需更新
        if last_local_date.date() == now.date():
            needs_update = False
        # 如果数据是昨天，且当前未到收盘后，则无需更新
        elif last_local_date.date() == (now - timedelta(days=1)).date() and now.time() < datetime.strptime("15:30", "%H:%M").time():
            needs_update = False

    if not needs_update:
        print(f"数据已是最新 (更新到 {last_local_date.strftime('%Y-%m-%d')})，从本地数据库加载 {code}。")
    else:
        # 3. 执行网络更新
        fetch_start_date = (last_local_date + timedelta(days=1)).strftime('%Y%m%d') if last_local_date else "19700101"
        print(f"本地数据需要更新, 从网络获取 {code} 从 {fetch_start_date} 到 {end_date} 的数据...")
        try:
            online_data = ak.stock_zh_a_hist(symbol=code, start_date=fetch_start_date, end_date=end_date, period="daily", adjust="qfq")
            if not online_data.empty:
                online_data['stock_code'] = code
                column_mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'turnover', '振幅': 'amplitude', '涨跌幅': 'pct_chg', '涨跌额': 'chg_amount', '换手率': 'turnover_rate'}
                online_data.rename(columns=column_mapping, inplace=True)
                db_columns = ['stock_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
                df_to_save = online_data[db_columns]
                
                save_dataframe(df_to_save, 'stock_daily_kline', if_exists='append')
                
                # 将新数据合并到全量数据中
                all_local_data = pd.concat([all_local_data, df_to_save]).drop_duplicates(subset=['stock_code', 'date']).reset_index(drop=True)
            else:
                print(f"从网络未获取到 {code} 的新数据。")
        except Exception as e:
            print(f"!!! 获取 {code} 历史数据失败: {e} !!!")

    # 4. 无论是否更新，都从最终的全量本地数据中筛选出请求的时间范围并返回
    if not all_local_data.empty:
        # 再次确保date列是Timestamp类型，以防万一
        all_local_data['date'] = pd.to_datetime(all_local_data['date'])
        return all_local_data[(all_local_data['date'] >= pd.to_datetime(start_date)) & (all_local_data['date'] <= pd.to_datetime(end_date))].copy()
    
    return pd.DataFrame()
