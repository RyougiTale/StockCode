import os
from datetime import datetime
import baostock as bs
import pandas as pd


class StockDataLoader:
    """股票数据加载"""
    def __init__(self, data_dir='./data'):
        """初始化数据加载器
        Args:
            data_dir: 股票数据目录
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def load_stock_data(self, stock_code, start_date=None, end_date=None):
        """从BaoStock加载真实股票数据，并实现增量更新本地缓存。

        Args:
            stock_code: 股票代码，如 '600000'（上证）或 '000001'（深证）
            start_date: 请求数据的起始日期，如 '2020-01-01'
            end_date: 请求数据的结束日期，如 '2023-12-31'

        Returns:
            DataFrame: 包含指定日期范围内股票数据的DataFrame，
                       列包括 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg'。
                       如果加载失败则返回None。
        """
        if not start_date or not end_date:
            raise ValueError("start_date 和 end_date 不能为空")

        req_start_date = pd.to_datetime(start_date)
        req_end_date = pd.to_datetime(end_date)

        if req_start_date > req_end_date:
            raise ValueError("start_date 不能晚于 end_date")

        cache_filename = f"{stock_code}.csv"
        cache_file_path = os.path.join(self.data_dir, cache_filename)
        
        local_data = None
        new_data_downloaded = False

        # 调整股票代码格式（BaoStock需要前缀：sh或sz）
        bs_stock_code = stock_code
        if stock_code.startswith('6'):
            bs_stock_code = f'sh.{stock_code}'
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            bs_stock_code = f'sz.{stock_code}'
        else:
            raise ValueError("无效的股票代码 (不是以 6/0/3 开头)")

        try:
            if os.path.exists(cache_file_path):
                print(f"从缓存文件加载数据: {cache_file_path}")
                local_data = pd.read_csv(cache_file_path, parse_dates=['date'])
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
                for col in numeric_columns:
                    if col in local_data.columns:
                        local_data[col] = pd.to_numeric(local_data[col], errors='coerce')
                
                if not local_data.empty:
                    local_data.sort_values(by='date', inplace=True)
                    local_data.drop_duplicates(subset=['date'], keep='first', inplace=True) # 确保唯一性
                    print(f"本地缓存包含 {stock_code} 的 {len(local_data)} 条数据，日期范围：{local_data['date'].min().strftime('%Y-%m-%d')} 至 {local_data['date'].max().strftime('%Y-%m-%d')}")
                else:
                    print(f"本地缓存文件 {cache_file_path} 为空。")
                    local_data = None # 视为空文件，后续逻辑会重新下载

            # 确定需要下载的日期范围
            download_ranges = []
            if local_data is not None and not local_data.empty:
                local_min_date = local_data['date'].min()
                local_max_date = local_data['date'].max()

                # 1. 请求范围在本地数据之前
                if req_end_date < local_min_date:
                    download_ranges.append((req_start_date, req_end_date))
                # 2. 请求范围在本地数据之后
                elif req_start_date > local_max_date:
                    download_ranges.append((req_start_date, req_end_date))
                # 3. 请求范围与本地数据有重叠
                else:
                    # 下载早于本地数据部分
                    if req_start_date < local_min_date:
                        download_ranges.append((req_start_date, local_min_date - pd.Timedelta(days=1)))
                    # 下载晚于本地数据部分
                    if req_end_date > local_max_date:
                        download_ranges.append((local_max_date + pd.Timedelta(days=1), req_end_date))
            else: # 本地无数据或数据为空
                download_ranges.append((req_start_date, req_end_date))

            all_downloaded_data = []
            if download_ranges:
                print(f"需要下载的数据范围: {[(s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d')) for s, e in download_ranges]}")
                
                # 登录BaoStock
                bs_result = bs.login()
                if bs_result.error_code != '0':
                    print(f"BaoStock登录失败: {bs_result.error_msg}")
                    return None # 或者根据策略抛出异常

                for dl_start, dl_end in download_ranges:
                    if dl_start > dl_end: # 避免无效下载范围
                        continue
                    
                    print(f"正在从BaoStock获取股票 {bs_stock_code} 的数据，范围: {dl_start.strftime('%Y-%m-%d')} 至 {dl_end.strftime('%Y-%m-%d')}...")
                    fields = "date,open,high,low,close,volume,amount,turn,pctChg"
                    rs = bs.query_history_k_data_plus(
                        bs_stock_code,
                        fields,
                        start_date=dl_start.strftime('%Y-%m-%d'),
                        end_date=dl_end.strftime('%Y-%m-%d'),
                        frequency="d",
                        adjustflag="2"
                    )

                    if rs.error_code != '0':
                        print(f"BaoStock获取数据失败 ({dl_start.strftime('%Y-%m-%d')} to {dl_end.strftime('%Y-%m-%d')}): {rs.error_msg}")
                        # 可以选择继续尝试其他范围，或者直接失败
                        continue
                    
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if data_list:
                        downloaded_df = pd.DataFrame(data_list, columns=rs.fields)
                        downloaded_df['date'] = pd.to_datetime(downloaded_df['date'])
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
                        for col in numeric_cols:
                            downloaded_df[col] = pd.to_numeric(downloaded_df[col], errors='coerce')
                        all_downloaded_data.append(downloaded_df)
                        new_data_downloaded = True
                
                bs.logout() # 完成所有下载后注销

            # 合并数据
            if all_downloaded_data:
                new_data_df = pd.concat(all_downloaded_data, ignore_index=True)
                if local_data is not None:
                    combined_data = pd.concat([local_data, new_data_df], ignore_index=True)
                else:
                    combined_data = new_data_df
                
                combined_data.sort_values(by='date', inplace=True)
                combined_data.drop_duplicates(subset=['date'], keep='first', inplace=True)
                
                # 更新本地缓存文件
                combined_data.to_csv(cache_file_path, index=False)
                print(f"数据已更新并缓存到: {cache_file_path}")
                local_data = combined_data # 更新 local_data 以供后续筛选
            
            # 如果没有下载新数据，且本地数据存在，则 local_data 就是最终使用的数据源
            # 如果下载了新数据，local_data 已经被更新为合并后的数据

            if local_data is None or local_data.empty:
                print(f"未能加载或下载到 {stock_code} 的任何数据。")
                return None

            # 从最终的 local_data (可能是原始缓存或合并后的数据) 中筛选请求的日期范围
            final_data = local_data[
                (local_data['date'] >= req_start_date) &
                (local_data['date'] <= req_end_date)
            ]

            if final_data.empty:
                 print(f"在日期范围 {start_date} 到 {end_date} 内没有找到 {stock_code} 的数据。")
                 # 即使本地有数据，但请求的范围可能无数据，例如请求了未来的日期或节假日
            else:
                print(f"成功加载 {stock_code} 的 {len(final_data)} 条数据，时间范围：{final_data['date'].min().strftime('%Y-%m-%d')} 至 {final_data['date'].max().strftime('%Y-%m-%d')}")
            
            return final_data.copy() # 返回副本以避免外部修改影响缓存

        except Exception as e:
            print(f"加载股票数据过程中发生错误: {e}")
            # 确保在异常时也尝试注销BaoStock
            try:
                if new_data_downloaded: # 只有在尝试过登录下载后才需要注销
                    bs.logout()
            except Exception as logout_e:
                print(f"注销BaoStock时发生错误: {logout_e}")
            return None
