import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta

class EtfKLineLoader:
    def __init__(self, output_dir='../data/etf_kline_data'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self._login_baostock()

    def _login_baostock(self):
        lg = bs.login()
        if lg.error_code != '0':
            print(f"登录baostock失败: {lg.error_msg}")
            raise ConnectionError("Baostock login failed.")
        print("Baostock login successful.")

    def _logout_baostock(self):
        bs.logout()
        print("Baostock logout successful.")

    def get_etf_kline_data(self, etf_code, start_date='2010-01-01', end_date=None, frequency='d', adjustflag='2'):
        """
        获取ETF的K线数据
        :param etf_code: ETF代码，例如 "sh.510300"
        :param start_date: 开始日期，格式 YYYY-MM-DD
        :param end_date: 结束日期，格式 YYYY-MM-DD，默认为昨天
        :param frequency: 数据频率，d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据
        :param adjustflag: 复权类型，默认不复权：3；1：后复权；2：前复权。ETF一般使用前复权或不复权。
        :return: pandas DataFrame
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        file_path = os.path.join(self.output_dir, f"{etf_code.replace('.', '_')}_{frequency}.csv")

        if os.path.exists(file_path):
            print(f"从本地文件加载数据: {file_path}")
            df = pd.read_csv(file_path)
            # 检查本地数据是否最新
            min_date_from_df_dt_safe = pd.NaT
            max_date_from_df_dt_safe = pd.NaT
            min_date_from_df_str_safe = "2999-12-31" # Default to "out of range" for start date
            max_date_from_df_str_safe = "1900-01-01" # Default to "out of range" for end date

            if not df.empty and 'date' in df.columns:
                date_series_temp = pd.to_datetime(df['date'], errors='coerce')
                
                _temp_min_dt = date_series_temp.min()
                if not pd.isna(_temp_min_dt):
                    min_date_from_df_dt_safe = _temp_min_dt
                    min_date_from_df_str_safe = min_date_from_df_dt_safe.strftime('%Y-%m-%d')
                else:
                    print(f"警告: 文件 {file_path} 中无有效最小日期。")
                
                _temp_max_dt = date_series_temp.max()
                if not pd.isna(_temp_max_dt):
                    max_date_from_df_dt_safe = _temp_max_dt
                    max_date_from_df_str_safe = max_date_from_df_dt_safe.strftime('%Y-%m-%d')
                else:
                    print(f"警告: 文件 {file_path} 中无有效最大日期。")
            else:
                print(f"警告: 文件 {file_path} 为空或缺少 'date' 列。")
            
            last_date_in_file = max_date_from_df_str_safe # Use the safe string

            if last_date_in_file >= end_date and min_date_from_df_str_safe <= start_date :
                 print(f"本地数据已是最新 ({last_date_in_file}) 且覆盖所需时间范围。")
                 df['date'] = pd.to_datetime(df['date'])
                 return df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

            if last_date_in_file < end_date:
                 if not pd.isna(max_date_from_df_dt_safe):
                     start_date_for_fetch = (max_date_from_df_dt_safe + timedelta(days=1)).strftime('%Y-%m-%d')
                     print(f"本地数据不是最新，从 {start_date_for_fetch} 开始更新数据...")
                 else:
                     print(f"由于本地最大日期无效，无法增量更新，将从 {start_date} 开始获取。")
                     start_date_for_fetch = start_date
            elif min_date_from_df_str_safe > start_date:
                 print(f"本地数据开始日期 ({min_date_from_df_str_safe}) 晚于请求的开始日期 ({start_date})，将重新下载所有数据从 {start_date} 开始。")
                 start_date_for_fetch = start_date
            else:
                 # This case implies local data exists, min_date is before or at start_date,
                 # and max_date is after or at end_date. This should have been caught by the first 'if'.
                 # Or, it implies dates were invalid and defaulted, leading here.
                 # Defaulting to fetching from start_date is safest if logic is unclear.
                 print(f"本地日期范围检查后，未明确更新或使用本地数据路径，默认从 {start_date} 开始获取。")
                 start_date_for_fetch = start_date

        else:
            print(f"本地文件不存在，将从网络下载数据: {etf_code}")
            start_date_for_fetch = start_date


        print(f"从baostock下载数据: {etf_code}, 开始日期: {start_date_for_fetch}, 结束日期: {end_date}")
        rs = bs.query_history_k_data_plus(
            etf_code,
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=start_date_for_fetch,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag
        )

        if rs.error_code != '0':
            print(f"获取K线数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        # 数据类型转换
        for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        result['date'] = pd.to_datetime(result['date'])


        # 使用之前安全获取的 min_date_from_df_str_safe 和 last_date_in_file (即 max_date_from_df_str_safe)
        # last_date_in_file 已经在前面被赋值为 max_date_from_df_str_safe
        # min_date_from_df_str_safe 也已在前面被安全赋值

        should_append = False
        should_overwrite_for_earlier_start = False

        if os.path.exists(file_path):
            # 检查是否需要追加数据 (本地最新日期 < 请求结束日期 AND 本地最早日期 <= 请求开始日期)
            # 并且本地日期是有效的
            if last_date_in_file < end_date and min_date_from_df_str_safe <= start_date and last_date_in_file != "1900-01-01" and min_date_from_df_str_safe != "2999-12-31":
                should_append = True
            # 检查是否因为请求的开始日期更早而需要覆盖 (本地最早日期 > 请求开始日期)
            # 并且本地日期是有效的
            elif min_date_from_df_str_safe > start_date and min_date_from_df_str_safe != "2999-12-31":
                should_overwrite_for_earlier_start = True
            # 如果本地日期无效，或者其他不明确的情况，倾向于覆盖
            elif last_date_in_file == "1900-01-01" or min_date_from_df_str_safe == "2999-12-31":
                 print(f"本地文件 {file_path} 中的日期范围无效或不完整，将用新下载的数据覆盖。")
                 # 此处不设置特定标志，后续逻辑会走到else（全新写入或覆盖）

        if should_append:
            print("追加新获取的数据到现有文件中。")
            existing_df = pd.read_csv(file_path)
            existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
            # 过滤掉新数据中在旧数据中已存在的日期，避免重复
            result_to_append = result[~result['date'].isin(existing_df['date'])]
            if not result_to_append.empty:
                combined_df = pd.concat([existing_df, result_to_append], ignore_index=True)
                combined_df.sort_values(by='date', inplace=True)
                combined_df.to_csv(file_path, index=False)
                print(f"数据已更新并保存到: {file_path}")
                df = combined_df
            else:
                print("没有新的不重复数据需要追加。")
                df = existing_df # 保持为旧数据
        elif should_overwrite_for_earlier_start:
            print(f"请求的开始日期 {start_date} 早于本地文件中的最早日期 {min_date_from_df_str_safe}，将用新下载的完整数据覆盖文件: {file_path}")
            result.sort_values(by='date', inplace=True)
            result.to_csv(file_path, index=False)
            print(f"数据已下载并保存到: {file_path}")
            df = result
        else: # 文件不存在，或本地日期无效，或不满足追加/特定覆盖条件，则直接写入/覆盖
            if os.path.exists(file_path):
                print(f"覆盖现有文件: {file_path} (原因：不满足追加条件或本地日期无效/不完整)。")
            else:
                print(f"文件不存在，直接保存新下载的数据到: {file_path}")
            result.sort_values(by='date', inplace=True)
            result.to_csv(file_path, index=False)
            print(f"数据已下载并保存到: {file_path}")
            df = result

        df['date'] = pd.to_datetime(df['date']) #确保date列是datetime类型
        return df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]


    def close(self):
        self._logout_baostock()

if __name__ == '__main__':
    # 示例：获取沪深300 ETF (sh.510300) 的日K线数据
    loader = EtfKLineLoader(output_dir='./data/etf_kline_data')
    try:
        # 首次下载或更新数据
        etf_data = loader.get_etf_kline_data(etf_code="sh.510300", start_date="2013-01-01")
        if not etf_data.empty:
            print("\n获取到的沪深300 ETF数据 (sh.510300) (前5行):")
            print(etf_data.head())
            print(f"\n数据条数: {len(etf_data)}")
            print(f"数据时间范围: {etf_data['date'].min()} to {etf_data['date'].max()}")

        # 再次获取，如果数据已存在且最新，则从本地加载
        etf_data_again = loader.get_etf_kline_data(etf_code="sh.510300", start_date="2023-01-01", end_date="2023-12-31")
        if not etf_data_again.empty:
            print("\n再次获取到的沪深300 ETF数据 (sh.510300) (2023年):")
            print(etf_data_again.head())
            print(f"\n数据条数: {len(etf_data_again)}")
            print(f"数据时间范围: {etf_data_again['date'].min()} to {etf_data_again['date'].max()}")

        # 测试获取一个不存在的ETF或者获取失败的情况
        # non_existent_etf = loader.get_etf_kline_data(etf_code="sh.999999")
        # if non_existent_etf.empty:
        #     print("\n尝试获取不存在的ETF数据，返回为空DataFrame，符合预期。")

    finally:
        loader.close()