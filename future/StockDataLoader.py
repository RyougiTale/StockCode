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
        """从BaoStock加载真实股票数据
        
        Args:
            stock_code: 股票代码，如 '600000'（上证）或 '000001'（深证）
            start_date: 起始日期，如 '2020-01-01'
            end_date: 结束日期，默认是今天
            
        Returns:
            DataFrame: 股票数据，包含日期、开盘价、收盘价、最高价、最低价、成交量等
        """
        try:
            # 默认日期设置
            if not start_date:
                raise ValueError("start_date is required")
            if not end_date:
                raise ValueError("end_date is required")
                # end_date = datetime.now().strftime('%Y-%m-%d')

            # 构造缓存文件名和路径
            # 确保 start_date 和 end_date 中的 ':' (如果存在于时间部分) 替换掉，避免文件名问题
            # BaoStock的日期通常不含时间，但以防万一
            safe_start_date = start_date.replace(":", "-")
            safe_end_date = end_date.replace(":", "-")
            cache_filename = f"{stock_code}_{safe_start_date}_{safe_end_date}.csv"
            cache_file_path = os.path.join(self.data_dir, cache_filename)

            # 检查缓存文件是否存在
            if os.path.exists(cache_file_path):
                try:
                    print(f"从缓存文件加载数据: {cache_file_path}")
                    # 尝试将名为 'date' 的列解析为日期时间（datetime）对象
                    data = pd.read_csv(cache_file_path, parse_dates=['date'])
                    # 基本的数据类型转换，确保和新下载的数据一致
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
                    for col in numeric_columns:
                        if col in data.columns:
                            # 当尝试将一个列的数据转换为数值类型时，如果遇到无法转换的值（比如一个字符串混在了数字中间），Pandas 不会报错并停止执行，而是会将这个无法转换的值替换为 NaN (Not a Number)。
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                    print(f"成功从缓存加载{stock_code}的{len(data)}条数据，时间范围：{data['date'].min()} 至 {data['date'].max()}")
                    return data
                except Exception as e:
                    print(f"加载缓存文件 {cache_file_path} 失败: {e}. 将尝试重新下载。")
            
            # 调整股票代码格式（BaoStock需要前缀：sh或sz）
            bs_stock_code = stock_code
            if stock_code.startswith('6'):
                bs_stock_code = f'sh.{stock_code}'
            elif stock_code.startswith('0') or stock_code.startswith('3'):
                bs_stock_code = f'sz.{stock_code}'
            else:
                raise ValueError("invalid code(not starts with 6/0/3)")
            
            print(f"正在从BaoStock获取股票{bs_stock_code}的数据...")
            
            # 登录BaoStock
            bs_result = bs.login()
            if bs_result.error_code != '0':
                print(f"BaoStock登录失败: {bs_result.error_msg}")
                return None
            
            # 获取股票日K线数据
            fields = "date,open,high,low,close,volume,amount,turn,pctChg"
            rs = bs.query_history_k_data_plus(
                bs_stock_code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",  # 日线
                adjustflag="2"  # 复权类型，2表示前复权，更适合预测分析
            )
            
            if rs.error_code != '0':
                print(f"BaoStock获取数据失败: {rs.error_msg}")
                bs.logout()
                return None
            
            # 将数据转换为DataFrame
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            # 数据处理
            data = pd.DataFrame(data_list, columns=rs.fields)
            
            # 转换数据类型
            data['date'] = pd.to_datetime(data['date'])
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 注销BaoStock
            bs.logout()
            
            # 保存数据到新的缓存文件
            if not os.path.exists(self.data_dir): # self.data_dir 的创建已在 __init__ 中保证
                os.makedirs(self.data_dir) # 再次检查以防万一，但通常不需要
            data.to_csv(cache_file_path, index=False) # 使用包含日期的缓存文件名
            print(f"数据已缓存到: {cache_file_path}")
            
            print(f"成功从BaoStock获取{stock_code}的{len(data)}条股票数据，时间范围：{data['date'].min()} 至 {data['date'].max()}")
            return data
            
        except Exception as e:
            print(f"加载股票数据失败: {e}")
            # 确保注销BaoStock
            try:
                bs.logout()
            except:
                pass
            return None
