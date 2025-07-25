# 指数
import akshare as ak
import pandas as pd
from datetime import date
from Akshare.db_manager import read_table, save_dataframe, get_db_connection

# 所有成分
def read_all_index():
    info = ak.index_stock_info()
    print(info)

def _read_index_constituents(index_code: str, index_name: str) -> pd.DataFrame:
    """
    读取指数成分股的帮助函数，实现“先查本地，再查网络”的缓存逻辑。
    """
    today_str = date.today().strftime('%Y-%m-%d')
    
    # 1. 尝试从数据库读取今天的数据
    local_data = read_table('index_constituents', where_clause=f"index_code = '{index_code}' AND update_date = '{today_str}'")
    if not local_data.empty:
        print(f"从本地数据库加载 {index_name} ({index_code}) 成分股。")
        return local_data

    # 2. 如果本地没有最新数据，则从网络获取
    print(f"本地数据不是最新, 从网络获取 {index_name} ({index_code}) 成分股...")
    try:
        df = ak.index_stock_cons(symbol=index_code)
        if df is None or df.empty:
            print(f"从网络获取 {index_name} 数据为空。")
            return pd.DataFrame()

        # 增加数据清洗步骤：去重
        if df.duplicated(subset=['品种代码']).any():
            print(f"警告：从网络获取的 {index_name} 数据中发现重复项，将自动去重。")
            df.drop_duplicates(subset=['品种代码'], keep='first', inplace=True)

        # 3. 处理数据并存入数据库
        df['update_date'] = today_str
        df['index_code'] = index_code
        df.rename(columns={'品种代码': 'stock_code', '品种名称': 'stock_name'}, inplace=True)
        df_to_save = df[['index_code', 'stock_code', 'stock_name', 'update_date']]
        
        # 更新数据库：使用一个事务来保证操作的原子性
        # 先删除该指数当天的旧记录，再插入新记录
        conn = get_db_connection()
        try:
            with conn:
                # 更新逻辑：删除该指数的所有旧记录，然后插入最新的记录
                conn.execute("DELETE FROM index_constituents WHERE index_code = ?", (index_code,))
                df_to_save.to_sql('index_constituents', conn, if_exists='append', index=False)
            print(f"成功更新本地数据库中的 {index_name} 成分股。")
        except Exception as db_e:
            print(f"更新数据库失败: {db_e}")
        finally:
            conn.close()
        
        return df_to_save
    except Exception as e:
        print(f"获取 {index_name} 数据失败，错误信息：{e}")
        return pd.DataFrame()

def read_hs300():
    """获取沪深300成分股 (带缓存)"""
    return _read_index_constituents('000300', '沪深300')

def read_sz50():
    """获取上证50成分股 (带缓存)"""
    return _read_index_constituents('000016', '上证50')
