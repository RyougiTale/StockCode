import sqlite3
import pandas as pd
import os


# 将数据目录放在Akshare文件夹内
# os.path.dirname(__file__) -> Akshare
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'stock_data.db')

def _init_db():
    """
    内部函数，用于在模块首次导入时初始化数据库和表。
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建指数成分股表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS index_constituents (
        index_code TEXT NOT NULL,
        stock_code TEXT NOT NULL,
        stock_name TEXT,
        update_date DATE NOT NULL,
        PRIMARY KEY (index_code, stock_code)
    )
    ''')
    
    # 创建股票日线数据表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_daily_kline (
        stock_code TEXT NOT NULL,
        date DATE NOT NULL,
        open REAL,
        close REAL,
        high REAL,
        low REAL,
        volume INTEGER,
        turnover REAL,
        amplitude REAL,
        pct_chg REAL,
        chg_amount REAL,
        turnover_rate REAL,
        PRIMARY KEY (stock_code, date)
    )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """获取数据库连接"""
    return sqlite3.connect(DB_PATH)

# --- 模块导入时执行的初始化代码 ---
_init_db()
# ------------------------------------

def save_dataframe(df, table_name, if_exists='replace'):
    """
    将DataFrame保存到指定的表中
    :param df: pandas DataFrame
    :param table_name: 表名
    :param if_exists: 如果表存在时的行为 ('fail', 'replace', 'append')
    """
    try:
        conn = get_db_connection()
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()
    except Exception as e:
        print(f"保存数据到表 '{table_name}' 失败: {e}")

def read_table(table_name, where_clause=None):
    """
    从指定表读取数据
    :param table_name: 表名
    :param where_clause: SQL的WHERE子句 (e.g., "stock_code = '000001'")
    :return: pandas DataFrame
    """
    try:
        conn = get_db_connection()
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"从表 '{table_name}' 读取数据失败: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print("这是一个数据库管理模块，请作为模块导入使用，不要直接运行。")