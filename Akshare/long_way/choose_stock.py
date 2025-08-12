import akshare as ak
import pandas as pd
pd.set_option('display.width', 1000) 
pd.set_option('display.max_rows', None) 
try:
    # 获取沪深100指数的成分股列表
    csi_100_stocks_df = ak.index_stock_cons(symbol="000903")
    
    print("成功获取沪深100成分股！")
    print(f"成分股数量: {len(csi_100_stocks_df)}")
    print(csi_100_stocks_df)
    
    # 提取股票代码列表
    stock_code_list = csi_100_stocks_df['品种代码'].tolist()
    print("\n部分成分股代码:")
    print(stock_code_list[:100])

except Exception as e:
    print(f"获取沪深100成分股时出错: {e}")