import time
from index import read_sz50
from stock_util import read_history_stock_by_code

def update_sz50_kline_data():
    """
    获取上证50所有成分股的最新历史K线数据并缓存到本地数据库。
    在每次API调用之间会暂停10秒。
    """
    print("开始更新上证50成分股历史K线数据...")
    
    # 1. 获取上证50成分股列表
    sz50_stocks = read_sz50()
    if sz50_stocks.empty:
        print("获取上证50成分股列表失败，无法继续更新。")
        return
        
    stock_list = sz50_stocks[['stock_code', 'stock_name']].to_dict('records')
    total = len(stock_list)
    print(f"成功获取到 {total} 只上证50成分股。")
    
    # 2. 遍历并更新每一只股票
    for i, stock in enumerate(stock_list):
        code = stock['stock_code']
        name = stock['stock_name']
        print(f"\n--- [{i+1}/{total}] 正在更新: {name} ({code}) ---")
        
        try:
            # 调用已经写好的、带缓存的函数
            read_history_stock_by_code(code)
            print(f"--- {name} ({code}) 更新成功 ---")
        except Exception as e:
            print(f"!!! 更新 {name} ({code}) 失败: {e} !!!")
            
        # 3. 暂停10秒
        if i < total - 1: # 最后一只更新完后不需要暂停
            print("...暂停10秒，防止API调用过于频繁...")
            time.sleep(10)
            
    print("\n所有上证50成分股历史K线数据更新完成！")

if __name__ == "__main__":
    update_sz50_kline_data()
