def get_all_etf():
    import akshare as ak

    try:
        # 获取所有ETF的实时行情数据
        etf_spot_df = ak.fund_etf_spot_em()
        
        print("成功获取所有ETF的实时行情！")
        print(f"共找到 {len(etf_spot_df)} 只ETF。")
        
        # 打印前5只ETF的信息
        print("\n前5只ETF行情示例:")
        print(etf_spot_df.head())
        
        # 您可以根据代码或名称进行筛选，比如找到“沪深300”相关的ETF
        hs300_etfs = etf_spot_df[etf_spot_df['名称'].str.contains('沪深300')]
        print("\n筛选出的部分沪深300 ETF:")
        print(hs300_etfs[['代码', '名称', '最新价', '涨跌幅', '成交额']])
        
    except Exception as e:
        print(f"获取数据时出错: {e}")
        
        
        
import akshare as ak

# 设置要查询的ETF代码和日期范围
etf_code = "510300"  # 华泰柏瑞沪深300ETF
start_date = "19700101"
end_date = "20250804" # 您可以设置为今天的日期

try:
    print(f"正在获取 {etf_code} 的前复权日K线数据...")
    
    # 获取前复权（qfq）的日K线数据
    etf_hist_df = ak.fund_etf_hist_em(
        symbol=etf_code, 
        period="daily", 
        start_date=start_date, 
        end_date=end_date, 
        adjust="qfq"
    )
    
    print("数据获取成功！")
    print(etf_hist_df.head()) # 打印前几行
    print("\n...")
    print(etf_hist_df.tail()) # 打印后几行

    # 返回的列名已经是标准的OHLCV格式，可以直接用于您的模型
    # 列名: '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
    
except Exception as e:
    print(f"获取历史数据时出错: {e}")