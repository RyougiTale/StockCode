from Util.StockDataLoader import StockDataLoader
import os


DATA_CONFIG = {
    'data_dir': './data',
    'save_model_dir': './transformer/models',
    'stock_code': '600036',

    # 训练集时段
    'train_start_date': '2008-01-01',
    'train_end_date': '2021-06-30',

    # 验证集时段 (紧跟训练集之后)
    'validation_start_date': '2023-07-01',
    'validation_end_date': '2024-07-01',

    # 评估/回测集时段 (紧跟验证集之后)
    'eval_start_date': '2023-07-01',
    'eval_end_date': '2025-05-21',
}

if __name__ == "__main__":
    os.makedirs(DATA_CONFIG['save_model_dir'], exist_ok=True)
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    

    # stock_szse_sector_summary_df = ak.stock_szse_sector_summary(symbol="当年", date="202501")
    # print(stock_szse_sector_summary_df)
    # stock_szse_summary_df = ak.stock_szse_summary(date="20200619")
    # print(stock_szse_summary_df)
    # stock_sse_deal_daily_df = ak.stock_sse_deal_daily(date="20250221")
    # print(stock_sse_deal_daily_df)
    # macro_bank_usa_interest_rate_df = ak.macro_bank_usa_interest_rate()
    # print(macro_bank_usa_interest_rate_df)
    
    
    
    

