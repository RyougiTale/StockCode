#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据config中的TRAINING_PHASE获取相应股票的历史K线数据
支持自定义起始和结束日期
"""
import time
import sys
import os
from datetime import datetime, timedelta

# 添加long_way目录到路径
long_way_dir = os.path.join(os.path.dirname(__file__), 'long_way')
sys.path.insert(0, long_way_dir)

# 导入配置和工具函数
import config
from stock_util import read_history_stock_by_code

def get_stock_list_by_training_phase():
    """
    根据config中的TRAINING_PHASE获取相应的股票列表
    
    Returns:
        list: 股票代码列表
    """
    print(f"当前训练阶段: {config.TRAINING_PHASE}")
    
    if config.TRAINING_PHASE == "pretraining":
        stock_codes = config.STOCK_CODES
        print(f"预训练模式 - 使用多股票列表: {len(stock_codes)} 只股票")
    elif config.TRAINING_PHASE == "finetuning":
        stock_codes = [config.FINETUNING_CONFIG["target_stock"]]
        print(f"微调模式 - 使用目标股票: {stock_codes[0]}")
    else:
        raise ValueError(f"无效的TRAINING_PHASE: {config.TRAINING_PHASE}")
    
    return stock_codes

def format_date_for_api(date_str):
    """
    将YYYY-MM-DD格式的日期转换为YYYYMMDD格式
    
    Args:
        date_str (str): YYYY-MM-DD格式的日期字符串
        
    Returns:
        str: YYYYMMDD格式的日期字符串
    """
    if date_str is None:
        return None
    
    try:
        # 解析并重新格式化日期
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%Y%m%d')
    except ValueError as e:
        print(f"日期格式错误: {date_str}，应该是YYYY-MM-DD格式")
        raise e

def update_training_stocks_data(start_date=None, end_date=None, sleep_seconds=3):
    """
    根据训练阶段获取相应股票的历史K线数据
    
    Args:
        start_date (str, optional): 起始日期，格式: 'YYYY-MM-DD'。默认为None（获取全部历史数据）
        end_date (str, optional): 结束日期，格式: 'YYYY-MM-DD'。默认为None（到最新数据）
        sleep_seconds (int): API调用间隔秒数，默认3秒
    """
    print("=" * 60)
    print(f"开始根据训练阶段更新股票历史K线数据...")
    print(f"起始日期: {start_date or '全部历史'}")
    print(f"结束日期: {end_date or '最新数据'}")
    print(f"API调用间隔: {sleep_seconds} 秒")
    print("=" * 60)
    
    # 1. 获取股票列表
    try:
        stock_codes = get_stock_list_by_training_phase()
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return
    
    if not stock_codes:
        print("股票列表为空，无法继续更新。")
        return
    
    # 2. 转换日期格式
    api_start_date = format_date_for_api(start_date) if start_date else None
    api_end_date = format_date_for_api(end_date) if end_date else None
    
    total = len(stock_codes)
    print(f"\\n准备更新 {total} 只股票的数据...")
    
    # 3. 遍历并更新每一只股票
    success_count = 0
    failed_stocks = []
    
    for i, code in enumerate(stock_codes):
        print(f"\\n--- [{i+1}/{total}] 正在更新: {code} ---")
        
        try:
            # 调用股票数据获取函数（使用转换后的日期格式）
            if api_start_date or api_end_date:
                print(f"  获取时间范围: {start_date or '最早'} ~ {end_date or '最新'}")
                data = read_history_stock_by_code(code, start_date=api_start_date or "19700101", end_date=api_end_date)
            else:
                data = read_history_stock_by_code(code)
            
            if data is not None and not data.empty:
                print(f"  --- {code} 更新成功 (共 {len(data)} 条记录) ---")
                success_count += 1
            else:
                print(f"  !!! {code} 返回空数据 !!!")
                failed_stocks.append(code)
                
        except Exception as e:
            print(f"  !!! 更新 {code} 失败: {e} !!!")
            failed_stocks.append(code)
        
        # 4. 暂停指定秒数，避免API调用过于频繁
        if i < total - 1:  # 最后一只股票更新完后不需要暂停
            print(f"  ...暂停{sleep_seconds}秒，防止API调用过于频繁...")
            time.sleep(sleep_seconds)
    
    # 5. 打印总结信息
    print("\\n" + "=" * 60)
    print("更新完成总结:")
    print(f"  训练阶段: {config.TRAINING_PHASE}")
    print(f"  成功更新: {success_count}/{total} 只股票")
    print(f"  失败数量: {len(failed_stocks)} 只股票")
    
    if failed_stocks:
        print(f"  失败股票: {', '.join(failed_stocks)}")
    
    print("=" * 60)

def update_recent_data(days=30, sleep_seconds=3):
    """
    更新最近N天的数据（便捷函数）
    
    Args:
        days (int): 最近天数，默认30天
        sleep_seconds (int): API调用间隔秒数
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"更新最近 {days} 天的数据...")
    update_training_stocks_data(start_date=start_date, end_date=end_date, sleep_seconds=sleep_seconds)

def update_year_data(year, sleep_seconds=3):
    """
    更新指定年份的数据（便捷函数）
    
    Args:
        year (int): 年份
        sleep_seconds (int): API调用间隔秒数
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    print(f"更新 {year} 年的数据...")
    update_training_stocks_data(start_date=start_date, end_date=end_date, sleep_seconds=sleep_seconds)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='根据训练阶段更新股票数据')
    parser.add_argument('--start', type=str, help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--sleep', type=int, default=3, help='API调用间隔秒数 (默认3秒)')
    parser.add_argument('--recent', type=int, help='更新最近N天的数据')
    parser.add_argument('--year', type=int, help='更新指定年份的数据')
    
    args = parser.parse_args()
    
    if args.recent:
        update_recent_data(days=args.recent, sleep_seconds=args.sleep)
    elif args.year:
        update_year_data(year=args.year, sleep_seconds=args.sleep)
    else:
        update_training_stocks_data(start_date=args.start, end_date=args.end, sleep_seconds=args.sleep)