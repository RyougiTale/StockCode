import unittest
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from Akshare.index import read_sz50
from Akshare.db_manager import read_table
from datetime import datetime, timedelta

class TestVisualization(unittest.TestCase):
    def test_plot_sz50_close_prices_all_history(self):
        """
        从本地数据库读取所有上证50成分股的完整历史收盘价，
        进行归一化处理后，绘制走势对比图，并附上平均值走势。
        """
        print("\n--- 单元测试：绘制上证50成分股【全历史】归一化走势图 ---")

        # 1. 获取上证50成分股列表
        sz50_stocks = read_sz50()
        self.assertFalse(sz50_stocks.empty, "获取上证50成分股列表失败。")
        stock_codes = tuple(sz50_stocks['stock_code'])

        # 2. 一次性从数据库读取所有相关股票的数据
        print(f"正在从数据库一次性读取 {len(stock_codes)} 只股票的历史数据...")
        codes_str = ','.join(f"'{code}'" for code in stock_codes)
        all_data = read_table('stock_daily_kline', where_clause=f"stock_code IN ({codes_str})")
        self.assertFalse(all_data.empty, "未能从数据库读取到任何K线数据，请先运行period_update.py填充数据。")
        print("数据读取完成。")

        # 3. 数据重塑：将长表转换为宽表
        print("正在进行数据重塑...")
        close_prices = all_data.pivot(index='date', columns='stock_code', values='close')
        close_prices.index = pd.to_datetime(close_prices.index) # 确保索引是日期类型

        # 4. 归一化处理
        print("正在进行归一化处理...")
        # 使用每列的第一个有效值进行归一化
        normalized_prices = close_prices.apply(lambda x: x / x.dropna().iloc[0])

        # 5. 计算平均值走势
        mean_normalized_prices = normalized_prices.mean(axis=1)

        # 6. 绘图
        print("正在绘图...")
        plt.style.use('seaborn-v0_8-whitegrid') # 使用一个好看的样式
        fig, ax = plt.subplots(figsize=(15, 8))

        # 绘制所有个股的走势（半透明）
        normalized_prices.plot(ax=ax, legend=False, alpha=0.3, linewidth=1)

        # 绘制平均值走势（加粗，黑色）
        mean_normalized_prices.plot(ax=ax, color='black', linewidth=2.5, label='上证50平均走势')

        # 设置matplotlib支持中文显示（提前设置）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 设置图表属性
        ax.set_title('上证50成分股【全历史】归一化收盘价走势', fontsize=16)
        ax.set_xlabel('日期')
        ax.set_ylabel('归一化价格（以上市首日为1）')
        ax.legend()
        ax.grid(True)
        
        # 添加交互式光标
        cursor = mplcursors.cursor(ax.get_lines(), hover=True)
        @cursor.connect("add")
        def on_add(sel):
            # sel.target.index 是日期, sel.target[1] 是y值
            # sel.artist.get_label() 是曲线的标签（即股票代码）
            sel.annotation.set_text(sel.artist.get_label())
            sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)
            sel.annotation.get_bbox_patch().set(facecolor="lightblue", alpha=0.7)

        fig.tight_layout()

        print("正在打开matplotlib图表窗口... 关闭窗口后测试将结束。")
        plt.show()
        print("图表窗口已关闭。")

    def test_plot_sz50_close_prices_last_5_years(self):
        """
        绘制上证50成分股【最近5年】的归一化收盘价走势图。
        """
        print("\n--- 单元测试：绘制上证50成分股【最近5年】归一化走势图 ---")

        # 1. 获取上证50成分股列表
        sz50_stocks = read_sz50()
        self.assertFalse(sz50_stocks.empty, "获取上证50成分股列表失败。")
        stock_codes = tuple(sz50_stocks['stock_code'])

        # 2. 一次性从数据库读取所有相关股票的数据
        print(f"正在从数据库一次性读取 {len(stock_codes)} 只股票的历史数据...")
        codes_str = ','.join(f"'{code}'" for code in stock_codes)
        all_data = read_table('stock_daily_kline', where_clause=f"stock_code IN ({codes_str})")
        self.assertFalse(all_data.empty, "未能从数据库读取到任何K线数据，请先运行period_update.py填充数据。")
        print("数据读取完成。")
        
        # 3. 筛选最近5年的数据
        five_years_ago = datetime.now() - timedelta(days=5*365)
        all_data['date'] = pd.to_datetime(all_data['date']) # 确保date列是datetime类型
        recent_data = all_data[all_data['date'] >= five_years_ago].copy()
        self.assertFalse(recent_data.empty, "数据库中没有最近5年的数据。")
        print(f"已筛选出 {recent_data['date'].min().strftime('%Y-%m-%d')}至今的数据。")

        # 4. 数据重塑
        close_prices = recent_data.pivot(index='date', columns='stock_code', values='close')

        # 5. 归一化处理
        normalized_prices = close_prices.apply(lambda x: x / x.dropna().iloc[0])

        # 6. 计算平均值走势
        mean_normalized_prices = normalized_prices.mean(axis=1)

        # 7. 绘图
        print("正在绘图...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 设置matplotlib支持中文显示（提前设置）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制所有个股的走势（半透明）
        normalized_prices.plot(ax=ax, legend=False, alpha=0.3, linewidth=1)

        # 绘制平均值走势（加粗，黑色）
        mean_normalized_prices.plot(ax=ax, color='black', linewidth=2.5, label='上证50平均走势')

        # 设置图表属性
        ax.set_title('上证50成分股【最近5年】归一化收盘价走势', fontsize=16)
        ax.set_xlabel('日期')
        ax.set_ylabel('归一化价格（以5年前首日为1）')
        ax.legend()
        ax.grid(True)
        
        # 添加交互式光标
        cursor = mplcursors.cursor(ax.get_lines(), hover=True)
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(sel.artist.get_label())
            sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)
            sel.annotation.get_bbox_patch().set(facecolor="lightblue", alpha=0.7)

        fig.tight_layout()

        print("正在打开matplotlib图表窗口... 关闭窗口后测试将结束。")
        plt.show()
        print("图表窗口已关闭。")

if __name__ == '__main__':
    unittest.main()