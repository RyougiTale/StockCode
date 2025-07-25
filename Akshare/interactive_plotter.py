import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from Akshare.index import read_sz50
from Akshare.db_manager import read_table

class StockChartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("上证50成分股走势分析")
        
        # --- Data ---
        self.all_data = None
        self.stock_name_map = {}
        self.highlighted_artist = None
        
        # --- GUI Setup ---
        self.setup_controls()
        self.setup_plot()
        
        # --- Initial Load ---
        self.load_all_data()
        self.plot_data(years=5, months=0) # 默认加载5年

    def setup_controls(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(control_frame, text="最近:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.years_var = tk.StringVar(value="5")
        self.years_entry = ttk.Entry(control_frame, textvariable=self.years_var, width=5)
        self.years_entry.pack(side=tk.LEFT)
        ttk.Label(control_frame, text="年").pack(side=tk.LEFT, padx=(0, 10))

        self.months_var = tk.StringVar(value="0")
        self.months_entry = ttk.Entry(control_frame, textvariable=self.months_var, width=5)
        self.months_entry.pack(side=tk.LEFT)
        ttk.Label(control_frame, text="月").pack(side=tk.LEFT, padx=(0, 10))

        self.update_button = ttk.Button(control_frame, text="更新图表", command=self.update_plot)
        self.update_button.pack(side=tk.LEFT)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def load_all_data(self):
        print("正在从数据库加载所有上证50历史数据...")
        sz50_stocks = read_sz50()
        if sz50_stocks.empty:
            print("获取上证50列表失败")
            return
        stock_codes = tuple(sz50_stocks['stock_code'])
        codes_str = ','.join(f"'{code}'" for code in stock_codes)
        self.stock_name_map = sz50_stocks.set_index('stock_code')['stock_name'].to_dict()
        self.all_data = read_table('stock_daily_kline', where_clause=f"stock_code IN ({codes_str})")
        if self.all_data is not None and not self.all_data.empty:
            self.all_data['date'] = pd.to_datetime(self.all_data['date'])
            print("数据加载完成。")
        else:
            print("数据库中无数据。")

    def update_plot(self):
        try:
            years = int(self.years_var.get())
            months = int(self.months_var.get())
            self.plot_data(years, months)
        except ValueError:
            print("请输入有效的年和月数值。")

    def plot_data(self, years, months):
        if self.all_data is None or self.all_data.empty:
            self.ax.clear()
            self.ax.text(0.5, 0.5, '无数据可供显示', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        print(f"开始绘制最近 {years} 年 {months} 月的数据...")
        self.ax.clear()
        self.highlighted_artist = None # 重置高亮状态

        # 1. 筛选数据
        start_date = datetime.now() - relativedelta(years=years, months=months)
        recent_data = self.all_data[self.all_data['date'] >= start_date].copy()
        if recent_data.empty:
            self.ax.text(0.5, 0.5, '指定时间范围内无数据', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        # 2. 重塑 & 归一化
        close_prices = recent_data.pivot(index='date', columns='stock_code', values='close')
        normalized_prices = close_prices.apply(lambda x: x / x.dropna().iloc[0])
        mean_normalized_prices = normalized_prices.mean(axis=1)

        # 3. 绘图
        lines = self.ax.plot(normalized_prices.index, normalized_prices.values, alpha=0.3, linewidth=1)
        for i, line in enumerate(lines):
            line.set_label(normalized_prices.columns[i])

        mean_line, = self.ax.plot(mean_normalized_prices.index, mean_normalized_prices.values, color='black', linewidth=2.5)
        mean_line.set_label("平均走势")

        # 4. 设置图表
        self.ax.set_title(f'上证50成分股最近 {years} 年 {months} 月归一化走势', fontsize=16)
        self.ax.set_xlabel('日期')
        self.ax.set_ylabel('归一化价格')
        self.ax.grid(True)
        
        # 5. 添加交互式光标和高亮效果
        cursor = mplcursors.cursor(self.ax.get_lines(), hover=True)
        @cursor.connect("add")
        def on_add(sel):
            if self.highlighted_artist and self.highlighted_artist != sel.artist:
                is_mean = self.highlighted_artist.get_label() == "平均走势"
                self.highlighted_artist.set_alpha(0.3 if not is_mean else 1.0)
                self.highlighted_artist.set_linewidth(1 if not is_mean else 2.5)
            
            sel.artist.set_alpha(1.0)
            sel.artist.set_linewidth(2.0 if sel.artist.get_label() != "平均走势" else 3.0)
            self.highlighted_artist = sel.artist
            
            stock_code = sel.artist.get_label()
            if stock_code == "平均走势":
                sel.annotation.set_text("平均走势")
            else:
                stock_name = self.stock_name_map.get(stock_code, "")
                sel.annotation.set_text(f"{stock_code}\n{stock_name}")
            
            sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)
            sel.annotation.get_bbox_patch().set(facecolor="lightblue", alpha=0.7)
            
            self.canvas.draw_idle()

        self.canvas.draw()
        print("图表更新完成。")

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    root = tk.Tk()
    app = StockChartApp(root)
    root.mainloop()