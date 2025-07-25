import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import akshare as ak
from datetime import datetime
from dateutil.relativedelta import relativedelta

class SingleStockChartApp:
    def __init__(self, root, stock_code):
        self.root = root
        self.stock_code = stock_code
        self.root.title(f"{self.stock_code} - 股价走势分析 (不复权)")
        
        # --- Data ---
        self.all_data = None
        
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
        # 创建两个子图，共享X轴
        self.fig, (self.ax_price, self.ax_volume) = plt.subplots(2, 1, sharex=True, figsize=(15, 9), gridspec_kw={'height_ratios': [3, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def load_all_data(self):
        print(f"正在从网络加载 {self.stock_code} 的全部不复权历史数据...")
        try:
            self.all_data = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily", adjust="")
            if self.all_data is not None and not self.all_data.empty:
                self.all_data['日期'] = pd.to_datetime(self.all_data['日期'])
                self.all_data.set_index('日期', inplace=True)
                print("数据加载完成。")
            else:
                print("未能获取到数据。")
        except Exception as e:
            print(f"获取数据失败: {e}")
            self.all_data = None

    def update_plot(self):
        try:
            years = int(self.years_var.get())
            months = int(self.months_var.get())
            self.plot_data(years, months)
        except ValueError:
            print("请输入有效的年和月数值。")

    def plot_data(self, years, months):
        self.ax_price.clear()
        self.ax_volume.clear()

        if self.all_data is None or self.all_data.empty:
            self.ax_price.text(0.5, 0.5, '无数据可供显示', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        print(f"开始绘制最近 {years} 年 {months} 月的数据...")
        
        # 1. 筛选数据
        start_date = datetime.now() - relativedelta(years=years, months=months)
        recent_data = self.all_data[self.all_data.index >= start_date].copy()
        
        if recent_data.empty:
            self.ax_price.text(0.5, 0.5, '指定时间范围内无数据', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        # 2. 绘图
        # 绘制收盘价
        self.ax_price.plot(recent_data.index, recent_data['收盘'], label='收盘价', color='blue')
        self.ax_price.set_title(f'{self.stock_code} 股价走势 (不复权)', fontsize=16)
        self.ax_price.set_ylabel('价格 (元)')
        self.ax_price.grid(True)
        self.ax_price.legend()

        # 绘制成交量
        self.ax_volume.bar(recent_data.index, recent_data['成交量'], label='成交量', color='grey', alpha=0.6)
        self.ax_volume.set_ylabel('成交量 (手)')
        self.ax_volume.set_xlabel('日期')
        self.ax_volume.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
        print("图表更新完成。")

if __name__ == '__main__':
    # --- 在这里修改为您想查看的股票代码 ---
    STOCK_TO_PLOT = "600597" 
    # ------------------------------------

    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    root = tk.Tk()
    app = SingleStockChartApp(root, stock_code=STOCK_TO_PLOT)
    root.mainloop()