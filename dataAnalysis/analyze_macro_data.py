import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_macro_data(csv_file_path='../data/macro_data/money_supply_monthly.csv'):
    """
    读取宏观经济数据CSV文件并绘制相关图表。
    """
    if not os.path.exists(csv_file_path):
        print(f"错误：数据文件 {csv_file_path} 未找到。")
        print("请先运行 Util/MacroDataLoader.py 来下载数据。")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"读取CSV文件 {csv_file_path} 时出错: {e}")
        return

    if 'date' not in df.columns:
        print("错误：CSV文件中缺少 'date' 列。")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 检查必要的列是否存在
    required_value_cols = ['m0Month', 'm1Month', 'm2Month']
    required_yoy_cols = ['m0YOY', 'm1YOY', 'm2YOY']
    required_chain_cols = ['m0ChainRelative', 'm1ChainRelative', 'm2ChainRelative']

    missing_cols = [col for col in required_value_cols + required_yoy_cols + required_chain_cols if col not in df.columns]
    if missing_cols:
        print(f"警告：数据中缺少以下列，可能无法绘制所有图表: {', '.join(missing_cols)}")

    # 1. 绘制 M0, M1, M2 供应量
    plot_cols_m_values = [col for col in required_value_cols if col in df.columns]
    if plot_cols_m_values:
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        for col in plot_cols_m_values:
            ax1.plot(df.index, df[col], label=col)
        ax1.set_title('货币供应量 (M0, M1, M2)')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('供应量 (亿元)')
        ax1.legend()
        ax1.grid(True)
        # 使用 FuncFormatter 自定义 Y 轴刻度标签格式，避免科学计数法并添加单位
        formatter = mticker.FuncFormatter(lambda x, p: format(int(x), ','))
        ax1.yaxis.set_major_formatter(formatter)
        plt.tight_layout()
    else:
        print("缺少 M0/M1/M2 月度数据列，无法绘制供应量图。")


    # 2. 绘制 M0, M1, M2 同比增长率 (YOY)
    plot_cols_yoy = [col for col in required_yoy_cols if col in df.columns]
    if plot_cols_yoy:
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        for col in plot_cols_yoy:
            ax2.plot(df.index, df[col], label=col)
        ax2.set_title('货币供应量同比增长率 (YOY)')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('同比增长率 (%)')
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--') # 添加0%参考线
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
    else:
        print("缺少 M0/M1/M2 同比增长率数据列，无法绘制YOY图。")

    # 3. 绘制 M0, M1, M2 环比增长率 (Chain Relative)
    plot_cols_chain = [col for col in required_chain_cols if col in df.columns]
    if plot_cols_chain:
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        for col in plot_cols_chain:
            ax3.plot(df.index, df[col], label=col)
        ax3.set_title('货币供应量环比增长率 (Chain Relative)')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('环比增长率 (%)')
        ax3.axhline(0, color='black', linewidth=0.5, linestyle='--') # 添加0%参考线
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
    else:
        print("缺少 M0/M1/M2 环比增长率数据列，无法绘制环比图。")
        
    if any([plot_cols_m_values, plot_cols_yoy, plot_cols_chain]):
        plt.show()
    else:
        print("没有足够的数据列来绘制任何图表。")

if __name__ == '__main__':
    # 假设 MacroDataLoader.py 脚本已经运行并将数据保存在默认位置
    # ../ 指的是从 dataAnalysis 目录返回上一级到项目根目录，然后再进入 data/macro_data
    csv_path = './data/macro_data/money_supply_monthly.csv'
    plot_macro_data(csv_path)