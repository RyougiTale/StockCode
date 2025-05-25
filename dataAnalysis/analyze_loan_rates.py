import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_loan_rate_data(csv_file_path='../data/macro_data/loan_rates.csv'):
    """
    读取贷款利率数据CSV文件并绘制相关图表。
    """
    if not os.path.exists(csv_file_path):
        print(f"错误：数据文件 {csv_file_path} 未找到。")
        print("请先运行 Util/LoanRateLoader.py 来下载数据。")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"读取CSV文件 {csv_file_path} 时出错: {e}")
        return

    if 'pubDate' not in df.columns:
        print("错误：CSV文件中缺少 'pubDate' 列。")
        return
        
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df.set_index('pubDate', inplace=True)
    df.sort_index(inplace=True)

    # 定义要绘制的列组
    commercial_loan_cols = [
        'loanRate6Month', 'loanRate6MonthTo1Year', 'loanRate1YearTo3Year', 
        'loanRate3YearTo5Year', 'loanRateAbove5Year'
    ]
    mortgage_loan_cols = [
        'mortgateRateBelow5Year', 'mortgateRateAbove5Year'
    ]

    # 检查列是否存在
    available_commercial_cols = [col for col in commercial_loan_cols if col in df.columns]
    available_mortgage_cols = [col for col in mortgage_loan_cols if col in df.columns]

    plots_drawn = 0

    # 1. 绘制各期限商业贷款利率
    if available_commercial_cols:
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        for col in available_commercial_cols:
            ax1.plot(df.index, df[col], label=col, marker='.', linestyle='-')
        ax1.set_title('各期限商业贷款利率')
        ax1.set_xlabel('发布日期')
        ax1.set_ylabel('利率 (%)')
        ax1.legend(loc='best')
        ax1.grid(True)
        plt.tight_layout()
        plots_drawn +=1
    else:
        print("缺少足够的商业贷款利率数据列，无法绘制图1。")

    # 2. 绘制住房公积金贷款利率
    if available_mortgage_cols:
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        for col in available_mortgage_cols:
            if df[col].notna().any(): # 确保列中有数据
                ax2.plot(df.index, df[col], label=col, marker='.', linestyle='-')
        
        if any(df[col].notna().any() for col in available_mortgage_cols):
            ax2.set_title('住房公积金贷款利率')
            ax2.set_xlabel('发布日期')
            ax2.set_ylabel('利率 (%)')
            ax2.legend(loc='best')
            ax2.grid(True)
            plt.tight_layout()
            plots_drawn +=1
        else:
            print("住房公积金贷款利率数据列全为空或不存在，无法绘制图2。")
            if fig2: # 关闭可能已创建的空图形
                 plt.close(fig2)
    else:
        print("缺少住房公积金贷款利率数据列，无法绘制图2。")
        
    if plots_drawn > 0:
        plt.show()
    else:
        print("没有足够的数据列来绘制任何贷款利率图表。")

if __name__ == '__main__':
    # 假设 LoanRateLoader.py 脚本已经运行并将数据保存在默认位置
    csv_path = './data/macro_data/loan_rates.csv'
    plot_loan_rate_data(csv_path)