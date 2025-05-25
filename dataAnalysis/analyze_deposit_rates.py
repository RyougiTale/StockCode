import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_deposit_rate_data(csv_file_path='../data/macro_data/deposit_rates.csv'):
    """
    读取存款利率数据CSV文件并绘制相关图表。
    """
    if not os.path.exists(csv_file_path):
        print(f"错误：数据文件 {csv_file_path} 未找到。")
        print("请先运行 Util/DepositRateLoader.py 来下载数据。")
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
    fixed_deposit_cols = [
        'demandDepositRate', 'fixedDepositRate3Month', 'fixedDepositRate6Month', 
        'fixedDepositRate1Year', 'fixedDepositRate2Year', 'fixedDepositRate3Year', 
        'fixedDepositRate5Year'
    ]
    installment_deposit_cols = [
        'installmentFixedDepositRate1Year', 'installmentFixedDepositRate3Year', 
        'installmentFixedDepositRate5Year'
    ]

    # 检查列是否存在
    available_fixed_cols = [col for col in fixed_deposit_cols if col in df.columns]
    available_installment_cols = [col for col in installment_deposit_cols if col in df.columns]

    plots_drawn = 0

    # 1. 绘制活期及各期限定期存款利率
    if available_fixed_cols:
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        for col in available_fixed_cols:
            ax1.plot(df.index, df[col], label=col, marker='.', linestyle='-') # 添加标记以便区分变化点
        ax1.set_title('活期及定期存款利率')
        ax1.set_xlabel('发布日期')
        ax1.set_ylabel('利率 (%)')
        ax1.legend(loc='best')
        ax1.grid(True)
        plt.tight_layout()
        plots_drawn +=1
    else:
        print("缺少足够的活期或定期存款利率数据列，无法绘制图1。")

    # 2. 绘制零存整取等存款利率
    if available_installment_cols:
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        for col in available_installment_cols:
            # 过滤掉完全是 NaN 的列，避免绘图错误或空图例
            if df[col].notna().any():
                 ax2.plot(df.index, df[col], label=col, marker='.', linestyle='-')
        
        # 检查是否有实际绘制的线条，才显示图例和标题
        if any(df[col].notna().any() for col in available_installment_cols):
            ax2.set_title('零存整取、整存零取、存本取息定期存款利率')
            ax2.set_xlabel('发布日期')
            ax2.set_ylabel('利率 (%)')
            ax2.legend(loc='best')
            ax2.grid(True)
            plt.tight_layout()
            plots_drawn +=1
        else:
            print("零存整取等存款利率数据列全为空，无法绘制图2。")
            if fig2: # 关闭可能已创建的空图形
                plt.close(fig2)
    else:
        print("缺少零存整取等存款利率数据列，无法绘制图2。")
        
    if plots_drawn > 0:
        plt.show()
    else:
        print("没有足够的数据列来绘制任何存款利率图表。")

if __name__ == '__main__':
    # 假设 DepositRateLoader.py 脚本已经运行并将数据保存在默认位置
    csv_path = './data/macro_data/deposit_rates.csv'
    plot_deposit_rate_data(csv_path)