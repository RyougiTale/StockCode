import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import argparse

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_index_kline_data(index_code="sh.000300", frequency="d", data_dir='../data/index_kline_data'):
    """
    读取指数K线数据CSV文件并绘制相关图表。
    index_code: 指数代码, e.g., "sh.000300"
    frequency: 数据频率, e.g., "d" for daily
    data_dir: 存放CSV文件的目录
    """
    csv_file_name = f"{index_code.replace('.', '_')}_{frequency}.csv"
    csv_file_path = os.path.join(data_dir, csv_file_name)

    if not os.path.exists(csv_file_path):
        print(f"错误：数据文件 {csv_file_path} 未找到。")
        print(f"请先运行 Util/IndexKLineLoader.py 来下载 {index_code} ({frequency}) 的数据。")
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

    plots_drawn = 0
    plot_title_suffix = f" ({index_code} - {frequency})"

    # 1. 绘制收盘价
    if 'close' in df.columns:
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(df.index, df['close'], label='收盘价 (Close)', color='blue')
        ax1.set_title('指数收盘价' + plot_title_suffix)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('价格')
        ax1.legend(loc='best')
        ax1.grid(True)
        formatter = mticker.FuncFormatter(lambda x, p: format(x, ',.2f'))
        ax1.yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        plots_drawn +=1
    else:
        print("缺少 'close' 列，无法绘制收盘价图。")

    # 2. 绘制成交量
    if 'volume' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        ax2.bar(df.index, df['volume'], label='成交量 (Volume)', color='green', alpha=0.7)
        ax2.set_title('指数成交量' + plot_title_suffix)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('成交量 (股)')
        ax2.legend(loc='best')
        ax2.grid(True)
        formatter_vol = mticker.FuncFormatter(lambda x, p: format(int(x), ','))
        ax2.yaxis.set_major_formatter(formatter_vol)
        plt.tight_layout()
        plots_drawn +=1
    else:
        print("缺少 'volume' 列，无法绘制成交量图。")
    
    # 3. 绘制涨跌幅 (pctChg)
    if 'pctChg' in df.columns:
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        ax3.plot(df.index, df['pctChg'], label='涨跌幅 (pctChg)', color='red', linestyle='--')
        ax3.set_title('指数日涨跌幅' + plot_title_suffix)
        ax3.set_xlabel('日期')
        ax3.set_ylabel('涨跌幅 (%)')
        ax3.axhline(0, color='black', linewidth=0.5, linestyle='-')
        ax3.legend(loc='best')
        ax3.grid(True)
        formatter_pct = mticker.FuncFormatter(lambda x, p: format(x, ',.2f') + '%')
        ax3.yaxis.set_major_formatter(formatter_pct)
        plt.tight_layout()
        plots_drawn +=1
    else:
        print("缺少 'pctChg' 列，无法绘制涨跌幅图。")


    if plots_drawn > 0:
        plt.show()
    else:
        print("没有足够的数据列来绘制任何指数K线图表。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="绘制指数K线数据的图表。")
    parser.add_argument('--code', type=str, default="sh.000300", help='指数代码 (例如: sh.000300, sz.399001)。')
    parser.add_argument('--freq', type=str, default="d", help='数据频率 (d=日, w=周, m=月)。')
    parser.add_argument('--datadir', type=str, default='./data/index_kline_data', help='存放CSV文件的目录路径。')
    
    args = parser.parse_args()
    
    # 修正从命令行传入的路径，确保与 plot_index_kline_data 函数内 os.path.join 的行为一致
    # 如果 datadir 是相对于脚本位置的，argparse 会正确处理。
    # 如果 datadir 是相对于项目根目录的，需要确保这里的路径拼接正确。
    # 鉴于函数内部使用 '../data/...' 这种模式，这里直接传递 args.datadir 即可。
    
    plot_index_kline_data(index_code=args.code, frequency=args.freq, data_dir=args.datadir)