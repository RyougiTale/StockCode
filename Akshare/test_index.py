import unittest
import pandas as pd
import webbrowser
import os
from Akshare.index import read_hs300, read_sz50

class TestIndexComparison(unittest.TestCase):
    def test_hs300_sz50_overlap(self):
        """
        测试沪深300和上证50成分股的重合与非重合部分,并输出代码和名称。
        """
        hs300_df = read_hs300()
        sz50_df = read_sz50()

        self.assertIsNotNone(hs300_df, "获取沪深300数据失败")
        self.assertIsNotNone(sz50_df, "获取上证50数据失败")

        hs300_codes = set(hs300_df['stock_code'])
        sz50_codes = set(sz50_df['stock_code'])
        
        overlap_codes = hs300_codes.intersection(sz50_codes)
        hs300_only_codes = hs300_codes.difference(sz50_codes)
        sz50_only_codes = sz50_codes.difference(hs300_codes)

        # Helper function to print stock info
        def print_stock_info(title, df, codes):
            print(f"\n--- {title} ({len(codes)}只) ---")
            if not codes:
                print("无")
                return
            
            filtered_df = df[df['stock_code'].isin(codes)].copy()
            filtered_df = filtered_df.sort_values('stock_code')
            for _, row in filtered_df.iterrows():
                print(f"{row['stock_code']} {row['stock_name']}")

        print(f"\n{'='*20} 沪深300 vs 上证50 成分股对比 {'='*20}")
        print(f"沪深300 成分股数量: {len(hs300_codes)}")
        print(f"上证50 成分股数量: {len(sz50_codes)}")
        
        # 提取重合部分的股票信息
        overlap_df = hs300_df[hs300_df['stock_code'].isin(overlap_codes)]
        print_stock_info("重合部分", overlap_df, overlap_codes)

        # 提取沪深300独有部分的股票信息
        hs300_only_df = hs300_df[hs300_df['stock_code'].isin(hs300_only_codes)]
        print_stock_info("仅在沪深300", hs300_only_df, hs300_only_codes)

        # 提取上证50独有部分的股票信息
        sz50_only_df = sz50_df[sz50_df['stock_code'].isin(sz50_only_codes)]
        print_stock_info("仅在上证50", sz50_only_df, sz50_only_codes)
        
        print(f"\n{'='*25} 测试结束 {'='*25}")

    def test_find_duplicates_in_hs300_source(self):
        """
        直接从akshare源数据中检查沪深300成分股是否存在重复项。
        """
        import akshare as ak
        print("\n--- 单元测试：检查沪深300源数据重复项 ---")
        
        # 直接调用akshare获取原始数据
        try:
            raw_df = ak.index_stock_cons(symbol="000300")
            self.assertIsNotNone(raw_df, "从akshare获取原始沪深300数据失败")
        except Exception as e:
            self.fail(f"调用akshare接口失败: {e}")

        # 找出基于'品种代码'的重复项
        duplicates = raw_df[raw_df.duplicated(subset=['品种代码'], keep=False)]
        
        if duplicates.empty:
            print("未在源数据中发现重复项。")
        else:
            print("在源数据中发现以下重复股票代码：")
            # 按代码排序，方便查看
            sorted_duplicates = duplicates.sort_values('品种代码')
            print(sorted_duplicates)
        
        # 这个断言的目的是，如果有一天数据源修复了，这个测试会失败，提醒我们
        self.assertFalse(duplicates.empty, "未在沪深300源数据中发现预期的重复项，数据源可能已修复。")
        print("--- 测试结束 ---")


    def test_display_hs300_in_browser(self):
        """
        在浏览器中以可滚动的HTML表格形式显示沪深300成分股。
        """
        print("\n--- 单元测试：浏览器显示沪深300成分股 ---")
        
        df = read_hs300()
        self.assertFalse(df.empty, "获取沪深300数据失败，无法生成HTML表格。")

        # 准备要显示的数据
        display_df = df[['stock_code', 'stock_name', 'update_date']].copy()
        display_df.reset_index(drop=True, inplace=True)
        display_df.index += 1 # 让索引从1开始

        # 生成HTML内容，并添加CSS样式
        html = f"""
        <html>
        <head>
            <title>沪深300 成分股</title>
            <style>
                body {{ font-family: sans-serif; }}
                table {{ border-collapse: collapse; width: 80%; margin: 20px auto; }}
                th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
                thead th {{ background-color: #f2f2f2; }}
                tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tbody tr:hover {{ background-color: #e2e2e2; }}
            </style>
        </head>
        <body>
            <h1 style="text-align:center;">沪深300 成分股</h1>
            {display_df.to_html(classes='table table-striped', index_names=['序号'])}
        </body>
        </html>
        """

        # 将HTML写入临时文件并用浏览器打开
        file_path = os.path.abspath('hs300_constituents.html')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML文件已生成: {file_path}")
        print("正在默认浏览器中打开... (测试将继续执行，无需等待浏览器关闭)")
        webbrowser.open(f'file://{file_path}')


if __name__ == '__main__':
    unittest.main()