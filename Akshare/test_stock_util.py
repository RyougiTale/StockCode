import unittest
import pandas as pd
import webbrowser
import os
from Akshare.stock_util import read_history_stock_by_code


class TestStockUtil(unittest.TestCase):
    def test_display_stock_kline_in_browser(self):
        stock_code = "600036"
        print(f"\n--- 单元测试：浏览器显示({stock_code}) K线数据 ---")

        df = read_history_stock_by_code(stock_code)  # 获取去年的数据作为示例
        self.assertFalse(df.empty, f"获取数据失败，无法生成HTML表格。")

        # 准备要显示的数据
        display_df = df.copy()
        display_df = display_df.sort_values("date", ascending=False)
        display_df.reset_index(drop=True, inplace=True)
        display_df.index += 1

        # 生成HTML内容
        html = f"""
        <html>
        <head>
            <title> ({stock_code}) 历史K线</title>
            <style>
                body {{ font-family: sans-serif; }}
                table {{ border-collapse: collapse; width: 95%; margin: 20px auto; font-size: 14px; }}
                th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
                thead th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
                tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tbody tr:hover {{ background-color: #e2e2e2; }}
            </style>
        </head>
        <body>
            <h1 style="text-align:center;">({stock_code}) 历史K线数据</h1>
            {display_df.to_html(classes='table table-striped', index_names=['序号'])}
        </body>
        </html>
        """

        # 将HTML写入临时文件并用浏览器打开
        file_path = os.path.abspath(f"{stock_code}_kline.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"HTML文件已生成: {file_path}")
        print("正在默认浏览器中打开...")
        webbrowser.open(f"file://{file_path}")


if __name__ == "__main__":
    unittest.main()
