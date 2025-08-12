#!/usr/bin/env python3
"""
快速测试修复效果
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from draw_3d_long_term import predict_3d_long_term
import pandas as pd
from scipy.stats import pearsonr

print('测试修复后的预测管道...')

try:
    df = predict_3d_long_term('002415', 'models/enhanced_pretraining/best_loss_top_1.pth', years=1)

    if not df.empty:
        df_valid = df[df['actual_return'].notna()]
        print(f'有效对比数据: {len(df_valid)} 条')
        
        if len(df_valid) > 10:  # 至少需要10个数据点计算相关性
            # 计算相关性
            corr_return, _ = pearsonr(df_valid['predicted_return'], df_valid['actual_return'])
            corr_sharpe, _ = pearsonr(df_valid['predicted_sharpe'], df_valid['actual_sharpe'])
            corr_drawdown, _ = pearsonr(df_valid['predicted_drawdown'], df_valid['actual_drawdown'])
            
            print(f'\n修复后的相关性:')
            print(f'  Return: {corr_return:.4f}')
            print(f'  Sharpe: {corr_sharpe:.4f}')
            print(f'  Drawdown: {corr_drawdown:.4f}')
            
            # 方向准确率
            direction_acc = (
                (df_valid['predicted_return'] > 0) == (df_valid['actual_return'] > 0)
            ).mean()
            print(f'  方向准确率: {direction_acc:.4f}')
            
            # 显示一些样本数据
            print(f'\n样本数据预览:')
            print(df_valid[['date', 'predicted_return', 'actual_return', 'predicted_sharpe', 'actual_sharpe']].head())
            
        else:
            print('有效数据不足，无法计算相关性')
    else:
        print('预测返回空数据')
        
except Exception as e:
    print(f'测试失败: {e}')
    import traceback
    traceback.print_exc()