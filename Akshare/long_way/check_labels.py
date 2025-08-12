import torch
import random
import sys
import os
import numpy as np

# --- 路径设置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 导入 long_way 模块 ---
from . import config
from .data_utils import get_all_samples
from .dataset_3d import Market3DClassificationDataset

def check_random_samples(num_samples=10):
    """
    加载数据集，随机抽取几个样本，并打印它们的软标签分布。
    """
    print("--- Loading data to check soft labels ---")
    # 注意：这里我们只加载数据，不关心scaler的返回
    all_samples, _ = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        print("No samples found. Exiting.")
        return

    # 使用3D数据集，从config获取参数
    full_dataset = Market3DClassificationDataset(
        all_samples,
        look_forward_days=config.SOFT_LABEL_CONFIG["LOOK_FORWARD_DAYS"],
        temperature=config.SOFT_LABEL_CONFIG["TEMPERATURE"],
        use_relative_metrics=True
    )
    print(f"Total samples available: {len(full_dataset)}")
    print(f"Temperature setting in config: {config.SOFT_LABEL_CONFIG['TEMPERATURE']}")
    print("-" * 40)

    # 随机抽取样本
    indices_to_check = random.sample(range(len(full_dataset)), num_samples)

    for i, idx in enumerate(indices_to_check):
        sample = full_dataset[idx]
        # 3D系统中获取原始指标
        raw_metrics = full_dataset.raw_metrics[idx]
        soft_labels_3d = sample['labels_3d']

        print(f"\n--- Sample #{i+1} (Index: {idx}) ---")
        print(f"Raw Metrics:")
        print(f"  Total Return: {raw_metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {raw_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {raw_metrics['max_drawdown']:.2%}")
        
        # 打印3D软标签
        for metric_name in ['return', 'sharpe', 'drawdown']:
            probs = soft_labels_3d[metric_name].numpy()
            print(f"\n{metric_name.title()} Label Distribution:")
            
            # 获取当前样本对应的股票代码（用于获取自适应中心点）
            stock_code = full_dataset.samples[idx].get('stock_code', 'UNKNOWN')
            
            # 获取该指标的自适应中心点（绝对值）
            if metric_name == 'return':
                centers = full_dataset.label_generator.relative_calculator.get_adaptive_centers(stock_code, 'total_return')
                center_labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']
            elif metric_name == 'sharpe':
                centers = full_dataset.label_generator.relative_calculator.get_adaptive_centers(stock_code, 'sharpe_ratio')  
                center_labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']
            else:  # drawdown
                centers = full_dataset.label_generator.relative_calculator.get_adaptive_centers(stock_code, 'max_drawdown')
                center_labels = ['Terrible', 'Bad', 'Average', 'Good', 'Excellent']
            
            # 找到最高概率的索引
            peak_idx = np.argmax(probs)
            
            for j, prob in enumerate(probs):
                marker = " <-- Peak" if j == peak_idx else ""
                if metric_name == 'return':
                    print(f"  Class {j} ({center_labels[j]}, ≈{centers[j]:+.1%}): Probability = {prob:.4f}{marker}")
                elif metric_name == 'sharpe':
                    print(f"  Class {j} ({center_labels[j]}, ≈{centers[j]:+.2f}): Probability = {prob:.4f}{marker}")
                else:  # drawdown
                    print(f"  Class {j} ({center_labels[j]}, ≈{centers[j]:.1%}): Probability = {prob:.4f}{marker}")
    
    print("\n" + "-" * 40)
    print("Check complete.")


if __name__ == '__main__':
    check_random_samples()