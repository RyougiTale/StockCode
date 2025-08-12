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
from .dataset import MarketClassificationDataset

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

    full_dataset = MarketClassificationDataset(all_samples)
    print(f"Total samples available: {len(full_dataset)}")
    print(f"Temperature setting in config: {config.SOFT_LABEL_CONFIG['TEMPERATURE']}")
    print("-" * 40)

    # 随机抽取样本
    indices_to_check = random.sample(range(len(full_dataset)), num_samples)

    for i, idx in enumerate(indices_to_check):
        sample = full_dataset[idx]
        true_return = full_dataset.samples[idx]['label'] # 获取原始回报率
        soft_label = sample['label']

        print(f"\n--- Sample #{i+1} (Index: {idx}) ---")
        print(f"True Future Return: {true_return * 100:.2f}%")
        
        print("Generated Soft Label Distribution:")
        centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"].numpy()
        probs = soft_label.numpy()
        
        for j, center in enumerate(centers):
            # 找到最接近的中心点并标记
            is_peak = np.argmin(np.abs(centers - true_return)) == j
            marker = "<-- Peak" if is_peak else ""
            print(f"  - Center {center*100: >5.1f}%: Probability = {probs[j]:.4f} {marker}")
    
    print("\n" + "-" * 40)
    print("Check complete.")


if __name__ == '__main__':
    check_random_samples()