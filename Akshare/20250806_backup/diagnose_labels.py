import torch
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 路径设置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 导入 long_way 模块 ---
from . import config
from .data_utils import get_all_samples
from .dataset import MarketClassificationDataset, create_soft_label

def diagnose_random_samples(num_samples=5):
    """
    加载数据集，随机抽取样本，并可视化其未来走势与生成的软标签。
    """
    print("--- Loading data for label diagnosis ---")
    all_samples, _ = get_all_samples(config.STOCK_CODES)
    if not all_samples:
        print("No samples found. Exiting.")
        return

    full_dataset = MarketClassificationDataset(all_samples)
    print(f"Total samples available: {len(full_dataset)}")
    print(f"Temperature setting in config: {config.SOFT_LABEL_CONFIG['TEMPERATURE']}")
    print("-" * 50)

    # 随机抽取样本
    indices_to_check = random.sample(range(len(full_dataset)), num_samples)

    # 创建子图
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, num_samples * 3))
    fig.suptitle("Label Diagnosis: Future Trend vs. Soft Label", fontsize=16)

    for i, idx in enumerate(indices_to_check):
        sample_data = full_dataset.samples[idx]
        
        # 获取数据
        future_prices = sample_data['future_prices']
        true_return = sample_data['label']
        
        # 生成软标签
        soft_label = create_soft_label(
            torch.tensor(true_return, dtype=torch.float32),
            config.SOFT_LABEL_CONFIG["CLASS_CENTERS"],
            config.SOFT_LABEL_CONFIG["TEMPERATURE"]
        ).numpy()

        # 归一化价格以便于可视化
        normalized_prices = future_prices / future_prices[0]

        # 绘图
        ax = axes[i]
        ax.plot(normalized_prices, marker='o', linestyle='-', markersize=4)
        ax.axhline(1.0, color='grey', linestyle='--', linewidth=1) # 起始水平线
        
        # 构建标题
        centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"].numpy()
        soft_label_str = ", ".join([f"{p:.2f}" for p in soft_label])
        peak_idx = np.argmax(soft_label)
        title_str = (
            f"Sample #{idx} | True Return: {true_return*100:.2f}% | Peak Center: {centers[peak_idx]*100:.1f}%\n"
            f"Soft Label: [{soft_label_str}]"
        )
        ax.set_title(title_str, fontsize=10)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图像
    output_path = os.path.join(config.MODEL_DIR, "label_diagnosis.png")
    plt.savefig(output_path)
    print(f"\nDiagnosis plot saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    diagnose_random_samples()