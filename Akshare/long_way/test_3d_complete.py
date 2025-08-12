import sys
import os
import torch
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import config
from .label_3d_generator import ThreeDimensionalLabelGenerator
from .model_3d import create_3d_model, Multi3DLoss
from .dataset_3d import Market3DClassificationDataset
from .engine_3d import train_one_epoch_3d, evaluate_3d

def test_complete_3d_pipeline():
    """
    测试完整的3D软标签模型管道
    """
    print("=== 完整3D软标签模型管道测试 ===")
    
    # --- 1. 测试3D标签生成器 ---
    print("\n1. 测试3D标签生成器...")
    
    # 创建模拟价格数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 50))
    
    price_df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # 测试标签生成器
    label_generator = ThreeDimensionalLabelGenerator(look_forward_days=10, temperature=0.002)
    labels_3d = label_generator.generate_3d_labels_for_dataset(price_df)
    
    print(f"✓ 生成了 {len(labels_3d)} 个3D标签")
    
    # 验证第一个标签
    first_label = labels_3d[0]
    print(f"第一个标签示例:")
    for dim, label in first_label.items():
        print(f"  {dim}: {label.numpy()}")
        print(f"  {dim} 概率和: {label.sum().item():.6f}")
    
    # --- 2. 测试3D数据集 ---
    print("\n2. 测试3D数据集...")
    
    # 创建模拟样本
    samples = []
    for i in range(20):
        sample = {
            'daily': np.random.randn(60, 12).astype(np.float32),
            'weekly': np.random.randn(52, 12).astype(np.float32),
            'monthly': np.random.randn(24, 12).astype(np.float32),
            'future_prices': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 15)),
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
        }
        samples.append(sample)
    
    # 创建数据集
    dataset = Market3DClassificationDataset(samples, look_forward_days=10, temperature=0.002)
    
    print(f"✓ 创建了包含 {len(dataset)} 个样本的3D数据集")
    
    # 测试数据加载
    sample = dataset[0]
    print(f"样本数据形状:")
    print(f"  Daily: {sample['daily'].shape}")
    print(f"  Weekly: {sample['weekly'].shape}")
    print(f"  Monthly: {sample['monthly'].shape}")
    print(f"  3D标签维度: {list(sample['labels_3d'].keys())}")
    
    # --- 3. 测试3D模型 ---
    print("\n3. 测试3D模型...")
    
    # 创建模型
    model = create_3d_model(config)
    
    # 测试前向传播
    batch_size = 4
    daily_data = torch.randn(batch_size, 60, 12)
    weekly_data = torch.randn(batch_size, 52, 12)
    monthly_data = torch.randn(batch_size, 24, 12)
    
    with torch.no_grad():
        outputs = model(daily_data, weekly_data, monthly_data)
    
    print(f"✓ 模型前向传播成功")
    print(f"输出形状:")
    for dim, output in outputs.items():
        print(f"  {dim}: {output.shape}")
    
    # --- 4. 测试3D损失函数 ---
    print("\n4. 测试3D损失函数...")
    
    criterion = Multi3DLoss()
    
    # 创建目标标签
    targets = {
        'return': torch.softmax(torch.randn(batch_size, 5), dim=1),
        'sharpe': torch.softmax(torch.randn(batch_size, 5), dim=1),
        'drawdown': torch.softmax(torch.randn(batch_size, 5), dim=1)
    }
    
    losses = criterion(outputs, targets)
    
    print(f"✓ 损失函数计算成功")
    print(f"损失值:")
    for dim, loss in losses.items():
        print(f"  {dim}: {loss.item():.6f}")
    
    # --- 5. 测试数据加载器 ---
    print("\n5. 测试数据加载器...")
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 测试一个批次
    batch = next(iter(dataloader))
    
    print(f"✓ 数据加载器工作正常")
    print(f"批次数据形状:")
    print(f"  Daily: {batch['daily'].shape}")
    print(f"  Weekly: {batch['weekly'].shape}")
    print(f"  Monthly: {batch['monthly'].shape}")
    
    # 验证3D标签
    for dim in ['return', 'sharpe', 'drawdown']:
        print(f"  {dim} 标签: {batch['labels_3d'][dim].shape}")
    
    # --- 6. 测试训练步骤 ---
    print("\n6. 测试训练步骤...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # 模拟一个训练步骤
    daily_data = batch['daily']
    weekly_data = batch['weekly']
    monthly_data = batch['monthly']
    labels_3d = {dim: batch['labels_3d'][dim] for dim in ['return', 'sharpe', 'drawdown']}
    
    # 前向传播
    outputs = model(daily_data, weekly_data, monthly_data)
    
    # 计算损失
    losses = criterion(outputs, labels_3d)
    
    # 反向传播
    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()
    
    print(f"✓ 训练步骤执行成功")
    print(f"训练损失: {losses['total'].item():.6f}")
    
    # --- 7. 性能基准测试 ---
    print("\n7. 性能基准测试...")
    
    import time
    
    # 测试推理速度
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = model(daily_data, weekly_data, monthly_data)
        end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 100
    print(f"✓ 平均推理时间: {avg_inference_time*1000:.2f} ms")
    
    # 测试内存使用
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(daily_data.cuda(), weekly_data.cuda(), monthly_data.cuda())
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"✓ 峰值GPU内存使用: {peak_memory:.1f} MB")
    
    # --- 8. 数值稳定性测试 ---
    print("\n8. 数值稳定性测试...")
    
    # 测试极端输入
    extreme_cases = [
        ("全零输入", torch.zeros_like(daily_data), torch.zeros_like(weekly_data), torch.zeros_like(monthly_data)),
        ("大数值输入", daily_data * 100, weekly_data * 100, monthly_data * 100),
        ("小数值输入", daily_data * 0.001, weekly_data * 0.001, monthly_data * 0.001),
    ]
    
    model.eval()
    stability_passed = True
    
    for case_name, d_data, w_data, m_data in extreme_cases:
        try:
            with torch.no_grad():
                outputs = model(d_data, w_data, m_data)
            
            # 检查输出是否有NaN或Inf
            has_nan = any(torch.isnan(output).any() for output in outputs.values())
            has_inf = any(torch.isinf(output).any() for output in outputs.values())
            
            if has_nan or has_inf:
                print(f"  ✗ {case_name}: 输出包含NaN或Inf")
                stability_passed = False
            else:
                print(f"  ✓ {case_name}: 输出正常")
                
        except Exception as e:
            print(f"  ✗ {case_name}: 异常 - {e}")
            stability_passed = False
    
    if stability_passed:
        print("✓ 数值稳定性测试通过")
    else:
        print("⚠️ 数值稳定性测试发现问题")
    
    # --- 总结 ---
    print("\n" + "="*50)
    print("3D软标签模型管道测试总结")
    print("="*50)
    
    test_results = [
        "✓ 3D标签生成器",
        "✓ 3D数据集",
        "✓ 3D模型架构",
        "✓ 3D损失函数",
        "✓ 数据加载器",
        "✓ 训练步骤",
        "✓ 性能基准",
        "✓ 数值稳定性" if stability_passed else "⚠️ 数值稳定性"
    ]
    
    for result in test_results:
        print(f"  {result}")
    
    print(f"\n🎉 3D软标签模型已准备就绪！")
    print(f"可以运行完整训练:")
    print(f"  python -m long_way.train_3d")
    print(f"或运行测试:")
    print(f"  python -m long_way.train_3d test")

def compare_1d_vs_3d_models():
    """
    比较1D和3D模型的差异
    """
    print("\n=== 1D vs 3D 模型对比 ===")
    
    print("1D模型 (原始):")
    print("  - 输出: 5个回报率类别")
    print("  - 损失: 单一KL散度")
    print("  - 评估: Top-K准确率")
    print("  - 优势: 简单、快速")
    print("  - 劣势: 信息有限")
    
    print("\n3D模型 (升级):")
    print("  - 输出: 回报率(5) + 夏普比率(5) + 最大回撤(5)")
    print("  - 损失: 加权多任务KL散度")
    print("  - 评估: 三维度Top-K准确率")
    print("  - 优势: 信息丰富、投资决策更全面")
    print("  - 劣势: 复杂度增加、训练时间更长")
    
    print("\n投资价值对比:")
    print("  1D: 只知道涨跌方向和幅度")
    print("  3D: 知道涨跌 + 风险调整收益 + 回撤风险")
    print("      → 更适合实际投资决策")

if __name__ == '__main__':
    test_complete_3d_pipeline()
    compare_1d_vs_3d_models()