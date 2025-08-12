import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import config
from .data_utils import get_all_samples
from .dataset import MarketClassificationDataset, create_soft_label
from .model import MultiEncoderFusionModel

def test_data_pipeline():
    """测试数据管道是否存在NaN问题"""
    print("=== 测试数据管道 ===")
    
    # 1. 测试数据获取
    print("1. 测试数据获取...")
    all_samples, scalers = get_all_samples(config.STOCK_CODES)
    
    if not all_samples:
        print("ERROR: 无法获取样本数据")
        return False
    
    print(f"✓ 成功获取 {len(all_samples)} 个样本")
    
    # 2. 测试数据集创建
    print("2. 测试数据集创建...")
    dataset = MarketClassificationDataset(all_samples)
    print(f"✓ 成功创建数据集，大小: {len(dataset)}")
    
    # 3. 测试几个样本
    print("3. 测试样本数据...")
    test_indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    
    for i, idx in enumerate(test_indices):
        sample = dataset[idx]
        
        # 检查每个组件
        for key, data in sample.items():
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"✗ 样本 {idx} 的 {key} 包含 NaN/Inf")
                return False
        
        print(f"✓ 样本 {i+1} 数据正常")
    
    return True

def test_model_forward():
    """测试模型前向传播"""
    print("\n=== 测试模型前向传播 ===")
    
    # 创建模型
    daily_config = {'feature_size': len(config.FEATURE_COLUMNS['daily']), **config.SHARED_ENCODER_CONFIG}
    weekly_config = {'feature_size': len(config.FEATURE_COLUMNS['weekly']), **config.SHARED_ENCODER_CONFIG}
    monthly_config = {'feature_size': len(config.FEATURE_COLUMNS['monthly']), **config.SHARED_ENCODER_CONFIG}
    
    model = MultiEncoderFusionModel(
        daily_config=daily_config,
        weekly_config=weekly_config,
        monthly_config=monthly_config,
        fusion_dim=config.FUSION_DIM,
        num_classes=config.NUM_CLASSES
    )
    
    # 创建测试数据
    batch_size = 4
    daily_data = torch.randn(batch_size, config.DAILY_SEQ_LEN, len(config.FEATURE_COLUMNS['daily']))
    weekly_data = torch.randn(batch_size, config.WEEKLY_SEQ_LEN, len(config.FEATURE_COLUMNS['weekly']))
    monthly_data = torch.randn(batch_size, config.MONTHLY_SEQ_LEN, len(config.FEATURE_COLUMNS['monthly']))
    
    print("1. 测试正常输入...")
    try:
        outputs = model(daily_data, weekly_data, monthly_data)
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("✗ 模型输出包含 NaN/Inf")
            return False
        
        print(f"✓ 模型输出正常，形状: {outputs.shape}")
        print(f"  输出范围: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
        
    except Exception as e:
        print(f"✗ 模型前向传播失败: {e}")
        return False
    
    # 测试极端输入
    print("2. 测试极端输入...")
    extreme_cases = [
        ("全零输入", torch.zeros_like(daily_data), torch.zeros_like(weekly_data), torch.zeros_like(monthly_data)),
        ("很大的值", daily_data * 100, weekly_data * 100, monthly_data * 100),
        ("很小的值", daily_data * 0.001, weekly_data * 0.001, monthly_data * 0.001),
    ]
    
    for case_name, d_data, w_data, m_data in extreme_cases:
        try:
            outputs = model(d_data, w_data, m_data)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"✗ {case_name}: 输出包含 NaN/Inf")
                return False
            print(f"✓ {case_name}: 输出正常")
        except Exception as e:
            print(f"✗ {case_name}: 失败 - {e}")
            return False
    
    return True

def test_soft_label_generation():
    """测试软标签生成的稳定性"""
    print("\n=== 测试软标签生成 ===")
    
    class_centers = config.SOFT_LABEL_CONFIG["CLASS_CENTERS"]
    temperature = config.SOFT_LABEL_CONFIG["TEMPERATURE"]
    
    # 测试各种回报率值
    test_returns = [
        0.0,      # 正常值
        0.1,      # 正常正值
        -0.1,     # 正常负值
        0.5,      # 较大正值
        -0.5,     # 较大负值
        1.0,      # 极大正值
        -1.0,     # 极大负值
        2.0,      # 非常大的值
        -2.0,     # 非常小的值
    ]
    
    for ret in test_returns:
        try:
            true_return = torch.tensor(ret, dtype=torch.float32)
            soft_label = create_soft_label(true_return, class_centers, temperature)
            
            # 检查软标签
            if torch.isnan(soft_label).any() or torch.isinf(soft_label).any():
                print(f"✗ 回报率 {ret}: 软标签包含 NaN/Inf")
                return False
            
            if soft_label.min() <= 0:
                print(f"✗ 回报率 {ret}: 软标签包含非正值")
                return False
            
            if abs(soft_label.sum().item() - 1.0) > 1e-6:
                print(f"✗ 回报率 {ret}: 软标签和不为1")
                return False
            
            print(f"✓ 回报率 {ret:6.2f}: 软标签正常 (熵: {-(soft_label * torch.log(soft_label)).sum().item():.4f})")
            
        except Exception as e:
            print(f"✗ 回报率 {ret}: 失败 - {e}")
            return False
    
    return True

def main():
    """运行所有测试"""
    print("开始测试修复效果...\n")
    
    tests = [
        ("数据管道", test_data_pipeline),
        ("模型前向传播", test_model_forward),
        ("软标签生成", test_soft_label_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("测试结果总结:")
    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！可以尝试重新训练。")
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")
    
    return all_passed

if __name__ == '__main__':
    main()