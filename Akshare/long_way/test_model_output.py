#!/usr/bin/env python3
"""
测试模型输出格式的简单脚本
"""
import torch
import numpy as np
import os
import sys

from . import config
from .model_3d import create_3d_model

def test_model_output():
    """测试模型输出格式"""
    print("=== 测试模型输出格式 ===")
    
    # 加载模型
    model_path = os.path.join(config.MODEL_DIR, "best_loss_top_1.pth")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    model = create_3d_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    # 创建测试数据
    batch_size = 2
    daily_data = torch.randn(batch_size, config.DAILY_SEQ_LEN, len(config.FEATURE_COLUMNS['daily'])).to(config.DEVICE)
    weekly_data = torch.randn(batch_size, config.WEEKLY_SEQ_LEN, len(config.FEATURE_COLUMNS['weekly'])).to(config.DEVICE)
    monthly_data = torch.randn(batch_size, config.MONTHLY_SEQ_LEN, len(config.FEATURE_COLUMNS['monthly'])).to(config.DEVICE)
    
    print(f"输入形状:")
    print(f"  daily: {daily_data.shape}")
    print(f"  weekly: {weekly_data.shape}")
    print(f"  monthly: {monthly_data.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(daily_data, weekly_data, monthly_data)
    
    print(f"\n原始模型输出:")
    for key, value in outputs.items():
        print(f"  {key}: shape={value.shape}, sample={value[0].cpu().numpy()}")
    
    print(f"\n使用 exp() 得到概率:")
    for key, value in outputs.items():
        probs = torch.exp(value)
        print(f"  {key}: sum={probs[0].sum():.6f}, probs={probs[0].cpu().numpy()}")
    
    print(f"\n错误的 softmax() 处理:")
    for key, value in outputs.items():
        wrong_probs = torch.softmax(value, dim=1)
        print(f"  {key}: sum={wrong_probs[0].sum():.6f}, probs={wrong_probs[0].cpu().numpy()}")

if __name__ == "__main__":
    test_model_output()