import torch
import numpy as np

# =============================================================================
# 1. 训练核心函数 (Training Core Function)
# =============================================================================
def train_epoch(model, optimizer, criterion, dataloader, device):
    """
    对模型进行一个epoch的训练。

    Args:
        model: 待训练的模型。
        optimizer: 优化器。
        criterion: 损失函数。
        dataloader: 训练数据加载器。
        device: 计算设备。

    Returns:
        float: 当前epoch的平均损失。
    """
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        # 训练时，解码器的输入是目标序列去掉最后一个时间点
        # e.g., [t1, t2, t3] -> [t1, t2]
        tgt_input = tgt[:, :-1, :]
        
        # 训练的目标是目标序列去掉第一个时间点
        # e.g., [t1, t2, t3] -> [t2, t3]
        tgt_output = tgt[:, 1:, :]
        
        optimizer.zero_grad()
        prediction = model(src, tgt_input, device)
        
        loss = criterion(prediction, tgt_output)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

# =============================================================================
# 2. 推理核心函数 (Inference Core Function)
# =============================================================================
def predict_sequence(model, src_sequence, prediction_steps, device):
    """
    使用自回归方式进行多步预测。

    Args:
        model: 已训练的模型。
        src_sequence (np.array): 输入的源序列, shape [seq_len, num_features]。
        prediction_steps (int): 需要预测的步数。
        device: 计算设备。

    Returns:
        np.array: 预测出的序列, shape [prediction_steps, num_features]。
    """
    model.eval()
    
    # 1. 准备编码器的输入，它在整个预测过程中保持不变
    encoder_input = torch.tensor(src_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. 准备解码器的初始输入，使用源序列的最后一个点作为“种子”
    decoder_input = encoder_input[:, -1:, :] # Shape: [1, 1, num_features]
    
    predicted_sequence = []

    with torch.no_grad():
        for _ in range(prediction_steps):
            # 3. 模型进行一次前向传播
            prediction = model(encoder_input, decoder_input, device)
            
            # 4. 我们只关心对最后一个时间点的预测结果
            next_step_prediction = prediction[:, -1:, :] # Shape: [1, 1, num_features]
            
            # 5. 保存这个预测结果
            predicted_sequence.append(next_step_prediction.cpu().numpy())
            
            # 6. 【关键】自回归：将刚预测的点拼接到解码器输入中，用于下一次预测
            decoder_input = torch.cat([decoder_input, next_step_prediction], dim=1)
            
    # 将预测结果列表拼接成一个完整的numpy数组
    return np.concatenate(predicted_sequence, axis=1).squeeze(0)