import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# 新版 PositionalEncoding (适配 batch_first=True)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 注意，我们不再需要 Dropout 层，因为它应该在主模块中被调用
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe 的形状是 [max_len, d_model]，我们不需要 batch 维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状: [batch_size, seq_len, d_model]
        # self.pe[:x.size(1), :] 的形状是 [seq_len, d_model]
        # PyTorch 的广播机制会自动将其加到每个 batch 上
        x = x + self.pe[:x.size(1), :]
        return x



# (这里先粘贴上面那个新版的 PositionalEncoding)

class SingleTimeframeEncoder(nn.Module):
    """
    一个完整的、遵循最佳实践的单时间框架编码器模块。
    - 使用 [CLS] Token
    - 使用 d_model 缩放
    - 完全兼容 batch_first=True，代码简洁
    """
    def __init__(self, feature_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, max_len=5000):
        super(SingleTimeframeEncoder, self).__init__()
        self.d_model = d_model
        
        # 1. 可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 2. 输入嵌入层和位置编码
        self.input_embedding = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # 3. Dropout 层
        self.dropout = nn.Dropout(p=dropout)
        
        # 4. 标准的 Transformer 编码器层，设置 batch_first=True
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        # src 的形状: [batch_size, seq_len, feature_size]
        batch_size = src.shape[0]

        # 步骤 1: 将输入特征映射到 d_model 维度
        src_embedded = self.input_embedding(src)
        
        # 步骤 2: 将 [CLS] token 拼接到每个序列的最前面
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src_with_cls = torch.cat([cls_tokens, src_embedded], dim=1)
        
        # 步骤 3: * math.sqrt(d_model) 缩放 (最佳实践)
        src_scaled = src_with_cls * math.sqrt(self.d_model)
        
        # 步骤 4: 添加位置编码 (新版PositionalEncoding直接处理batch_first)
        src_pos_encoded = self.pos_encoder(src_scaled)
        
        # 步骤 5: 应用 Dropout (最佳实践)
        src_final = self.dropout(src_pos_encoded)
        
        # 步骤 6: 送入 Transformer 编码器
        memory = self.transformer_encoder(src_final)

        # 步骤 7: 提取 [CLS] token 的输出作为上下文向量
        context_vector = memory[:, 0, :]
        
        return context_vector


# 组件3：完整的多编码器融合模型
class MultiEncoderFusionModel(nn.Module):
    def __init__(self, daily_config, weekly_config, monthly_config, 
                 fusion_dim, num_classes, dropout=0.1):
        super(MultiEncoderFusionModel, self).__init__()
        
        # 1. 创建三个独立的“专家”编码器
        self.daily_encoder = SingleTimeframeEncoder(**daily_config)
        self.weekly_encoder = SingleTimeframeEncoder(**weekly_config)
        self.monthly_encoder = SingleTimeframeEncoder(**monthly_config)
        
        # 每个专家的输出维度都是 d_model
        # 我们将它们拼接起来
        concatenated_dim = daily_config['d_model'] + weekly_config['d_model'] + monthly_config['d_model']
        
        # 2. 融合层 (Fusion Layer)
        # 一个简单的MLP，用于学习如何融合三个专家的意见
        self.fusion_layer = nn.Sequential(
            nn.Linear(concatenated_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. 决策头 (Classification Head)
        # 最终的分类器，输出每个类别的 Logits
        self.classification_head = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_daily, x_weekly, x_monthly):
        # x_daily: [batch_size, daily_seq_len, daily_feature_size]
        # x_weekly: [batch_size, weekly_seq_len, weekly_feature_size]
        # x_monthly: [batch_size, monthly_seq_len, monthly_feature_size]
        
        # 1. 并行地从每个专家获取上下文向量
        daily_context = self.daily_encoder(x_daily)   # Shape: [batch_size, d_model_daily]
        weekly_context = self.weekly_encoder(x_weekly) # Shape: [batch_size, d_model_weekly]
        monthly_context = self.monthly_encoder(x_monthly) # Shape: [batch_size, d_model_monthly]
        
        # 2. 将三个专家的意见（上下文向量）拼接起来
        concatenated_context = torch.cat([daily_context, weekly_context, monthly_context], dim=1)
        
        # 3. 通过融合层进行信息融合
        fused_representation = self.fusion_layer(concatenated_context)
        
        # 4. 通过决策头得出最终预测
        logits = self.classification_head(fused_representation)
        
        # 为了数值稳定性，先检查logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaN or Inf detected in logits before log_softmax")
            print(f"Logits: {logits}")
            # 用零替换异常值
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                               torch.zeros_like(logits), logits)
        
        # 限制logits的范围以避免数值溢出
        logits = torch.clamp(logits, min=-50, max=50)
        
        # KLDivLoss 要求输入是 log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 最终检查
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print("WARNING: NaN or Inf detected in log_probs after log_softmax")
            print(f"Original logits: {logits}")
            print(f"Log probs: {log_probs}")
            # 返回均匀分布的log概率
            uniform_log_probs = torch.full_like(log_probs, -math.log(log_probs.size(-1)))
            return uniform_log_probs
        
        return log_probs