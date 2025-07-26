import torch
import torch.nn as nn
import math

# =============================================================================
# 1. 位置编码 (Positional Encoding)
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    为序列数据注入位置信息。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# =============================================================================
# 2. Transformer 模型 (Seq2Seq)
# =============================================================================
class StockSeq2SeqTransformer(nn.Module):
    """
    用于股票价格预测的序列到序列Transformer模型。
    """
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(StockSeq2SeqTransformer, self).__init__()
        self.d_model = d_model
        
        # 输入层：将特征维度映射到d_model
        self.encoder_input_projection = nn.Linear(num_features, d_model)
        self.decoder_input_projection = nn.Linear(num_features, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # PyTorch原生的Transformer模块
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=False # 注意：PyTorch Transformer默认期望 (seq_len, batch, feature)
        )
        
        # 输出层：将d_model映射回特征维度
        self.fc_out = nn.Linear(d_model, num_features)

    def _generate_square_subsequent_mask(self, sz, device):
        """
        为解码器生成一个上三角的掩码，防止其在预测时看到未来的信息。
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, device):
        """
        前向传播。
        Args:
            src: 源序列, shape [batch_size, src_seq_len, num_features]
            tgt: 目标序列, shape [batch_size, tgt_seq_len, num_features]
            device: 计算设备
        """
        # PyTorch Transformer期望的输入形状是 (seq_len, batch_size, feature_dim)
        src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
        
        # 生成解码器掩码
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0), device)
        
        # 1. 输入投影和位置编码
        src_emb = self.pos_encoder(self.encoder_input_projection(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.decoder_input_projection(tgt) * math.sqrt(self.d_model))
        
        # 2. Transformer处理
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        
        # 3. 输出投影
        # 转换回 (batch_size, seq_len, feature_dim)
        return self.fc_out(output.transpose(0, 1))