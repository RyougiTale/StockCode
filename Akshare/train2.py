import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from stock_util import read_history_stock_by_code


# =============================================================================
# 1. 数据处理核心函数 (保留您的原始逻辑)
# =============================================================================
MAX_SEQ_LEN = 60

def prepare_data(stock_code):
    print(f"--- Loading, Processing and Preparing Data ---")
    full_df = read_history_stock_by_code(stock_code)
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
    full_df['SMA20'] = full_df['close'].rolling(window=20).mean()
    full_df['SMA60'] = full_df['close'].rolling(window=60).mean()
    feature_columns.extend(['SMA20', 'SMA60'])
    
    clean_df = full_df[feature_columns].dropna().reset_index(drop=True)
    if len(clean_df) < MAX_SEQ_LEN * 2:
        print("Not enough data after cleaning."); return None, None, None, None, None

    train_df = clean_df[:-MAX_SEQ_LEN]
    inference_df = clean_df.tail(MAX_SEQ_LEN).reset_index(drop=True)

    scalers = {}
    train_scaled_df = pd.DataFrame()
    for col in feature_columns:
        scaler = MinMaxScaler()
        train_scaled_df[col] = scaler.fit_transform(train_df[[col]]).flatten()
        scalers[col] = scaler
    
    # 使用训练集的scaler来转换推理数据
    inference_scaled_df = pd.DataFrame()
    for col in feature_columns:
        inference_scaled_df[col] = scalers[col].transform(inference_df[[col]]).flatten()

    train_data = train_scaled_df.values
    inference_input_data = inference_scaled_df.values
    
    print("--- Data Ready ---")
    return train_data, inference_input_data, scalers, feature_columns, inference_df

# =============================================================================
# 2. 数据集类 (保留您的原始逻辑)
# =============================================================================
class StockDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences) - MAX_SEQ_LEN
    def __getitem__(self, idx):
        input_seq = self.sequences[idx : idx + MAX_SEQ_LEN]
        target_seq = self.sequences[idx + 1 : idx + MAX_SEQ_LEN + 1]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)

# =============================================================================
# 3. 模型构建 (保留您的原始逻辑)
# =============================================================================
class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class StockSeq2SeqTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(StockSeq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.encoder_input_projection = nn.Linear(num_features, d_model)
        self.decoder_input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.fc_out = nn.Linear(d_model, num_features)
    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src, tgt, device):
        src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0), device)
        src_emb = self.pos_encoder(self.encoder_input_projection(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.decoder_input_projection(tgt) * math.sqrt(self.d_model))
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(output.transpose(0, 1))

# =============================================================================
# 4. 训练和推理 (Training and Inference)
# =============================================================================
def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        # 训练时，解码器的输入是目标序列去掉最后一个点
        tgt_input = tgt[:, :-1, :]
        # 训练的目标是目标序列去掉第一个点
        tgt_output = tgt[:, 1:, :]
        optimizer.zero_grad()
        prediction = model(src, tgt_input, device)
        loss = criterion(prediction, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# =============================================================================
# 5. 推理函数 (Inference Function) - 已修正
# =============================================================================
def predict_sequence(model, src_sequence, prediction_steps, device):
    """
    使用自回归方式进行序列预测。
    """
    model.eval()
    
    # 1. 准备编码器的输入，它在整个预测过程中保持不变
    encoder_input = torch.tensor(src_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. 准备解码器的初始输入。我们用源序列的最后一个时间点作为“种子”来启动生成过程。
    # 这相当于告诉解码器：“基于历史，从这里开始预测”。
    decoder_input = encoder_input[:, -1:, :] # Shape: [1, 1, num_features]
    
    predicted_sequence = []

    with torch.no_grad():
        for _ in range(prediction_steps):
            # 3. 模型进行一次前向传播
            # 编码器看到的是完整的历史(encoder_input)
            # 解码器看到的是到目前为止已经生成的部分(decoder_input)
            prediction = model(encoder_input, decoder_input, device)
            print(prediction.shape)
            print(prediction)
            
            # 4. 我们只关心解码器对最后一个时间点的预测结果
            # 这就是我们对下一个时间点的预测值
            next_step_prediction = prediction[:, -1:, :] # Shape: [1, 1, num_features]
            
            # 5. 保存这个预测结果
            predicted_sequence.append(next_step_prediction.cpu().numpy())
            
            # 6. 【关键】自回归步骤：
            # 将刚刚的预测结果拼接到解码器的输入序列中，
            # 作为下一次循环的输入。
            decoder_input = torch.cat([decoder_input, next_step_prediction], dim=1)
            
    # 将预测结果列表拼接成一个完整的numpy数组
    return np.concatenate(predicted_sequence, axis=1).squeeze(0)

# =============================================================================
# 6. 主执行函数 (Main Execution)
# =============================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT = 128, 8, 3, 3, 512, 0.1
    EPOCHS, BATCH_SIZE, LEARNING_RATE = 20, 32, 0.0001 # 增加了Epoch以便更好地学习
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据准备 ---
    train_data, inference_input_data, scalers, feature_columns, inference_df = prepare_data("600036")
    if train_data is None: exit()
    
    train_dataset = StockDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 2. 模型初始化 ---
    model = StockSeq2SeqTransformer(num_features=len(feature_columns), d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("--- Model Initialized ---")

    # --- 3. 训练模型 ---
    print("\n--- Training Started ---")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device)
        print(f"Epoch {epoch+1:02}/{EPOCHS} | Train Loss: {train_loss:.6f}")
    print("--- Training Finished ---\n")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "stock_transformer_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"--- Model Saved to {model_path} ---\n")


    # --- 4. 执行推理 ---
    print("--- Performing Inference ---")
    # 使用最后MAX_SEQ_LEN天的数据作为输入，预测未来5天
    predicted_scaled = predict_sequence(model, inference_input_data, prediction_steps=5, device=device)
    
    # --- 5. 反归一化并展示结果 ---
    predicted_df = pd.DataFrame(columns=feature_columns)
    for i, col in enumerate(feature_columns):
        # 注意：这里需要重塑一下形状以匹配scaler的期望输入
        predicted_df[col] = scalers[col].inverse_transform(predicted_scaled[:, i].reshape(-1, 1)).flatten()
    
    print("\n--- Prediction Results ---")
    print("Input Sequence Head (Last 5 days of clean data):")
    print(inference_df.tail())
    print("\nPredicted Sequence (Next 5 days):")
    # 为了美观，我们只展示部分关键列
    # print(predicted_df[['open', 'high', 'low', 'close', 'SMA20', 'SMA60']])
    print(predicted_df)
