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
# 0. 模拟数据生成 (为了代码可独立运行)
# =============================================================================
def create_dummy_stock_data(filepath="stock_data.csv"):
    if os.path.exists(filepath): return
    print(f"Creating dummy stock data at '{filepath}'...")
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=500))
    data = {'date': dates, 'open': np.random.uniform(9.5, 10.5, 500).cumsum() + 100}
    df = pd.DataFrame(data)
    df['high'] = df['open'] + np.random.uniform(0, 1, 500)
    df['low'] = df['open'] - np.random.uniform(0, 1, 500)
    df['close'] = (df['open'] + df['high'] + df['low']) / 3 + np.random.uniform(-0.2, 0.2, 500)
    # 确保 high >= low
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    df['volume'] = np.random.randint(100000, 500000, 500)
    df['turnover'] = df['volume'] * df['close']
    df['amplitude'] = (df['high'] - df['low']) / df['close'] * 100
    df['pct_chg'] = df['close'].pct_change() * 100
    df['chg_amount'] = df['close'].diff()
    df['turnover_rate'] = df['volume'] / 1e9
    df.to_csv(filepath, index=False)
    print("Dummy data created.")

# =============================================================================
# 1. 数据处理核心函数 (保留您的原始逻辑)
# =============================================================================
MAX_SEQ_LEN = 60

def prepare_data(stock_code):
    full_df = read_history_stock_by_code(stock_code)

    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate', 'SMA20', 'SMA60']
    full_df['SMA20'] = full_df['close'].rolling(window=20).mean()
    full_df['SMA60'] = full_df['close'].rolling(window=60).mean()
    # feature_columns.extend(['SMA20', 'SMA60'])
    
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
    
    inference_scaled_df = pd.DataFrame()
    for col in feature_columns:
        inference_scaled_df[col] = scalers[col].transform(inference_df[[col]]).flatten()

    train_data = train_scaled_df.values
    inference_input_data = inference_scaled_df.values
    
    print("--- Data Ready ---")
    return train_data, inference_input_data, scalers, feature_columns, inference_df

# =============================================================================
# 2. 数据集类 (恢复为预测所有特征)
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
# 3. 模型构建 (恢复为预测所有特征)
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
# 4. 自定义损失函数 (Custom Loss Function) - 核心改动
# =============================================================================
class CustomFinancialLoss(nn.Module):
    def __init__(self, feature_columns, lambda_penalty=1.0):
        super(CustomFinancialLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_penalty = lambda_penalty
        
        # 获取关键特征的索引，以便在计算中定位它们
        self.high_idx = feature_columns.index('high')
        self.low_idx = feature_columns.index('low')
        self.close_idx = feature_columns.index('close')
        self.amplitude_idx = feature_columns.index('amplitude')

    def forward(self, prediction, target):
        # 1. 计算标准的MSE损失，这是基础
        mse = self.mse_loss(prediction, target)
        
        # 2. 计算规则惩罚项
        penalty = 0.0
        
        # 提取预测值中的关键列
        pred_high = prediction[..., self.high_idx]
        pred_low = prediction[..., self.low_idx]
        pred_close = prediction[..., self.close_idx]
        pred_amplitude = prediction[..., self.amplitude_idx]
        
        # 规则1: high 必须 >= low。如果 low > high，则产生惩罚。
        # torch.relu(x) 当 x > 0 时等于 x，否则等于 0。
        penalty_high_low = torch.relu(pred_low - pred_high).mean()
        
        # 规则2: close 必须在 [low, high] 区间内。
        penalty_close_range = (torch.relu(pred_low - pred_close) + torch.relu(pred_close - pred_high)).mean()
        
        # 规则3: 振幅的数学关系。
        # 注意：为避免除以0，我们加上一个很小的数epsilon
        epsilon = 1e-6
        calculated_amplitude = (pred_high - pred_low) / (pred_close + epsilon) * 100
        penalty_amplitude_relation = self.mse_loss(calculated_amplitude, pred_amplitude)
        
        penalty = penalty_high_low + penalty_close_range + penalty_amplitude_relation
        
        # 3. 总损失 = MSE + lambda * 惩罚
        total_loss = mse + self.lambda_penalty * penalty
        
        return total_loss

# =============================================================================
# 5. 训练和推理 (已更新)
# =============================================================================
def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1, :]
        tgt_output = tgt[:, 1:, :]
        optimizer.zero_grad()
        prediction = model(src, tgt_input, device)
        
        # 使用自定义损失函数
        loss = criterion(prediction, tgt_output)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def predict_sequence(model, src_sequence, prediction_steps, device):
    model.eval()
    encoder_input = torch.tensor(src_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    decoder_input = encoder_input
    
    predicted_sequence = []

    with torch.no_grad():
        for _ in range(prediction_steps):
            prediction = model(encoder_input, decoder_input, device)
            next_step_prediction = prediction[:, -1:, :]
            predicted_sequence.append(next_step_prediction.cpu().numpy())
            decoder_input = torch.cat([decoder_input[:, 1:, :], next_step_prediction], dim=1)
            
    return np.concatenate(predicted_sequence, axis=1).squeeze(0)

# =============================================================================
# 6. 主执行函数 (已更新)
# =============================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT = 128, 8, 3, 3, 512, 0.1
    EPOCHS, BATCH_SIZE, LEARNING_RATE, LAMBDA_PENALTY = 30, 32, 0.0001, 1.0 # 增加Epoch和惩罚系数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据准备 ---
    train_data, inference_input_data, scalers, feature_columns, inference_df = prepare_data("600036")
    if train_data is None: exit()
    
    train_dataset = StockDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 2. 模型和损失函数初始化 ---
    model = StockSeq2SeqTransformer(
        num_features=len(feature_columns),
        d_model=D_MODEL, nhead=NHEAD, 
        num_encoder_layers=NUM_ENCODER_LAYERS, 
        num_decoder_layers=NUM_DECODER_LAYERS, 
        dim_feedforward=DIM_FEEDFORWARD, 
        dropout=DROPOUT
    ).to(device)
    
    # 使用自定义损失函数
    criterion = CustomFinancialLoss(feature_columns=feature_columns, lambda_penalty=LAMBDA_PENALTY)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("--- Model and Custom Loss Initialized ---")

    # --- 3. 训练模型 ---
    print("\n--- Training Started ---")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device)
        print(f"Epoch {epoch+1:02}/{EPOCHS} | Train Loss: {train_loss:.6f}")
    print("--- Training Finished ---\n")

    # --- 4. 执行推理 ---
    print("--- Performing Inference ---")
    predicted_scaled = predict_sequence(model, inference_input_data, prediction_steps=5, device=device)
    
    # --- 5. 反归一化并展示结果 ---
    predicted_df = pd.DataFrame(columns=feature_columns)
    for i, col in enumerate(feature_columns):
        predicted_df[col] = scalers[col].inverse_transform(predicted_scaled[:, i].reshape(-1, 1)).flatten()
    
    print("\n--- Prediction Results ---")
    print("Input Sequence Head (Last 5 days of clean data):")
    print(inference_df.tail())
    print("\nPredicted Sequence (Next 5 days):")
    print(predicted_df)