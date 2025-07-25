import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1. 配置 (Configuration)
# =============================================================================
MAX_SEQ_LEN = 60  # 使用过去60天的数据进行预测

# =============================================================================
# 2. 数据集类 (Dataset Class)
# =============================================================================
class StockDataset(Dataset):
    """
    用于股票数据的自定义PyTorch数据集。
    注意：这个类现在只接收已经处理好的numpy array。
    """
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences) - MAX_SEQ_LEN

    def __getitem__(self, idx):
        input_sequence = self.sequences[idx : idx + MAX_SEQ_LEN]
        target_sequence = self.sequences[idx + 1 : idx + MAX_SEQ_LEN + 1]
        return torch.tensor(input_sequence, dtype=torch.float32), torch.tensor(target_sequence, dtype=torch.float32)

# =============================================================================
# 3. 模型构建 (Model Building)
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
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class StockSeq2SeqTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(StockSeq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.encoder_input_projection = nn.Linear(num_features, d_model)
        self.decoder_input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.fc_out = nn.Linear(d_model, num_features)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, device):
        # 接收 [batch, seq, feature]，转换为 [seq, batch, feature]
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0), device)
        src_emb = self.pos_encoder(self.encoder_input_projection(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.decoder_input_projection(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        
        # 将输出转换回 [batch, seq, feature]
        return self.fc_out(output.transpose(0, 1))

# =============================================================================
# 4. 训练和推理 (Training and Inference)
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
        loss = criterion(prediction, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def predict_sequence(model, src_sequence, prediction_steps, device):
    model.eval()
    src_tensor = torch.tensor(src_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    tgt_input = src_tensor[:, -1:, :]
    predicted_sequence = []
    with torch.no_grad():
        for _ in range(prediction_steps):
            prediction = model(src_tensor, tgt_input, device)
            next_step_prediction = prediction[:, -1:, :]
            predicted_sequence.append(next_step_prediction.cpu().numpy())
            tgt_input = torch.cat([tgt_input, next_step_prediction], dim=1)
    return np.concatenate(predicted_sequence, axis=1).squeeze(0)

# =============================================================================
# 5. 主执行函数 (Main Execution)
# =============================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT = 128, 8, 3, 3, 512, 0.1
    EPOCHS, BATCH_SIZE, LEARNING_RATE = 20, 32, 0.0001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 数据加载、处理和准备 ---
    from Akshare.stock_util import read_history_stock_by_code
    stock_code = "600036"
    print(f"--- 1. Loading, Processing and Preparing Data for {stock_code} ---")

    full_df = read_history_stock_by_code(stock_code)
    if full_df.empty: exit(f"No data for {stock_code}")

    # a. 特征工程
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
    full_df['SMA20'] = full_df['close'].rolling(window=20).mean()
    full_df['SMA60'] = full_df['close'].rolling(window=60).mean()
    feature_columns.extend(['SMA20', 'SMA60'])
    
    # b. 清洗数据 (统一dropna)
    clean_df = full_df[feature_columns].dropna().reset_index(drop=True)
    if len(clean_df) < MAX_SEQ_LEN * 2: exit("Not enough data after cleaning.")

    # c. 归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(clean_df)
    
    # d. 切分训练和推理数据
    train_data = scaled_data[:-MAX_SEQ_LEN]
    inference_input_data = scaled_data[-MAX_SEQ_LEN:]
    
    # e. 创建DataLoader
    train_dataset = StockDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("--- Data Ready ---")

    # --- 2. 模型初始化 ---
    print(f"--- 2. Initializing Model ---")
    num_features = len(feature_columns)
    model = StockSeq2SeqTransformer(
        num_features=num_features, d_model=D_MODEL, nhead=NHEAD, 
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, 
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("--- Model Initialized ---")

    # --- 3. 训练模型 ---
    print("\n--- 3. Training Started ---")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device)
        print(f"Epoch {epoch+1:02}/{EPOCHS} | Train Loss: {train_loss:.4f}")
    print("--- Training Finished ---\n")

    # --- 4. 保存模型 ---
    model_dir = "Akshare/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "stock_transformer_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"--- Model Saved to {model_path} ---\n")

    # --- 5. 执行推理 ---
    print("--- 5. Performing Inference ---")
    predicted_scaled = predict_sequence(model, inference_input_data, prediction_steps=5, device=device)
    
    # a. 反归一化
    predicted_features = scaler.inverse_transform(predicted_scaled)
    predicted_df = pd.DataFrame(predicted_features, columns=feature_columns)
    
    print("\n--- Prediction Results ---")
    print("Input Sequence Head (Last 5 days of clean data):")
    print(clean_df.tail())
    print("\nPredicted Sequence (Next 5 days):")
    print(predicted_df)