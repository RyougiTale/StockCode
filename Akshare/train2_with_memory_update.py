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
# 1. 数据集类 (已重构为预测“变化率”)
# =============================================================================
class StockDataset(Dataset):
    def __init__(self, dataframe, lookback_period):
        self.lookback_period = lookback_period
        
        self.df = dataframe.copy()
        # --- 特征工程 ---
        self.df['SMA20'] = self.df['close'].rolling(window=20).mean()
        self.df['SMA60'] = self.df['close'].rolling(window=60).mean()
        # 确保pct_chg在最前面计算，以防数据对齐问题
        self.df['pct_chg'] = self.df['close'].pct_change() * 100
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate', 'SMA20', 'SMA60']
        self.features = self.df[self.feature_columns]
        
        # --- 数据归一化 ---
        # 归一化所有输入特征
        self.feature_scaler = MinMaxScaler()
        self.scaled_features = self.feature_scaler.fit_transform(self.features)
        
        # 单独为目标（涨跌幅）创建一个归一化器
        self.target_scaler = MinMaxScaler()
        self.scaled_targets = self.target_scaler.fit_transform(self.df[['pct_chg']])

    def __len__(self):
        return len(self.scaled_features) - self.lookback_period

    def __getitem__(self, idx):
        # 输入序列是所有归一化后的特征
        input_seq = self.scaled_features[idx : idx + self.lookback_period]
        # 目标值是下一天的归一化后的涨跌幅
        target_val = self.scaled_targets[idx + self.lookback_period]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_val, dtype=torch.float32)

# =============================================================================
# 2. 模型构建 (Encoder-Only架构)
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

class TransformerPredictor(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, 1) # 输出一个值（预测的涨跌幅）

    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        prediction = self.fc_out(output)
        return prediction

# =============================================================================
# 3. 训练和推理
# =============================================================================

# --- 具有方向惩罚的自定义损失函数 ---
class DirectionalMSELoss(nn.Module):
    def __init__(self, lambda_penalty=1.0):
        super(DirectionalMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_penalty = lambda_penalty

    def forward(self, prediction, target):
        mse = self.mse_loss(prediction, target)
        epsilon = 1e-6
        product = torch.sign(prediction) * torch.sign(target + epsilon)
        directional_penalty = torch.relu(-product).mean()
        total_loss = mse + self.lambda_penalty * directional_penalty
        return total_loss

def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0
    for input_seq, target_val in dataloader:
        input_seq, target_val = input_seq.to(device), target_val.to(device)
        optimizer.zero_grad()
        prediction = model(input_seq)
        loss = criterion(prediction, target_val)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 推理函数已重构，实现更真实的自回归预测 ---
def predict_future(model, historical_df, dataset, prediction_steps, device):
    model.eval()
    
    # 复制一份历史数据，用于滚动更新
    live_sequence_df = historical_df.copy()
    
    # 存储最终的预测结果行
    predicted_rows = []

    with torch.no_grad():
        for _ in range(prediction_steps):
            # 1. 准备当前序列作为模型输入
            scaled_input = dataset.feature_scaler.transform(live_sequence_df[dataset.feature_columns])
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

            # 2. 预测下一个时间步的归一化涨跌幅
            prediction_scaled = model(input_tensor)
            
            # 3. 反归一化得到真实的涨跌幅
            predicted_pct_chg_real = dataset.target_scaler.inverse_transform(prediction_scaled.cpu().numpy()).item()

            # 4. 基于预测的涨跌幅，构建一个完整的、逻辑自洽的“未来”行
            last_row = live_sequence_df.iloc[-1]
            new_row = {}

            # 计算新的收盘价
            new_row['close'] = last_row['close'] * (1 + predicted_pct_chg_real / 100)
            
            # 用逻辑和统计规律构建其他核心特征
            new_row['open'] = last_row['close'] # 假设：开盘价=昨日收盘价
            avg_amplitude_pct = (live_sequence_df['high'] - live_sequence_df['low']).mean() / live_sequence_df['close'].mean()
            amplitude_val = new_row['close'] * avg_amplitude_pct
            new_row['high'] = new_row['close'] + amplitude_val / 2
            new_row['low'] = new_row['close'] - amplitude_val / 2
            new_row['volume'] = live_sequence_df['volume'].mean() # 假设：成交量=近期均值
            
            # 确保价格逻辑正确
            new_row['high'] = max(new_row['high'], new_row['open'], new_row['close'])
            new_row['low'] = min(new_row['low'], new_row['open'], new_row['close'])

            # 将新行转换为DataFrame，以便计算衍生特征
            new_row_df = pd.DataFrame([new_row])
            
            # 将新行加入历史序列，用于计算需要滚动窗口的衍生特征
            temp_df_for_calc = pd.concat([live_sequence_df, new_row_df], ignore_index=True)
            
            # 计算所有衍生特征
            new_row['turnover'] = new_row['volume'] * new_row['close']
            new_row['amplitude'] = (new_row['high'] - new_row['low']) / new_row['close'] * 100
            new_row['pct_chg'] = predicted_pct_chg_real
            new_row['chg_amount'] = new_row['close'] - last_row['close']
            new_row['turnover_rate'] = new_row['volume'] / 1e9 # 简化
            new_row['SMA20'] = temp_df_for_calc['close'].rolling(window=20).mean().iloc[-1]
            new_row['SMA60'] = temp_df_for_calc['close'].rolling(window=60).mean().iloc[-1]
            
            # 5. 将这个完整的、新构建的行添加到结果中
            final_new_row_df = pd.DataFrame([new_row], columns=dataset.feature_columns)
            predicted_rows.append(final_new_row_df)
            
            # 6. 【核心】滚动更新我们的“历史”序列，用于下一次预测
            live_sequence_df = pd.concat([live_sequence_df.iloc[1:], final_new_row_df], ignore_index=True)

    # 将所有预测的行合并成一个最终的DataFrame
    return pd.concat(predicted_rows, ignore_index=True)

# =============================================================================
# 4. 主执行函数
# =============================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    LOOKBACK_PERIOD = 60
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT = 64, 4, 2, 256, 0.1
    EPOCHS, BATCH_SIZE, LEARNING_RATE = 40, 32, 0.0005
    LAMBDA_DIRECTIONAL = 0.5 # 方向惩罚的权重
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据准备 ---
    data_filepath = "stock_data.csv"
    create_dummy_stock_data(data_filepath)
    full_df = pd.read_csv(data_filepath)
    
    train_df = full_df[:-LOOKBACK_PERIOD]
    inference_history_df = full_df.tail(LOOKBACK_PERIOD).reset_index(drop=True)

    train_dataset = StockDataset(train_df, LOOKBACK_PERIOD)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"--- Data Ready ---")

    # --- 2. 模型初始化 ---
    model = TransformerPredictor(
        num_features=len(train_dataset.feature_columns), 
        d_model=D_MODEL, nhead=NHEAD, 
        num_encoder_layers=NUM_ENCODER_LAYERS, 
        dim_feedforward=DIM_FEEDFORWARD, 
        dropout=DROPOUT
    ).to(device)
    
    # 使用新的自定义损失函数
    criterion = DirectionalMSELoss(lambda_penalty=LAMBDA_DIRECTIONAL)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("--- Model Initialized with Directional Loss ---")

    # --- 3. 训练模型 ---
    print("\n--- Training Started ---")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device)
        print(f"Epoch {epoch+1:02}/{EPOCHS} | Train Loss: {train_loss:.6f}")
    print("--- Training Finished ---\n")

    # --- 4. 执行推理 ---
    print("--- Performing Inference ---")
    predicted_df = predict_future(model, inference_history_df, train_dataset, prediction_steps=5, device=device)
    
    print("\n--- Prediction Results ---")
    print("Last 5 days of historical data:")
    print(inference_history_df.tail())
    print("\nPredicted Sequence (Next 5 days):")
    print(predicted_df)