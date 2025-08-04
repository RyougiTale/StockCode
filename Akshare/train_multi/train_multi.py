import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pickle

from transformer import config
from transformer.data_utils import prepare_multistock_data, StockDataset
from transformer.model import StockSeq2SeqTransformer
from transformer.engine import train_epoch

def main():
    """
    主训练函数，现在用于训练多只股票的模型。
    """
    # --- 1. 数据准备 ---
    # 从配置文件读取股票列表
    stock_codes = config.STOCK_CODES
    
    # 调用新的多股票数据准备函数
    train_data, scalers, feature_columns = prepare_multistock_data(stock_codes)
    
    if train_data is None:
        print(f"Could not prepare data for the given stock codes. Exiting.")
        return

    train_dataset = StockDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # --- 2. 模型初始化 ---
    model = StockSeq2SeqTransformer(
        num_features=len(feature_columns),
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print("--- Model Initialized ---")

    # --- 3. 训练模型 ---
    print(f"\n--- Training Started on {config.DEVICE} ---")
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, config.DEVICE)
        print(f"Epoch {epoch+1:02}/{config.EPOCHS} | Train Loss: {train_loss:.6f}")
    print("--- Training Finished ---\n")

    # --- 4. 保存模型和Scalers ---
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 保存模型状态
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"--- Model Saved to {config.MODEL_PATH} ---")
    
    # 保存scalers字典
    scaler_path = os.path.join(config.MODEL_DIR, "scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"--- Scalers Saved to {scaler_path} ---\n")

if __name__ == '__main__':
    main()