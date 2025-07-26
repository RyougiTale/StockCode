import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from transformer import config
from transformer.data_utils import prepare_data, StockDataset
from transformer.model import StockSeq2SeqTransformer
from transformer.engine import train_epoch

def main(stock_code):
    """
    主训练函数。
    """
    # --- 1. 数据准备 ---
    train_data, _, _, feature_columns, _ = prepare_data(stock_code)
    if train_data is None:
        print(f"Could not prepare data for {stock_code}. Exiting.")
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

    # --- 4. 保存模型 ---
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"--- Model Saved to {config.MODEL_PATH} ---\n")

if __name__ == '__main__':
    main(STOCK_CODE)