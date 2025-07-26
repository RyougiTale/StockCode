import torch
import pandas as pd
import argparse
import os

from transformer import config
from transformer.data_utils import prepare_data
from transformer.model import StockSeq2SeqTransformer
from transformer.engine import predict_sequence

def main(stock_code):
    """
    主预测函数。
    """
    # --- 1. 检查模型文件是否存在 ---
    if not os.path.exists(config.MODEL_PATH):
        print(f"Model file not found at {config.MODEL_PATH}. Please train the model first by running train.py.")
        return

    # --- 2. 数据准备 ---
    # 我们只需要用于推理的数据
    _, inference_input_data, scalers, feature_columns, inference_df = prepare_data(stock_code)
    if inference_input_data is None:
        print(f"Could not prepare data for {stock_code}. Exiting.")
        return

    # --- 3. 加载模型 ---
    model = StockSeq2SeqTransformer(
        num_features=len(feature_columns),
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    print("--- Model Loaded ---")

    # --- 4. 执行推理 ---
    print(f"--- Performing Inference for the next {config.PREDICTION_STEPS} steps ---")
    predicted_scaled = predict_sequence(model, inference_input_data, prediction_steps=config.PREDICTION_STEPS, device=config.DEVICE)
    
    # --- 5. 反归一化并展示结果 ---
    predicted_df = pd.DataFrame(columns=feature_columns)
    for i, col in enumerate(feature_columns):
        predicted_df[col] = scalers[col].inverse_transform(predicted_scaled[:, i].reshape(-1, 1)).flatten()
    
    print("\n--- Prediction Results ---")
    print("Input Sequence Head (Last 5 days of available data):")
    print(inference_df.tail())
    print(f"\nPredicted Sequence (Next {config.PREDICTION_STEPS} days):")
    print(predicted_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict stock prices using a trained Transformer model.")
    parser.add_argument(
        '--stock_code', 
        type=str, 
        default=config.STOCK_CODE, 
        help=f'Stock code to predict on (default: {config.STOCK_CODE})'
    )
    args = parser.parse_args()
    
    main(args.stock_code)