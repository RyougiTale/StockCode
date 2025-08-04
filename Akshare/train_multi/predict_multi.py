import torch
import pandas as pd
import argparse
import os
import pickle

from transformer import config
from transformer.data_utils import prepare_singlestock_data_for_inference
from transformer.model import StockSeq2SeqTransformer
from transformer.engine import predict_sequence

def main(stock_code_to_predict):
    """
    主预测函数。
    """
    # --- 1. 检查模型和scaler文件是否存在 ---
    model_path = config.MODEL_PATH
    scaler_path = os.path.join(config.MODEL_DIR, "scalers.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scalers file not found. Please train the model first by running train.py.")
        return

    # --- 2. 加载Scalers ---
    with open(scaler_path, 'rb') as f:
        all_scalers = pickle.load(f)
    
    if stock_code_to_predict not in all_scalers:
        print(f"Error: Scaler for stock {stock_code_to_predict} not found.")
        print("Please ensure this stock was part of the training set.")
        return
        
    # --- 3. 数据准备 ---
    inference_input, _, feature_columns, inference_df = prepare_singlestock_data_for_inference(stock_code_to_predict)
    
    if inference_input is None:
        print(f"Could not prepare data for {stock_code_to_predict}. Exiting.")
        return

    # --- 4. 加载模型 ---
    model = StockSeq2SeqTransformer(
        num_features=len(feature_columns),
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    print("--- Model Loaded ---")

    # --- 5. 执行推理 ---
    print(f"--- Performing Inference for {stock_code_to_predict} for the next {config.PREDICTION_STEPS} steps ---")
    predicted_scaled = predict_sequence(model, inference_input, prediction_steps=config.PREDICTION_STEPS, device=config.DEVICE)
    
    # --- 6. 反归一化并展示结果 ---
    # 获取这只特定股票的scalers
    stock_scalers = all_scalers[stock_code_to_predict]
    
    predicted_df = pd.DataFrame(columns=feature_columns)
    for i, col in enumerate(feature_columns):
        # 使用对应的scaler进行反归一化
        predicted_df[col] = stock_scalers[col].inverse_transform(predicted_scaled[:, i].reshape(-1, 1)).flatten()
    
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
        required=True,
        help='The stock code to predict on (e.g., 600036).'
    )
    args = parser.parse_args()
    
    main(args.stock_code)