import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import MinMaxScaler

# 导入需要测试的模块
from Akshare.train import PositionalEncoding, StockSeq2SeqTransformer, predict_sequence
from Akshare.stock_util import read_history_stock_by_code

class TestModelInference(unittest.TestCase):

    def setUp(self):
        """
        在每个测试方法运行前执行的设置代码。
        """
        print("\n--- Setting up for test ---")
        # 定义模型超参数 (必须与训练时完全一致)
        self.D_MODEL = 128
        self.NHEAD = 8
        self.NUM_ENCODER_LAYERS = 3
        self.NUM_DECODER_LAYERS = 3
        self.DIM_FEEDFORWARD = 512
        self.DROPOUT = 0.1
        self.MAX_SEQ_LEN = 60
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "Akshare/models/stock_transformer_model.pth"
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate', 'SMA20', 'SMA60']
        self.num_features = len(self.feature_columns)
        # 设置pandas显示选项，确保所有列都能被打印出来
        pd.set_option('display.max_columns', None)

    def test_predict_sequence(self):
        """
        单元测试：加载已训练模型并进行序列预测。
        """
        print(f"--- 单元测试：加载模型并预测未来序列 ---")
        
        # 1. 检查模型文件是否存在
        self.assertTrue(os.path.exists(self.model_path), f"模型文件不存在: {self.model_path}")

        # 2. 准备模型
        model = StockSeq2SeqTransformer(
            num_features=self.num_features, d_model=self.D_MODEL, nhead=self.NHEAD, 
            num_encoder_layers=self.NUM_ENCODER_LAYERS, num_decoder_layers=self.NUM_DECODER_LAYERS, 
            dim_feedforward=self.DIM_FEEDFORWARD, dropout=self.DROPOUT
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print("模型加载成功。")

        # 3. 准备数据
        stock_code_to_test = "600036" # 招商银行
        full_df = read_history_stock_by_code(stock_code_to_test)
        self.assertFalse(full_df.empty, f"股票 {stock_code_to_test} 的数据加载失败。")
        self.assertGreaterEqual(len(full_df), self.MAX_SEQ_LEN, "数据长度不足以进行推理。")

        # 特征工程
        full_df['SMA20'] = full_df['close'].rolling(window=20).mean()
        full_df['SMA60'] = full_df['close'].rolling(window=60).mean()
        full_df.dropna(inplace=True)

        # 重建Scaler
        train_data_for_scaler = full_df[:-self.MAX_SEQ_LEN]
        scaler = MinMaxScaler()
        scaler.fit(train_data_for_scaler[self.feature_columns])

        # 准备推理输入
        inference_source_df = full_df.tail(self.MAX_SEQ_LEN)
        inference_input_data = scaler.transform(inference_source_df[self.feature_columns])
        print("测试数据准备完毕。")

        # 4. 执行预测
        prediction_steps = 5
        print(inference_source_df.tail())
        predicted_scaled = predict_sequence(model, inference_input_data, prediction_steps, self.device)
        print("模型预测执行完毕。")

        # 5. 反归一化并断言结果
        predicted_features = scaler.inverse_transform(predicted_scaled)
        predicted_df = pd.DataFrame(predicted_features, columns=self.feature_columns)
        
        self.assertIsInstance(predicted_df, pd.DataFrame, "预测结果应为DataFrame。")
        self.assertFalse(predicted_df.empty, "预测结果不应为空。")
        self.assertEqual(predicted_df.shape, (prediction_steps, self.num_features), "预测结果的形状不正确。")
        
        print("\n--- 预测结果 ---")
        print(predicted_df)
        print("\n--- 单元测试成功 ---")


if __name__ == "__main__":
    unittest.main()