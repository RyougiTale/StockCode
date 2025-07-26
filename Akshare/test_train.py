import unittest
import torch
import pandas as pd
import os

# 导入需要测试的模块 (从 train2.py)
from train2 import StockSeq2SeqTransformer, predict_sequence, prepare_data

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "models/stock_transformer_model.pth"
        pd.set_option('display.max_columns', None)

    def test_predict_sequence(self):
        """
        单元测试：加载已训练模型并进行序列预测。
        """
        print(f"--- 单元测试：加载模型并预测未来序列 ---")
        
        # 1. 检查模型文件是否存在
        self.assertTrue(os.path.exists(self.model_path), f"模型文件不存在: {self.model_path}")

        # 2. 准备数据 (完全复刻 train2.py 的逻辑)
        stock_code_to_test = "600036"
        _, inference_input_data, scalers, feature_columns, inference_df = prepare_data(stock_code_to_test)
        
        self.assertIsNotNone(inference_input_data, "数据准备失败，无法进行测试。")
        
        # 3. 准备模型
        model = StockSeq2SeqTransformer(
            num_features=len(feature_columns), d_model=self.D_MODEL, nhead=self.NHEAD, 
            num_encoder_layers=self.NUM_ENCODER_LAYERS, num_decoder_layers=self.NUM_DECODER_LAYERS, 
            dim_feedforward=self.DIM_FEEDFORWARD, dropout=self.DROPOUT
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print("模型加载成功。")

        # 4. 执行预测
        prediction_steps = 5
        predicted_scaled = predict_sequence(model, inference_input_data, prediction_steps, self.device)
        print("模型预测执行完毕。")

        # 5. 反归一化并断言结果
        predicted_df = pd.DataFrame(columns=feature_columns)
        for i, col in enumerate(feature_columns):
            predicted_df[col] = scalers[col].inverse_transform(predicted_scaled[:, i].reshape(-1, 1)).flatten()
        
        self.assertIsInstance(predicted_df, pd.DataFrame, "预测结果应为DataFrame。")
        self.assertFalse(predicted_df.empty, "预测结果不应为空。")
        self.assertEqual(predicted_df.shape, (prediction_steps, len(feature_columns)), "预测结果的形状不正确。")
        
        print("\n--- 预测结果 ---")
        print("Input Sequence Head (Last 5 days of clean data):")
        print(inference_df.tail())
        print("\nPredicted Sequence (Next 5 days):")
        print(predicted_df)
        print("\n--- 单元测试成功 ---")


if __name__ == "__main__":
    unittest.main()