import torch

STOCK_CODES = ["002415"]

SOFT_LABEL_CONFIG = {
    "LOOK_FORWARD_DAYS": 20,
    "CLASS_CENTERS": torch.tensor([-0.10, -0.045, 0.0, 0.045, 0.10]),
    "TEMPERATURE": 0.002
}

DAILY_SEQ_LEN = 60
WEEKLY_SEQ_LEN = 52
MONTHLY_SEQ_LEN = 24


BASE_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
# 可以添加更多技术指标，例如 'SMA5', 'RSI14', 'MACD' 等
# 注意：这里的名称需要与 data_utils.py 中的计算函数对应
# 为不同时间尺度定义不同的技术指标
TECH_INDICATORS = {
    'daily': ['SMA20', 'SMA60'],
    'weekly': ['SMA20', 'SMA60'],
    'monthly': ['SMA20', 'SMA60']
}

# 为每个时间尺度预先定义好完整的特征列
FEATURE_COLUMNS = {
    'daily': BASE_FEATURES + TECH_INDICATORS['daily'],
    'weekly': BASE_FEATURES + TECH_INDICATORS['weekly'],
    'monthly': BASE_FEATURES + TECH_INDICATORS['monthly']
}

# =============================================================================
# 4. 模型架构 (Model Architecture)
# =============================================================================
# 共享的编码器配置
SHARED_ENCODER_CONFIG = {
    'd_model': 128,
    'nhead': 8,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.3
}

# 融合层和输出层配置
FUSION_DIM = 256
NUM_CLASSES = 5

# =============================================================================
# 5. 训练过程 (Training Process)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.08


MODEL_DIR = "models/long_way"
MODEL_NAME = "market_classifier_model.pth"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"