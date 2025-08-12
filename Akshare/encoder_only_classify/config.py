import torch

# =============================================================================
# 1. 数据源与标签定义 (Data Source & Labeling)
# =============================================================================
# 用于训练的股票/指数代码。我们先用一个指数做例子，后续可以换成多只股票。
STOCK_CODES = ["600036"] # .SH for 沪市, .SZ for 深市

# 标签定义：未来 N 个交易日收盘价是否上涨
LABEL_LOOK_FORWARD_DAYS = 20

# =============================================================================
# 2. 输入序列长度 (Input Sequence Lengths)
# =============================================================================
DAILY_SEQ_LEN = 60
WEEKLY_SEQ_LEN = 52
MONTHLY_SEQ_LEN = 24

# =============================================================================
# 3. 特征工程 (Feature Engineering)
# =============================================================================
# 定义要使用的基础特征和技术指标
# OHLCV 是基础
BASE_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'pct_chg', 'chg_amount', 'turnover_rate']
# 可以添加更多技术指标，例如 'SMA5', 'RSI14', 'MACD' 等
# 注意：这里的名称需要与 data_utils.py 中的计算函数对应
TECH_INDICATORS = ['SMA20', 'SMA60']

# 最终送入模型的特征
# 最终送入模型的特征
FEATURE_COLUMNS = BASE_FEATURES + TECH_INDICATORS

# =============================================================================
# 4. 模型架构 (Model Architecture)
# =============================================================================
# 共享的编码器配置
SHARED_ENCODER_CONFIG = {
    'd_model': 128,
    'nhead': 8,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.1
}

# 融合层和输出层配置
FUSION_DIM = 256
NUM_CLASSES = 2  # 二分类：上涨/不上涨

# =============================================================================
# 5. 训练过程 (Training Process)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 900
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
# 使用 AdamW 优化器，并设置权重衰减
WEIGHT_DECAY = 0.01

# =============================================================================
# 6. 文件路径 (File Paths)
# =============================================================================
MODEL_DIR = "models/long_way"
MODEL_NAME = "market_classifier_model.pth"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"