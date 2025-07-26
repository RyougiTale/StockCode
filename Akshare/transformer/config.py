import torch

# =============================================================================
# 1. 数据相关配置
# =============================================================================
# 定义了输入序列的最大长度。模型将处理长度为60的时间步。
MAX_SEQ_LEN = 60
# 用于训练和推断的股票代码
STOCK_CODE = "600036"
# 预测未来多少天
PREDICTION_STEPS = 5

# =============================================================================
# 2. 模型架构相关配置 (Transformer Hyperparameters)
# =============================================================================
# D_MODEL: Transformer模型中Encoder和Decoder的输入和输出维度。
D_MODEL = 128
# NHEAD: 多头注意力机制中的头数。D_MODEL必须能被NHEAD整除。
NHEAD = 8
# NUM_ENCODER_LAYERS: Encoder中的Encoder Layer层数。
NUM_ENCODER_LAYERS = 3
# NUM_DECODER_LAYERS: Decoder中的Decoder Layer层数。
NUM_DECODER_LAYERS = 3
# DIM_FEEDFORWARD: Encoder和Decoder中前馈神经网络的维度。
DIM_FEEDFORWARD = 512
# DROPOUT: Dropout的概率。
DROPOUT = 0.1

# =============================================================================
# 3. 训练相关配置 (Training Hyperparameters)
# =============================================================================
# EPOCHS: 训练的总轮数。
EPOCHS = 20
# BATCH_SIZE: 每一批次训练的数据量。
BATCH_SIZE = 32
# LEARNING_RATE: 学习率。
LEARNING_RATE = 0.0001

# =============================================================================
# 4. 环境配置 (Environment Configuration)
# =============================================================================
# 自动选择可用的设备 (CUDA或CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 5. 文件路径配置 (File Paths)
# =============================================================================
# 模型保存的目录
MODEL_DIR = "models"
# 模型保存的完整路径
MODEL_PATH = f"{MODEL_DIR}/stock_transformer_model.pth"