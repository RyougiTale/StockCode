# [important!!!]
# 训练阶段配置：'pretraining' 或 'finetuning'
# TRAINING_PHASE = "pretraining"  # 设置为预训练模式
TRAINING_PHASE = "finetuning"

import torch
import logging

# =============================================================================
# 0. 日志和调试配置 (Logging & Debug Configuration)
# =============================================================================
DEBUG_MODE = True  # 开发期间保持True，方便调试
LOGGING_LEVEL = logging.DEBUG  # 开发期间使用DEBUG级别
ENABLE_PERFORMANCE_LOGGING = True  # 记录性能指标
ENABLE_DATA_VALIDATION = True  # 开发期间启用数据验证

# =============================================================================
# 1. 股票代码和数据配置 (Stock Codes & Data Configuration)  
# =============================================================================
# python training_data_update.py --start 1970-01-01 --end 2025-08-07

# 多股票预训练配置
PRETRAINING_CONFIG = {
    "use_all_available": False,     # 暂时设置为False，使用手动指定的股票列表进行测试
    "stock_list_file": None,        # 可选：从文件加载股票列表，格式为每行一个股票代码
    "min_data_points": 500,         # 降低要求便于测试
    "max_stocks": 5,                # 限制为5只股票进行测试
    "exclude_stocks": ["ST*", "*ST"],  # 排除的股票模式（ST股票等）
}

# 单股票微调配置  
FINETUNING_CONFIG = {
    "target_stock": "002415",   # 目标微调股票
    "use_pretrained_model": True,  # 是否使用预训练模型
    "freeze_encoder_layers": 0,    # 冻结编码器层数（0表示全部微调）
}

# 根据训练阶段动态设置当前使用的股票代码（兼容性保持）
if TRAINING_PHASE == "pretraining":
    # 测试用的股票列表
    # STOCK_CODES = ['002001', '002463', '002648', '600489', '601989', '000425', '002422', '600150', '600584', '601166', '601669', '688041', '688256', '688271', '600009', '000538', '002252', '300896', '600760', '002179', '601006', '601816', '600547', '600030', '688012', '300308', '688008', '688111', '002466', '601390', '000938', '002371', '002241', '002230', '600111', '000338', '000792', '603986', '001979', '601985', '600893', '002049', '600089', '600426', '688981', '603993', '002460', '601600', '600010', '603799', '600989', '000100', '600941', '300274', '601919', '300014', '300124', '300760', '601728', '300750', '600436', '600438', '601012', '603501', '300122', '002714', '002475', '300015', '600031', '600346', '600406', '603259', '601888', '600309', '002352', '002027', '000725', '600690', '000333', '600276', '002594', '002415', '601668', '601766', '601899', '601857', '601088', '601318', '600048', '601398', '000002', '000063', '000651', '600019', '600028', '600036', '600050', '600519', '600585', '600900']
    STOCK_CODES = ['002001', '002463', '002648', '600489', '601989']
elif TRAINING_PHASE == "finetuning": 
    STOCK_CODES = [FINETUNING_CONFIG["target_stock"]]
else:
    raise ValueError(f"Invalid TRAINING_PHASE: {TRAINING_PHASE}. Must be 'pretraining' or 'finetuning'")

TRAINING_YEARS = 15 # 只使用最近N年的数据进行训练

# 市场时期切分配置 (Market Period Segmentation)
MARKET_PERIOD_CONFIG = {
    "enable_period_split": True,        # 是否启用市场时期切分
    "recent_years": 3,                  # 最近时期：最近3年
    "middle_years": 8,                  # 中期：最近3年到最近8年（即第3-8年）
    "distant_years": None,              # 远期：8年以前的所有数据（None表示无限制）
    "min_samples_per_period": 100,      # 每个时期最少样本数，不足则跳过该时期
    "period_names": ["recent", "middle", "distant"]  # 时期名称
}

SOFT_LABEL_CONFIG = {
    "LOOK_FORWARD_DAYS": 20,
    "CLASS_CENTERS": torch.tensor([-0.10, -0.045, 0.0, 0.045, 0.10]),
    "TEMPERATURE": 0.1  # 增加温度参数，适合相对化指标（0-1范围）
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
    'dropout': 0.3  # 从0.15增加到0.3，增强正则化
}

# 融合层和输出层配置
FUSION_DIM = 256
NUM_CLASSES = 5

# =============================================================================
# 5. 训练过程 (Training Process)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预训练阶段配置
PRETRAINING_EPOCHS = 500
PRETRAINING_BATCH_SIZE = 128  
PRETRAINING_LEARNING_RATE = 0.0001  # 预训练用较高学习率快速收敛

# 微调阶段配置  
FINETUNING_EPOCHS = 50       # 微调轮数可以适当减少，因为有预训练基础
FINETUNING_BATCH_SIZE = 64    # 微调用较小批次，更精细
FINETUNING_LEARNING_RATE = 0.00001  # 微调用更小学习率，避免破坏预训练特征

# 根据训练阶段动态设置当前参数
if TRAINING_PHASE == "pretraining":
    EPOCHS = PRETRAINING_EPOCHS
    BATCH_SIZE = PRETRAINING_BATCH_SIZE
    LEARNING_RATE = PRETRAINING_LEARNING_RATE
elif TRAINING_PHASE == "finetuning":
    EPOCHS = FINETUNING_EPOCHS
    BATCH_SIZE = FINETUNING_BATCH_SIZE
    LEARNING_RATE = FINETUNING_LEARNING_RATE

WEIGHT_DECAY = 0.15  # 从0.08增加到0.15，增强L2正则化
GRAD_CLIP_NORM = 2.0  # 3D多任务学习需要更大的梯度裁剪阈值

# 模型保存路径配置
MODEL_DIR = "models/long_way"
PRETRAINING_MODEL_NAME = "market_pretrained_model.pth"
FINETUNING_MODEL_NAME = "market_finetuned_model.pth"

# 根据训练阶段设置模型文件名
if TRAINING_PHASE == "pretraining":
    MODEL_NAME = PRETRAINING_MODEL_NAME
elif TRAINING_PHASE == "finetuning":
    MODEL_NAME = FINETUNING_MODEL_NAME

MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"