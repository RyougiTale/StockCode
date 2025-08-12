import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from .model import PositionalEncoding, SingleTimeframeEncoder
except ImportError:
    # 如果相对导入失败，尝试直接导入
    from model import PositionalEncoding, SingleTimeframeEncoder

class MultiOutput3DModel(nn.Module):
    """
    3D多输出模型
    基于原有的MultiEncoderFusionModel，扩展为三个输出头
    """
    
    def __init__(self, daily_config, weekly_config, monthly_config, fusion_dim, dropout=0.1):
        super(MultiOutput3DModel, self).__init__()
        
        # 1. 复用原有的编码器架构
        self.daily_encoder = SingleTimeframeEncoder(**daily_config)
        self.weekly_encoder = SingleTimeframeEncoder(**weekly_config)
        self.monthly_encoder = SingleTimeframeEncoder(**monthly_config)
        
        # 每个专家的输出维度都是 d_model
        concatenated_dim = daily_config['d_model'] + weekly_config['d_model'] + monthly_config['d_model']
        
        # 2. 共享的融合层
        self.shared_fusion = nn.Sequential(
            nn.Linear(concatenated_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. 三个专门的输出头
        self.return_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 5)  # 5个回报率类别
        )
        
        self.sharpe_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 5)  # 5个夏普比率类别
        )
        
        self.drawdown_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 5)  # 5个最大回撤类别
        )
        
        print("3D多输出模型初始化完成:")
        print(f"  融合维度: {fusion_dim}")
        print(f"  输出维度: 回报率(5), 夏普比率(5), 最大回撤(5)")

    def forward(self, x_daily, x_weekly, x_monthly):
        """
        前向传播
        
        Args:
            x_daily: [batch_size, daily_seq_len, daily_feature_size]
            x_weekly: [batch_size, weekly_seq_len, weekly_feature_size]
            x_monthly: [batch_size, monthly_seq_len, monthly_feature_size]
            
        Returns:
            dict: 包含三个维度log概率的字典
        """
        # 1. 并行地从每个专家获取上下文向量
        daily_context = self.daily_encoder(x_daily)
        weekly_context = self.weekly_encoder(x_weekly)
        monthly_context = self.monthly_encoder(x_monthly)
        
        # 2. 将三个专家的意见拼接起来
        concatenated_context = torch.cat([daily_context, weekly_context, monthly_context], dim=1)
        
        # 3. 通过共享融合层
        shared_repr = self.shared_fusion(concatenated_context)
        
        # 4. 通过三个专门的输出头
        return_logits = self.return_head(shared_repr)
        sharpe_logits = self.sharpe_head(shared_repr)
        drawdown_logits = self.drawdown_head(shared_repr)
        
        # 5. 数值稳定性检查和处理
        return_logits = self._stabilize_logits(return_logits, "return")
        sharpe_logits = self._stabilize_logits(sharpe_logits, "sharpe")
        drawdown_logits = self._stabilize_logits(drawdown_logits, "drawdown")
        
        # 6. 转换为log概率
        return {
            'return': F.log_softmax(return_logits, dim=-1),
            'sharpe': F.log_softmax(sharpe_logits, dim=-1),
            'drawdown': F.log_softmax(drawdown_logits, dim=-1)
        }
    
    def _stabilize_logits(self, logits, head_name):
        """
        稳定化logits，避免数值问题
        """
        # 检查异常值
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: NaN or Inf detected in {head_name} logits")
            # 用零替换异常值
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits)
        
        # 限制logits的范围
        logits = torch.clamp(logits, min=-50, max=50)
        
        return logits

class Multi3DLoss(nn.Module):
    """
    3D多任务损失函数
    结合三个维度的KL散度损失
    """
    
    def __init__(self, weights=None):
        super(Multi3DLoss, self).__init__()
        
        # 三个维度的权重
        if weights is None:
            self.weights = {'return': 1.0, 'sharpe': 0.8, 'drawdown': 0.6}
        else:
            self.weights = weights
            
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        print("3D多任务损失函数初始化:")
        print(f"  权重: {self.weights}")
    
    def forward(self, predictions, targets):
        """
        计算3D多任务损失
        
        Args:
            predictions (dict): 模型预测的log概率
            targets (dict): 目标软标签
            
        Returns:
            dict: 包含总损失和各维度损失的字典
        """
        losses = {}
        total_loss = 0
        
        for dim in ['return', 'sharpe', 'drawdown']:
            if dim in predictions and dim in targets:
                # 稳定化目标标签
                target = self._stabilize_target(targets[dim])
                
                # 计算KL散度
                try:
                    loss = self.kl_loss(predictions[dim], target)
                    
                    # 检查损失有效性
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"WARNING: Invalid loss for {dim}, using fallback")
                        loss = self._fallback_loss(predictions[dim], target)
                    
                    losses[dim] = loss
                    total_loss += self.weights[dim] * loss
                    
                except Exception as e:
                    print(f"ERROR in {dim} loss calculation: {e}")
                    # 使用备选损失计算
                    loss = self._fallback_loss(predictions[dim], target)
                    losses[dim] = loss
                    total_loss += self.weights[dim] * loss
        
        losses['total'] = total_loss
        return losses
    
    def _stabilize_target(self, target):
        """
        稳定化目标标签
        """
        epsilon = 1e-8
        target = torch.clamp(target, min=epsilon, max=1.0-epsilon)
        # 重新归一化
        target = target / target.sum(dim=1, keepdim=True)
        return target
    
    def _fallback_loss(self, log_probs, target):
        """
        备选损失计算（手动KL散度）
        """
        epsilon = 1e-8
        probs = torch.exp(log_probs)
        probs = torch.clamp(probs, min=epsilon, max=1.0-epsilon)
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # 手动计算KL散度: KL(P||Q) = sum(P * log(P/Q))
        kl_manual = (target * (torch.log(target + epsilon) - torch.log(probs + epsilon))).sum(dim=1).mean()
        return torch.clamp(kl_manual, min=0, max=100)

def create_3d_model(config_module):
    """
    创建3D模型的工厂函数
    """
    daily_config = {
        'feature_size': len(config_module.FEATURE_COLUMNS['daily']), 
        **config_module.SHARED_ENCODER_CONFIG
    }
    weekly_config = {
        'feature_size': len(config_module.FEATURE_COLUMNS['weekly']), 
        **config_module.SHARED_ENCODER_CONFIG
    }
    monthly_config = {
        'feature_size': len(config_module.FEATURE_COLUMNS['monthly']), 
        **config_module.SHARED_ENCODER_CONFIG
    }
    
    model = MultiOutput3DModel(
        daily_config=daily_config,
        weekly_config=weekly_config,
        monthly_config=monthly_config,
        fusion_dim=config_module.FUSION_DIM,
        dropout=0.1
    )
    
    return model

def test_3d_model():
    """
    测试3D模型
    """
    print("=== 测试3D模型 ===")
    
    # 模拟配置
    class MockConfig:
        FEATURE_COLUMNS = {
            'daily': ['open', 'high', 'low', 'close', 'volume'] * 2,  # 10个特征
            'weekly': ['open', 'high', 'low', 'close', 'volume'] * 2,
            'monthly': ['open', 'high', 'low', 'close', 'volume'] * 2
        }
        SHARED_ENCODER_CONFIG = {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1
        }
        FUSION_DIM = 256
    
    # 创建模型
    model = create_3d_model(MockConfig)
    
    # 创建测试数据
    batch_size = 4
    daily_data = torch.randn(batch_size, 60, 10)
    weekly_data = torch.randn(batch_size, 52, 10)
    monthly_data = torch.randn(batch_size, 24, 10)
    
    # 前向传播
    print("测试前向传播...")
    with torch.no_grad():
        outputs = model(daily_data, weekly_data, monthly_data)
    
    print("输出形状:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
        print(f"  {key} 范围: [{value.min().item():.4f}, {value.max().item():.4f}]")
    
    # 测试损失函数
    print("\n测试损失函数...")
    loss_fn = Multi3DLoss()
    
    # 创建目标标签
    targets = {
        'return': torch.softmax(torch.randn(batch_size, 5), dim=1),
        'sharpe': torch.softmax(torch.randn(batch_size, 5), dim=1),
        'drawdown': torch.softmax(torch.randn(batch_size, 5), dim=1)
    }
    
    losses = loss_fn(outputs, targets)
    print("损失值:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("✓ 3D模型测试通过")

if __name__ == '__main__':
    test_3d_model()