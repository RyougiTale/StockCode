# 测试训练时的标签生成过程
from long_way.improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from long_way import config
import torch
# 模拟训练时的过程
label_generator = ImprovedThreeDimensionalLabelGenerator(
    look_forward_days=config.SOFT_LABEL_CONFIG['LOOK_FORWARD_DAYS'],
    temperature=config.SOFT_LABEL_CONFIG['TEMPERATURE'],
    use_relative_metrics=True
)
# 模拟一个指标值
test_metrics = {
    'total_return': 0.05,  # 5%回报率
    'sharpe_ratio': 1.5,
    'max_drawdown': -0.08
}
print('=== 训练时标签生成过程 ===')
print('输入指标:', test_metrics)
# 这里需要模拟股票分布，但关键是看相对化过程
print('\\n在use_relative_metrics=True模式下:')
print('训练时使用的中心点是: [0.0, 0.25, 0.5, 0.75, 1.0]')
print('这是固定的相对中心点！')
print('\\n预测时如果用绝对中心点:')
print('预测用的中心点是: [-0.0907, -0.0392, 0.0053, 0.0486, 0.1120]')
print('这是绝对值中心点！')
print('\\n结论：训练和预测使用了不同的中心点空间！')