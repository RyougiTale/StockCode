from long_way.improved_label_generator import ImprovedThreeDimensionalLabelGenerator
from long_way.data_utils import get_all_samples
from long_way import config
# 创建标签生成器
label_generator = ImprovedThreeDimensionalLabelGenerator(
    look_forward_days=config.SOFT_LABEL_CONFIG['LOOK_FORWARD_DAYS'],
    temperature=config.SOFT_LABEL_CONFIG['TEMPERATURE'],
    use_relative_metrics=True
)
# 获取样本数据
all_samples, _ = get_all_samples(['002001'])
if all_samples:
    # 按股票代码分组
    stock_samples_dict = {'002001': []}
    for sample in all_samples:
        if sample.get('stock_code') == '002001':
            stock_samples_dict['002001'].append(sample)
    # 构建分布
    label_generator.fit_stock_distributions(stock_samples_dict)
    # 获取自适应中心点
    centers = label_generator.relative_calculator.get_adaptive_centers('002001', 'total_return')
    print('自适应中心点:', centers)
    print('类型:', type(centers))