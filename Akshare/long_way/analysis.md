下面是对 long_way/draw_3d_long_term.py 中“3D tag”生成与使用的核心逻辑解析（训练
时如何生、推理时如何用）。

**标签定义（训练用的 3D tag）**
- 维度含义: 3D = `total_return`（未来N日总收益）、`sharpe_ratio`（未来N日夏普）
、`max_drawdown`（未来N日最大回撤）。
- 软标签生成: `ImprovedThreeDimensionalLabelGenerator.create_soft_label_3d` 为每
个维度生成5类的概率分布（soft label），公式为
  - `probabilities = softmax(-|centers - value| / T)`（T 为温度，`config.SOFT_LA
BEL_CONFIG["TEMPERATURE"]`）。
- 中心点（centers）:
  - 绝对模式（备用）: 收益 `[-0.15,-0.05,0.02,0.08,0.20]`；夏普 `[-1.0,0.0,0.5,1
.0,2.0]`；回撤 `[-0.25,-0.15,-0.08,-0.04,-0.01]`。
  - 相对模式（默认）: 先把原始值映射到相对位置 [0,1]（按该股历史分位区间分段线性
插值），再对相对中心 `[0.0,0.25,0.5,0.75,1.0]` 做 softmax。

**自适应中心（相对标签的依据）**
- 由 `ImprovedRelativeMetricsCalculator.fit_stock_distributions` 对每只股票统计
训练样本的分布（收益/夏普/回撤的分位点）。
- `get_adaptive_centers(stock_code, metric)` 返回混合中心:
  - 股票分位点×70% + 基线中心×30%；若无该股分布则用全局分位点×50% + 基线×50%；再
不行退回基线。

**实际指标计算（用于对比）**
- `calculate_actual_3d_metrics(df, look_forward_days)` 按未来 N 日价格序列计算：
  - 总收益: `(P_T/P_0)-1`
  - 夏普: `mean(daily_returns)/std(daily_returns) * sqrt(len)`（近似年化）
  - 最大回撤: `(price - cummax)/cummax` 的最小值

**推理阶段如何从模型输出得到数值预测**
- 模型输出: `output['return']`, `output['sharpe']`, `output['drawdown']` 为各维
度的 logits。
- 概率: `torch.softmax(..., dim=1)` 得到每维5类概率。
- 数值映射（关键）:
  - 期望值（全分布）: `E = Σ p_i * center_i` → `pred_return_full`, `pred_sharpe_
full`, `pred_drawdown_full`。
  - Top-1: 取 `argmax(p)` 对应中心 → `pred_*_top1`。
  - Top-3: 取概率最高3类，重新归一化后加权中心 → `pred_*_top3`。
- 中心来源: 优先用上面的“自适应中心”；无数据时退回固定基线中心。

简言之：3D tag 是对未来收益/夏普/回撤的“软分类标签”，用距离中心点的温度 softmax
得到的5类概率分布；推理时模型也输出这三组概率，再用相同的中心将其还原为连续数值
（全分布期望、Top-3、Top-1），并与实际值做对比与可视化。