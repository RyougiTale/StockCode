分位数的含义

  **分位数(Quantile)**是统计学概念，表示数据中某个值的相对位置：

  10%分位数 = 有10%的数据小于等于这个值，90%的数据大于这个值
  25%分位数 = 有25%的数据小于等于这个值，75%的数据大于这个值
  50%分位数 = 中位数，一半数据在上面，一半在下面
  75%分位数 = 有75%的数据小于等于这个值，25%的数据大于这个值
  90%分位数 = 有90%的数据小于等于这个值，10%的数据大于这个值

  为什么是5个分位数值？

  因为我们的分类系统有5个类别，需要5个"代表性中心点"：

  # 在improved_label_generator.py:131中
  "quantiles": np.quantile(returns, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()

  # 对应5个类别：
  # Class 0 (Very Poor) ← 10%分位数  (最差的10%水平)
  # Class 1 (Poor)      ← 25%分位数  (较差的25%水平)
  # Class 2 (Average)   ← 50%分位数  (中等的50%水平/中位数)
  # Class 3 (Good)      ← 75%分位数  (较好的75%水平)
  # Class 4 (Excellent) ← 90%分位数  (最好的90%水平)

  训练数据统计过程

  让我用一个具体例子说明统计过程：

  步骤1：收集历史数据 (improved_label_generator.py:104-127)

  # 假设股票"600519"在训练集中有1000个样本
  stock_samples = [
      {'future_prices': [100, 102, 105, 103, 108]},  # 样本1
      {'future_prices': [100, 98, 95, 97, 102]},     # 样本2
      {'future_prices': [100, 103, 106, 110, 115]},  # 样本3
      # ... 共1000个样本
  ]

  步骤2：计算每个样本的指标 (improved_label_generator.py:317-361)

  # 对每个样本计算三个指标
  returns = []
  sharpes = []
  drawdowns = []

  for sample in stock_samples:
      prices = sample['future_prices']
      # 总回报率
      total_return = (prices[-1] / prices[0]) - 1
      returns.append(total_return)

      # 夏普比率
      daily_rets = np.diff(prices) / prices[:-1]
      sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
      sharpes.append(sharpe)

      # 最大回撤
      cummax = np.maximum.accumulate(prices)
      drawdown = ((prices - cummax) / cummax).min()
      drawdowns.append(drawdown)

  # 结果示例：
  returns = [0.08, -0.02, 0.15, 0.05, -0.12, 0.03, ...]  # 1000个值
  sharpes = [2.1, -0.8, 3.5, 1.2, -2.5, 0.9, ...]       # 1000个值
  drawdowns = [-0.05, -0.15, -0.03, -0.08, -0.25, -0.06, ...]  # 1000个值

  步骤3：计算分位数 (improved_label_generator.py:131)

  # 对return数据排序并计算分位数
  sorted_returns = [-0.25, -0.20, -0.15, ..., 0.03, 0.05, 0.08, 0.15, 0.20, 0.25]

  quantiles = np.quantile(returns, [0.1, 0.25, 0.5, 0.75, 0.9])
  # 结果示例：
  # 10%分位数: -0.097  (意思是90%的样本回报率都比-9.7%要好)
  # 25%分位数: -0.046  (意思是75%的样本回报率都比-4.6%要好)
  # 50%分位数: +0.004  (中位数，一半样本在上面一半在下面)
  # 75%分位数: +0.057  (意思是75%的样本回报率都在+5.7%以下)
  # 90%分位数: +0.119  (意思是90%的样本回报率都在+11.9%以下)

  步骤4：构建股票分布字典

  self.stock_distributions["600519"] = {
      "total_return": {
          "values": returns,                                    # 原始1000个值
          "mean": 0.025,                                       # 平均值2.5%
          "std": 0.089,                                        # 标准差8.9%
          "quantiles": [-0.097, -0.046, 0.004, 0.057, 0.119], # 5个分位数
      },
      "sharpe_ratio": {
          "values": sharpes,
          "mean": 0.75,
          "std": 1.85,
          "quantiles": [-3.84, -1.91, 0.44, 2.58, 4.57],
      },
      "max_drawdown": {
          "values": drawdowns,
          "mean": -0.078,
          "std": 0.045,
          "quantiles": [-0.159, -0.101, -0.066, -0.043, -0.028],
      }
  }

  实际意义

  这5个分位数代表了该股票在训练期间的典型表现水平：

  - 10%分位数(-9.7%)：这只股票表现很差的时候通常是这个水平
  - 25%分位数(-4.6%)：表现较差时的典型水平
  - 50%分位数(+0.4%)：该股票的"正常"表现水平
  - 75%分位数(+5.7%)：表现较好时的典型水平
  - 90%分位数(+11.9%)：该股票表现很好时通常是这个水平

  这样，当新样本的回报率是8.95%时，系统就能判断这个表现在该股票的历史分布中属于"Good到Excellent"之间的水平，从而生成合适的软标签概率分布。
