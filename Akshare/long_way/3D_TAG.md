  3D标签生成完整流程实例

  示例数据

  以刚才的Sample #1为例：
  - 股票代码: 假设是 "600519"
  - 原始指标:
    - Total Return: 8.95%
    - Sharpe Ratio: 3.298
    - Max Drawdown: -6.99%

--- Sample #1 (Index: 189519) ---
Raw Metrics:
  Total Return: 8.95%
  Sharpe Ratio: 3.298
  Max Drawdown: -6.99%

Return Label Distribution:
  Class 0 (Very Poor, ≈-9.7%): Probability = 0.0003
  Class 1 (Poor, ≈-4.6%): Probability = 0.0030
  Class 2 (Average, ≈+0.4%): Probability = 0.0371
  Class 3 (Good, ≈+5.7%): Probability = 0.4522
  Class 4 (Excellent, ≈+11.9%): Probability = 0.5074 <-- Peak

Sharpe Label Distribution:
  Class 0 (Very Poor, ≈-3.84): Probability = 0.0003
  Class 1 (Poor, ≈-1.91): Probability = 0.0042
  Class 2 (Average, ≈+0.44): Probability = 0.0516
  Class 3 (Good, ≈+2.58): Probability = 0.6290 <-- Peak
  Class 4 (Excellent, ≈+4.57): Probability = 0.3148

Drawdown Label Distribution:
  Class 0 (Terrible, ≈-15.9%): Probability = 0.0096
  Class 1 (Bad, ≈-10.1%): Probability = 0.1169
  Class 2 (Average, ≈-6.6%): Probability = 0.8023 <-- Peak
  Class 3 (Good, ≈-4.3%): Probability = 0.0659
  Class 4 (Excellent, ≈-2.8%): Probability = 0.0054
  ---
  第1步：计算原始指标 (improved_label_generator.py:183-224)

  从未来20天的价格序列计算三个指标：

  # 假设未来20天价格: [100, 102, 105, 103, ..., 108.95]
  price_series = pd.Series([100.0, 102.0, 105.0, 103.0, ..., 108.95])

  # 1. 总回报率
  total_return = (108.95 / 100.0) - 1 = 0.0895 = 8.95%

  # 2. 夏普比率（年化）
  daily_returns = [0.02, 0.029, -0.019, ..., 0.015]
  sharpe_ratio = daily_returns.mean() / daily_returns.std() * sqrt(252) = 3.298

  # 3. 最大回撤
  max_drawdown = min((price - cummax(price)) / cummax(price)) = -6.99%

  ---
  第2步：获取自适应中心点 (improved_label_generator.py:226-259)

  系统为股票"600519"计算自适应中心点：

  # 基线中心点（市场通用经验值）
  baseline_centers = {
      "total_return": [-0.10, -0.03, 0.00, 0.03, 0.10],     # [-10%, -3%, 0%, +3%, +10%]
      "sharpe_ratio": [-2.0, -0.5, 0.5, 1.5, 3.0],         # 年化夏普典型范围
      "max_drawdown": [-0.20, -0.10, -0.06, -0.03, -0.01]  # [-20%, -10%, -6%, -3%, -1%]
  }

  # 股票600519的历史分布分位点（从训练数据统计得出）
  stock_quantiles = {
      "total_return": [-0.097, -0.046, 0.004, 0.057, 0.119],    # 该股票历史10%-90%分位数
      "sharpe_ratio": [-3.84, -1.91, 0.44, 2.58, 4.57],
      "max_drawdown": [-0.159, -0.101, -0.066, -0.043, -0.028]
  }

  # 自适应融合：30% baseline + 70% stock_quantiles
  adaptive_centers = 0.3 * baseline + 0.7 * stock_quantiles
  # 结果：
  # total_return: [-0.097, -0.046, 0.004, 0.057, 0.119] = [-9.7%, -4.6%, +0.4%, +5.7%, +11.9%]
  # sharpe_ratio: [-3.84, -1.91, 0.44, 2.58, 4.57]
  # max_drawdown: [-0.159, -0.101, -0.066, -0.043, -0.028] = [-15.9%, -10.1%, -6.6%, -4.3%, -2.8%]

  ---
  第3步：相对化映射 (improved_label_generator.py:261-301)

  把绝对指标值映射到[0,1]相对位置：

  # 以total_return为例：8.95% 需要映射到 [0,1]
  value = 0.0895  # 8.95%
  centers = [-0.097, -0.046, 0.004, 0.057, 0.119]  # 自适应中心点

  # 查找区间：8.95% 位于 centers[2]=0.4% 和 centers[3]=5.7% 之间
  # 区间索引：i=2, left=0.004, right=0.057

  # 线性插值计算相对位置
  progress = (0.0895 - 0.004) / (0.057 - 0.004) = 0.0855 / 0.053 = 1.613
  # 但progress>1，说明超出右边界，实际位于centers[3]和centers[4]之间

  # 重新定位：8.95%位于centers[3]=5.7%和centers[4]=11.9%之间
  # i=3, left=0.057, right=0.119
  progress = (0.0895 - 0.057) / (0.119 - 0.057) = 0.0325 / 0.062 = 0.524

  # 相对位置
  relative_pos = (3 + 0.524) / (5 - 1) = 3.524 / 4 = 0.881

  # 同理计算其他指标：
  # sharpe_ratio: 3.298 → relative_pos ≈ 0.75
  # max_drawdown: -6.99% → relative_pos ≈ 0.5

  ---
  第4步：生成软标签 (improved_label_generator.py:526-540)

  使用温度缩放的距离softmax：

  # 相对化中心点（固定）
  relative_centers = [0.0, 0.25, 0.5, 0.75, 1.0]

  # 以total_return为例：relative_value = 0.881
  value = 0.881
  centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
  temperature = 0.1

  # 计算到各中心点的距离
  distances = |centers - 0.881| = [0.881, 0.631, 0.381, 0.131, 0.119]

  # 温度缩放logits（距离越小，logit越大）
  logits = -distances / temperature = [-8.81, -6.31, -3.81, -1.31, -1.19]

  # Softmax归一化
  probabilities = exp(logits) / sum(exp(logits))
  # = [0.0003, 0.0030, 0.0371, 0.4522, 0.5074]

  ---
  第5步：最终3D标签

  final_labels_3d = {
      "return": [0.0003, 0.0030, 0.0371, 0.4522, 0.5074],    # Peak在Class 4 (Excellent)
      "sharpe": [0.0003, 0.0042, 0.0516, 0.6290, 0.3148],    # Peak在Class 3 (Good)
      "drawdown": [0.0096, 0.1169, 0.8023, 0.0659, 0.0054]   # Peak在Class 2 (Average)
  }

  ---
  核心思想总结

  1. 自适应性: 每只股票有自己的中心点，基于历史数据统计
  2. 相对化: 绝对指标映射到[0,1]，实现跨股票可比性
  3. 软标签: 温度参数控制分布尖锐度，0.1产生合理的概率分布
  4. 多维度: 同时考虑回报、风险、回撤三个维度的综合表现

  这样生成的标签既保留了连续性信息，又便于深度学习模型进行分类训练，同时具有很好的可解释性。
