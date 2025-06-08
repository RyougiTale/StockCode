import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    一个简化的股票交易环境，用于强化学习。

    状态空间 (State Space):
        - 当前持有的股票数量 (离散值，例如 0 或 1)
        - 当前账户余额 (连续值)
        - 当前股票价格 (连续值)
        - 其他技术指标 (例如，移动平均线等，连续值)

    动作空间 (Action Space):
        - 买入 (Buy): 买入1单位股票
        - 卖出 (Sell): 卖出1单位股票
        - 持有 (Hold): 不进行任何操作

    奖励函数 (Reward Function):
        - 每次交易的利润或损失
        - 持有期间的未实现收益
        - 避免交易过于频繁的惩罚 (交易成本)
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, df, initial_balance=100000, lookback_window=20, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9, transaction_cost_pct=0.001):
        super(StockTradingEnv, self).__init__()

        self.df = df.copy()  # 股票数据 DataFrame (例如，包含 'Open', 'High', 'Low', 'Close', 'Volume' 等)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window  # 用于计算技术指标的回看窗口 (MA等)
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.transaction_cost_pct = transaction_cost_pct # 交易成本百分比

        # 动作空间: 0: 买入, 1: 卖出, 2: 持有
        self.action_space = gym.spaces.Discrete(3)

        # 先准备数据，计算指标，这样可以确定 observation_space 的维度
        self._prepare_data() # _prepare_data 会添加 MA, RSI, MACD 列

        # 状态空间: [持有股票数量, 账户余额, 当前股价, MA5, MA10, RSI, MACD, MACD_signal, MACD_hist]
        # MA5, MA10 (2个) + RSI (1个) + MACD, Signal, Hist (3个) = 6个技术指标
        # 总共 3 (基本状态) + 2 (MA) + 1 (RSI) + 3 (MACD) = 9 个特征
        # 注意：这里的 low 和 high 值需要根据实际情况调整
        # RSI is 0-100. MACD can be negative or positive.
        num_tech_indicators = 2 + 1 + 3 # MA(2) + RSI(1) + MACD(3)
        low_values = [0, 0, 0] + [0]*2 + [0] + [-np.inf]*3 # 持有量, 余额, 股价, MA5, MA10, RSI, MACD, Signal, Hist
        high_values = [1, np.inf, np.inf] + [np.inf]*2 + [100] + [np.inf]*3

        self.observation_space = gym.spaces.Box(
            low=np.array(low_values, dtype=np.float32),
            high=np.array(high_values, dtype=np.float32),
            dtype=np.float32
        )
        # print(f"Observation space shape: {self.observation_space.shape}")


        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trade_history = [] # 记录交易历史

        # _prepare_data() 已经在 __init__ 中被调用

    def _calculate_rsi(self, series, window):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series, fast_period, slow_period, signal_period):
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _prepare_data(self):
        """
        准备数据，例如计算技术指标。
        这里只是一个简单的示例，你需要根据你的策略添加更多指标。
        """
        if 'Close' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        # 示例：计算移动平均线
        self.df['MA5'] = self.df['Close'].rolling(window=5).mean() # 使用 self.lookback_window 或固定值
        self.df['MA10'] = self.df['Close'].rolling(window=10).mean()

        # 计算RSI
        self.df['RSI'] = self._calculate_rsi(self.df['Close'], window=self.rsi_window)

        # 计算MACD
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = self._calculate_macd(
            self.df['Close'],
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal
        )

        # 填充初始的 NaN 值 (所有计算出的指标列)
        self.df.bfill(inplace=True) # 用后面的有效值填充
        self.df.ffill(inplace=True) # 用前面的有效值填充，以防整个序列都是NaN

    def _next_observation(self):
        """
        获取下一个观察状态。
        """
        frame = np.array([
            self.shares_held,
            self.balance,
            self.df.loc[self.current_step, 'Close'],
            self.df.loc[self.current_step, 'MA5'],
            self.df.loc[self.current_step, 'MA10'],
            self.df.loc[self.current_step, 'RSI'],
            self.df.loc[self.current_step, 'MACD'],
            self.df.loc[self.current_step, 'MACD_signal'],
            self.df.loc[self.current_step, 'MACD_hist'],
        ])
        return frame.astype(np.float32)

    def _take_action(self, action):
        """
        执行给定的动作。
        """
        current_price = self.df.loc[self.current_step, 'Close']
        action_type = action # 0: 买入, 1: 卖出, 2: 持有

        if action_type == 0: # 买入
            if self.balance > current_price * (1 + self.transaction_cost_pct) and self.shares_held == 0: # 假设一次只买卖1股，且没有持仓时才能买
                self.shares_held = 1
                self.balance -= current_price * (1 + self.transaction_cost_pct)
                self.trade_history.append({'step': self.current_step, 'action': 'buy', 'price': current_price, 'shares': 1})
        elif action_type == 1: # 卖出
            if self.shares_held > 0:
                self.shares_held = 0
                self.balance += current_price * (1 - self.transaction_cost_pct)
                self.trade_history.append({'step': self.current_step, 'action': 'sell', 'price': current_price, 'shares': 1})
        # 如果是持有，则不进行任何操作

        self.net_worth = self.balance + self.shares_held * current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        环境向前推进一步。
        """
        self._take_action(action)
        self.current_step += 1

        # 奖励函数：这里使用简单的净值变化作为奖励
        # 你可以设计更复杂的奖励函数
        reward = self.net_worth - self.initial_balance # 相对于初始资金的收益
        # 或者 reward = self.net_worth - (self.balance + self.shares_held * self.df.loc[self.current_step -1, 'Close']) # 单步收益

        # 终止条件
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        obs = self._next_observation()
        info = {'net_worth': self.net_worth, 'balance': self.balance, 'shares_held': self.shares_held}

        return obs, reward, done, False, info # gymnasium 返回5个值

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        """
        super().reset(seed=seed) # 处理随机种子
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = 0 # 或者是一个随机的起始点，如果你的数据足够长
        self.trade_history = []
        return self._next_observation(), {} # gymnasium 返回 observation 和 info

    def render(self, mode='human'):
        """
        渲染环境状态 (可选)。
        """
        if mode == 'human':
            profit = self.net_worth - self.initial_balance
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Shares held: {self.shares_held}')
            print(f'Net worth: {self.net_worth} (Max: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print(f'Trades: {len(self.trade_history)}')
        else:
            super(StockTradingEnv, self).render(mode=mode) # 调用父类的render方法

    def close(self):
        """
        关闭环境，释放资源 (可选)。
        """
        pass

if __name__ == '__main__':
    # 创建一个示例 DataFrame
    data = {
        'Open': np.random.rand(100) * 100 + 10,
        'High': np.random.rand(100) * 100 + 15,
        'Low': np.random.rand(100) * 100 + 5,
        'Close': np.random.rand(100) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 100)
    }
    sample_df = pd.DataFrame(data)

    # 初始化环境
    env = StockTradingEnv(df=sample_df, lookback_window=5, rsi_window=14) # 调整 lookback_window 以匹配 MA 计算
    obs, info = env.reset()

    print("Initial Observation shape:", obs.shape)
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    # 测试环境交互
    for _ in range(10):
        action = env.action_space.sample()  # 随机选择一个动作
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            print("Episode finished.")
            break
    env.close()