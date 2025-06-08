import gymnasium as gym
import numpy as np
import pandas as pd

class SimpleStockTradingEnv(gym.Env):
    """
    一个非常简化的股票交易环境。
    状态空间: [持股数, 账户余额, 当前收盘价]
    动作空间: 0: 买入, 1: 卖出, 2: 持有
    奖励函数: 单步净值变化
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, df, initial_balance=100000, transaction_cost_pct=0.001):
        super(SimpleStockTradingEnv, self).__init__()

        self.df = df.copy()  # 股票数据 DataFrame, 至少包含 'Close' 列
        if 'Close' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct

        # 动作空间: 0: 买入, 1: 卖出, 2: 持有
        self.action_space = gym.spaces.Discrete(3)

        # 状态空间: [持股数 (0或1), 账户余额, 当前收盘价]
        # low/high 值的类型设为 float32 以避免 gymnasium 警告
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, np.inf, np.inf], dtype=np.float32), # 假设最多持有1股
            dtype=np.float32
        )

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance # 用于计算单步奖励
        self.trade_history = []

    def _next_observation(self):
        """获取下一个观察状态。"""
        obs = np.array([
            self.shares_held,
            self.balance,
            self.df.loc[self.current_step, 'Close']
        ], dtype=np.float32)
        return obs

    def _take_action(self, action):
        """执行给定的动作。"""
        current_price = self.df.loc[self.current_step, 'Close']
        action_type = action

        self.previous_net_worth = self.net_worth # 记录执行动作前的净值

        if action_type == 0:  # 买入
            # 简化逻辑：如果余额够并且当前未持仓，则买入1股
            if self.balance >= current_price * (1 + self.transaction_cost_pct) and self.shares_held == 0:
                self.shares_held = 1
                self.balance -= current_price * (1 + self.transaction_cost_pct)
                self.trade_history.append({'step': self.current_step, 'action': 'buy', 'price': current_price, 'shares': 1})
        elif action_type == 1:  # 卖出
            # 简化逻辑：如果持仓，则卖出
            if self.shares_held > 0:
                self.shares_held = 0
                self.balance += current_price * (1 - self.transaction_cost_pct)
                self.trade_history.append({'step': self.current_step, 'action': 'sell', 'price': current_price, 'shares': 1})

        self.net_worth = self.balance + self.shares_held * current_price

    def step(self, action):
        """环境向前推进一步。"""
        self._take_action(action)
        self.current_step += 1

        # 奖励函数: 单步净值变化
        reward = self.net_worth - self.previous_net_worth

        # 终止条件
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        obs = self._next_observation()
        info = {'net_worth': self.net_worth, 'balance': self.balance, 'shares_held': self.shares_held}

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.current_step = 0
        self.trade_history = []
        return self._next_observation(), {}

    def render(self, mode='human'):
        """渲染环境状态 (可选)。"""
        if mode == 'human':
            profit = self.net_worth - self.initial_balance
            print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}, Shares: {self.shares_held}, Balance: {self.balance:.2f}')
        else:
            super(SimpleStockTradingEnv, self).render(mode=mode)

    def close(self):
        pass

if __name__ == '__main__':
    # 创建一个示例 DataFrame
    data = {'Close': np.random.rand(100) * 100 + 10}
    sample_df = pd.DataFrame(data)

    env = SimpleStockTradingEnv(df=sample_df)
    obs, info = env.reset()

    print("Initial Observation:", obs)
    print("Initial Info:", info)

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            print(f"Episode finished after {i+1} steps.")
            break
    env.close()