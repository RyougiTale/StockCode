import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm # 用于显示进度条

from .trading_env import StockTradingEnv # 使用相对导入
from .dqn_agent import DQNAgent       # 使用相对导入
from Util.StockDataLoader import StockDataLoader

def train_agent(config):
    """
    训练强化学习交易代理。

    Args:
        config (dict): 包含训练参数的字典。
            - stock_code (str): 股票代码，例如 'sh_000300_d'
            - start_date (str): 开始日期 'YYYY-MM-DD'
            - end_date (str): 结束日期 'YYYY-MM-DD'
            - initial_balance (float): 初始资金
            - lookback_window (int): 环境的回看窗口
            - transaction_cost_pct (float): 交易成本百分比
            - episodes (int): 训练的总轮数
            - lr (float): DQN代理的学习率
            - gamma (float): DQN代理的折扣因子
            - epsilon_start (float): DQN代理的初始epsilon
            - epsilon_end (float): DQN代理的最终epsilon
            - epsilon_decay (float): DQN代理的epsilon衰减率
            - buffer_size (int): DQN代理的回放缓冲区大小
            - batch_size (int): DQN代理的批处理大小
            - target_update_freq (int): DQN代理的目标网络更新频率
            - model_save_path (str): 模型保存路径
            - results_save_path (str): 结果图表保存路径
    """
    # 1. 加载数据
    print("Loading data...")
    # 使用 StockDataLoader 加载数据
    # StockDataLoader 的 stock_code 参数不需要 'sh.' 或 'sz.' 前缀
    # 它接受的 data_dir 默认为 './data'，这与您项目结构中的 'data/' 目录匹配
    stock_code_for_loader = config['stock_code']
    if stock_code_for_loader.startswith(('sh.', 'sz.')): # 移除可能的前缀
        stock_code_for_loader = stock_code_for_loader.split('.')[1]

    data_loader = StockDataLoader(data_dir='./data') # 指定数据目录
    df_raw = data_loader.load_stock_data(
        stock_code=stock_code_for_loader,
        start_date=config['start_date'],
        end_date=config['end_date']
    )

    if df_raw is None or df_raw.empty:
        print(f"No data found for {config['stock_code']} between {config['start_date']} and {config['end_date']} using StockDataLoader.")
        return

    # 将列名转换为大写以匹配 StockTradingEnv 的期望
    # StockDataLoader 返回的列名是小写： 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg'
    # StockTradingEnv 期望 'Close', 'MA5', 'MA10' (MA是内部计算的)
    df = df_raw.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'amount': 'Amount',
        'turn': 'Turn',
        'pctChg': 'PctChg'
    })

    # StockTradingEnv 内部会基于行号（current_step）来索引 df.loc[self.current_step, ...]
    # 因此，我们需要确保 df 是一个标准的 DataFrame，其默认整数索引对应时间顺序
    # StockDataLoader 返回的 df 已经是按 'date' 排序的
    # 为了让 StockTradingEnv 能按 current_step 访问，我们重置索引，使 'date' 成为普通列
    # （尽管 StockTradingEnv 内部目前没有直接使用 'date' 列进行索引，但保持数据整洁）
    if 'date' in df.columns:
        df = df.sort_values(by='date').reset_index(drop=True)
    else:
        print("Warning: 'date' column not found in loaded data. Ensure data is sorted chronologically.")
        df = df.reset_index(drop=True)


    print(f"Data loaded and processed. Shape: {df.shape}")
    if not df.empty:
        print(df.head())

    # 2. 初始化环境
    env = StockTradingEnv(
        df=df,
        initial_balance=config['initial_balance'],
        lookback_window=config['lookback_window'],
        rsi_window=config['rsi_window'],
        macd_fast=config['macd_fast'],
        macd_slow=config['macd_slow'],
        macd_signal=config['macd_signal'],
        transaction_cost_pct=config['transaction_cost_pct']
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Environment initialized. State_dim: {state_dim}, Action_dim: {action_dim}")

    # 3. 初始化DQN代理
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config['lr'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq']
    )
    print("DQN Agent initialized.")

    # 4. 训练循环
    episode_rewards = []
    episode_net_worths = []
    losses = []

    print(f"Starting training for {config['episodes']} episodes...")
    for episode in tqdm(range(config['episodes'])):
        state, _ = env.reset()
        done = False
        truncated = False # Gymnasium 的 step 返回5个值
        total_reward = 0
        episode_loss = []

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done or truncated) # 存储经验

            if len(agent.replay_buffer) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)

            state = next_state
            total_reward += reward # 这里的 reward 是每一步的净值变化或你定义的奖励

            if done or truncated:
                break

        agent.update_epsilon() # 每轮结束后更新epsilon

        episode_rewards.append(total_reward) # 这里记录的是累积奖励
        episode_net_worths.append(info.get('net_worth', env.initial_balance)) # 记录每轮结束时的净值
        if episode_loss:
            losses.append(np.mean(episode_loss))
        else:
            losses.append(0) # 如果没有学习步骤，则损失为0

        if (episode + 1) % 10 == 0: # 每10轮打印一次信息
            tqdm.write(f"Episode {episode + 1}/{config['episodes']}, "
                       f"Net Worth: {info.get('net_worth', 0):.2f}, "
                       f"Total Reward: {total_reward:.2f}, "
                       f"Epsilon: {agent.epsilon:.4f}, "
                       f"Avg Loss: {np.mean(losses[-10:]) if losses else 0:.4f}")

    print("Training finished.")

    # 5. 保存模型
    agent.save_model(config['model_save_path'])

    # 6. 可视化结果
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(3, 1, 2)
    plt.plot(episode_net_worths)
    plt.title('Episode Net Worth')
    plt.xlabel('Episode')
    plt.ylabel('Net Worth')

    plt.subplot(3, 1, 3)
    plt.plot(losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(config['results_save_path'])
    print(f"Results plot saved to {config['results_save_path']}")
    # plt.show() # 如果在本地运行可以取消注释

if __name__ == '__main__':
    # 定义训练配置
    training_config = {
        "stock_code": "600036", # 示例：使用您 data 目录下的一个股票代码 (不带sh/sz前缀)
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "initial_balance": 100000,
        "lookback_window": 10, # MA窗口，可以根据需要调整
        "rsi_window": 14,      # RSI 窗口
        "macd_fast": 12,       # MACD 快线周期
        "macd_slow": 26,       # MACD 慢线周期
        "macd_signal": 9,      # MACD 信号线周期
        "transaction_cost_pct": 0.001,
        "episodes": 100, # 增加轮数以获得更好的训练效果
        "lr": 0.0005, # 学习率
        "gamma": 0.99, # 折扣因子
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 50000, # 经验回放池大小
        "batch_size": 64, # 批处理大小
        "target_update_freq": 20, # 目标网络更新频率
        "model_save_path": "Reinforencement/trained_dqn_agent.pth",
        "results_save_path": "Reinforencement/training_results.png"
    }

    # 确保 Reinforencement 文件夹存在，用于保存模型和结果
    import os
    if not os.path.exists("Reinforencement"):
        os.makedirs("Reinforencement")

    train_agent(training_config)