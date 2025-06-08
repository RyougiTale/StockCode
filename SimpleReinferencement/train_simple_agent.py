import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 使用相对导入来导入同目录下的模块
from .simple_trading_env import SimpleStockTradingEnv
from .simple_dqn_agent import DQNAgent
# 需要能够从项目根目录找到 Util 包
import sys
# 假设此脚本在 SimpleReinferencement 文件夹下，项目根目录是其父目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from Util.StockDataLoader import StockDataLoader


def train_agent(config):
    """
    训练简化的强化学习交易代理。
    """
    print("Loading data for simple agent...")
    stock_code_for_loader = config['stock_code']
    if stock_code_for_loader.startswith(('sh.', 'sz.')):
        stock_code_for_loader = stock_code_for_loader.split('.')[1]

    data_loader = StockDataLoader(data_dir=config.get('data_dir', './data'))
    df_raw = data_loader.load_stock_data(
        stock_code=stock_code_for_loader,
        start_date=config['start_date'],
        end_date=config['end_date']
    )

    if df_raw is None or df_raw.empty:
        print(f"No data found for {config['stock_code']} between {config['start_date']} and {config['end_date']}.")
        return

    df = df_raw.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
    required_cols = ['Close'] # Simple env至少需要 Close
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Loaded data must contain required columns: {required_cols}")

    if 'date' in df.columns:
        df = df.sort_values(by='date').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True) # 假设数据已按时间排序

    print(f"Data loaded. Shape: {df.shape}")
    if not df.empty:
        print(df.head())

    env = SimpleStockTradingEnv(
        df=df,
        initial_balance=config['initial_balance'],
        transaction_cost_pct=config['transaction_cost_pct']
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Simple Environment initialized. State_dim: {state_dim}, Action_dim: {action_dim}")

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
    print("Simple DQN Agent initialized.")

    episode_rewards = []
    episode_net_worths = []
    losses = []

    print(f"Starting training for {config['episodes']} episodes...")
    for episode in tqdm(range(config['episodes'])):
        state, _ = env.reset()
        done = False
        truncated = False
        current_episode_reward = 0 # 用于累积当前episode的单步奖励

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            current_episode_reward += reward # 累积单步奖励

            loss = agent.learn() # learn 方法现在可能返回 None
            if loss is not None:
                 if not hasattr(episode, 'episode_loss_list'): # 简易处理，实际应在episode开始时初始化
                    episode_loss_list = []
                 else:
                    episode_loss_list = episode.episode_loss_list
                 episode_loss_list.append(loss)


            state = next_state
            if done or truncated:
                break
        
        agent.update_epsilon()
        episode_rewards.append(current_episode_reward) # 记录整个episode的累积奖励
        episode_net_worths.append(info.get('net_worth', env.initial_balance))
        
        # 处理 losses 列表的填充
        if 'episode_loss_list' in locals() and episode_loss_list:
            losses.append(np.mean(episode_loss_list))
            del episode_loss_list # 清理，以便下一轮重新创建
        elif len(agent.replay_buffer) < agent.batch_size : # 如果整个episode都没有学习，也记录0
             losses.append(0) # 或者记录一个特殊值，如 np.nan
        else: # 如果有学习但loss_list为空（不太可能发生，除非learn()一直返回None且buffer够大）
            losses.append(0)


        if (episode + 1) % 10 == 0:
            avg_loss_display = np.mean(losses[-10:]) if losses and not np.all(np.isnan(losses[-10:])) else 0
            tqdm.write(f"Episode {episode + 1}/{config['episodes']}, "
                       f"Net Worth: {info.get('net_worth', 0):.2f}, "
                       f"Total Ep Reward: {current_episode_reward:.2f}, "
                       f"Epsilon: {agent.epsilon:.4f}, "
                       f"Avg Loss (last 10): {avg_loss_display:.4f}")

    print("Training finished.")
    agent.save_model(config['model_save_path'])

    plt.figure(figsize=(12, 9))
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Cumulative Rewards (Sum of step rewards)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(3, 1, 2)
    plt.plot(episode_net_worths)
    plt.title('Episode Net Worth at End')
    plt.xlabel('Episode')
    plt.ylabel('Net Worth')

    plt.subplot(3, 1, 3)
    valid_losses = [l for l in losses if l is not None] # 过滤掉可能的None值
    if valid_losses:
        plt.plot(valid_losses)
    plt.title('Average DQN Loss per Episode (if learning occurred)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(config['results_save_path'])
    print(f"Results plot saved to {config['results_save_path']}")

if __name__ == '__main__':
    # 定义训练配置 (简化版)
    simple_training_config = {
        "stock_code": "600036",
        "start_date": "2022-01-01", # 使用较短时间范围测试
        "end_date": "2022-12-31",
        "initial_balance": 100000,
        "transaction_cost_pct": 0.001,
        "data_dir": "./data", # StockDataLoader的数据目录
        # DQN Agent 参数
        "episodes": 50,  # 减少轮数以便快速测试
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1, # 较高的最终epsilon，鼓励探索
        "epsilon_decay": 0.99, # 较快的衰减
        "buffer_size": 5000, # 较小的buffer
        "batch_size": 32,    # 较小的batch
        "target_update_freq": 5,
        # 保存路径
        "model_save_path": "SimpleReinferencement/simple_dqn_agent.pth",
        "results_save_path": "SimpleReinferencement/simple_training_results.png"
    }

    # 确保 SimpleReinferencement 文件夹存在
    if not os.path.exists("SimpleReinferencement"):
        os.makedirs("SimpleReinferencement")

    train_agent(simple_training_config)