import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """
    深度Q网络 (DQN) 模型。
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    经验回放缓冲区。
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    使用DQN算法的强化学习代理。
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq # 目标网络更新频率

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 初始化目标网络权重
        self.target_net.eval() # 目标网络不进行训练

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step_counter = 0

    def select_action(self, state):
        """
        根据当前状态和epsilon-greedy策略选择动作。
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim) # 探索：随机选择动作
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item() # 利用：选择Q值最高的动作

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储经验到回放缓冲区。
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_epsilon(self):
        """
        更新epsilon值。
        """
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def learn(self):
        """
        从经验回放缓冲区中采样数据，并训练Q网络。
        """
        if len(self.replay_buffer) < self.batch_size:
            return # 缓冲区中的经验不足以进行采样

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device) # (batch_size, 1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # (batch_size, 1)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)    # (batch_size, 1)

        # 计算当前状态的Q值: Q(s, a)
        # policy_net(states_tensor) 输出 (batch_size, action_dim)
        # gather(1, actions_tensor) 提取对应动作的Q值，输出 (batch_size, 1)
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        # 计算下一个状态的最大Q值: max_a' Q_target(s', a')
        # target_net(next_states_tensor) 输出 (batch_size, action_dim)
        # .max(1) 返回 (values, indices)，我们取 values，即最大Q值
        # .unsqueeze(1) 将其形状变为 (batch_size, 1)
        next_q_values = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
        # 如果是终止状态，则下一个状态的Q值为0
        next_q_values[dones_tensor] = 0.0

        # 计算期望的Q值 (TD Target): r + gamma * max_a' Q_target(s', a')
        expected_q_values = rewards_tensor + (self.gamma * next_q_values)

        # 计算损失 (MSE Loss)
        loss = F.mse_loss(current_q_values, expected_q_values.detach()) # detach() 防止梯度流向target_net

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # 可选：梯度裁剪
        self.optimizer.step()

        self.learn_step_counter += 1
        # 定期更新目标网络
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 同样加载到目标网络
        self.policy_net.eval() # 设置为评估模式
        self.target_net.eval()
        print(f"Model loaded from {path}")

if __name__ == '__main__':
    # 示例用法
    state_dim_example = 5 # 假设状态维度为5 (例如：持有量, 余额, 股价, MA5, MA10)
    action_dim_example = 3 # 假设动作维度为3 (买入, 卖出, 持有)

    agent = DQNAgent(state_dim_example, action_dim_example)

    # 模拟一些经验
    for _ in range(200): # 至少要大于 batch_size
        dummy_state = np.random.rand(state_dim_example)
        dummy_action = agent.select_action(dummy_state) # 使用 agent 的方法选择动作
        dummy_reward = random.random()
        dummy_next_state = np.random.rand(state_dim_example)
        dummy_done = random.choice([True, False])
        agent.store_transition(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)

    # 模拟学习过程
    if len(agent.replay_buffer) >= agent.batch_size:
        loss_val = agent.learn()
        print(f"Simulated learning step. Loss: {loss_val}")
        agent.update_epsilon()
        print(f"New epsilon: {agent.epsilon}")
    else:
        print("Not enough samples in replay buffer to learn.")

    # 保存和加载模型示例
    # agent.save_model("dqn_agent_test.pth")
    # agent.load_model("dqn_agent_test.pth")
    # print("Model save and load test complete.")