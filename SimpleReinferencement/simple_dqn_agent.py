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
        # 简单的网络结构，可以根据需要调整
        self.fc1 = nn.Linear(input_dim, 64) # 减少了神经元数量以求简单
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

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
        # 确保所有推入的数据都是Python原生类型或Numpy数组，以便后续转换为Tensor
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # 将action, reward, done转换为numpy数组，方便后续处理
        return np.array(state, dtype=np.float32), \
               np.array(action, dtype=np.int64), \
               np.array(reward, dtype=np.float32), \
               np.array(next_state, dtype=np.float32), \
               np.array(done, dtype=np.bool_)


    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    使用DQN算法的强化学习代理。
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None # 返回 None 表示没有学习

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        # dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device) # PyTorch 1.9+
        dones_tensor = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)


        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)
        next_q_values_target = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
        next_q_values_target[dones_tensor] = 0.0

        expected_q_values = rewards_tensor + (self.gamma * next_q_values_target)

        loss = F.mse_loss(current_q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()
        print(f"Model loaded from {path}")

if __name__ == '__main__':
    state_dim_example = 3 # 对应 SimpleStockTradingEnv 的状态维度
    action_dim_example = 3
    agent = DQNAgent(state_dim_example, action_dim_example, buffer_size=200, batch_size=32) # 减小buffer和batch用于快速测试

    # 模拟一些经验
    for _ in range(100): # 至少要大于 batch_size
        dummy_state = np.random.rand(state_dim_example).astype(np.float32)
        dummy_action = agent.select_action(dummy_state)
        dummy_reward = float(random.random())
        dummy_next_state = np.random.rand(state_dim_example).astype(np.float32)
        dummy_done = bool(random.choice([True, False]))
        agent.store_transition(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)

    if len(agent.replay_buffer) >= agent.batch_size:
        loss_val = agent.learn()
        print(f"Simulated learning step. Loss: {loss_val if loss_val is not None else 'N/A'}")
        agent.update_epsilon()
        print(f"New epsilon: {agent.epsilon}")
    else:
        print(f"Not enough samples ({len(agent.replay_buffer)}) in replay buffer to learn (requires {agent.batch_size}).")

    # agent.save_model("simple_dqn_agent_test.pth")
    # agent.load_model("simple_dqn_agent_test.pth")