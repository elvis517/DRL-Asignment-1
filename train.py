import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
from simple_custom_taxi_env import SimpleTaxiEnv

# 設定超參數
EPISODES = 5000
LEARNING_RATE = 0.001
GAMMA = 0.99  # 折扣因子
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 100  # 更新 target network 頻率

# 環境 & 模型設定
env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)
state_size = len(env.get_state())
action_size = 6

# DQN 架構
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 輸出 Q 值

# 初始化 DQN
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # 初始化 target network
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# 經驗回放記憶體
memory = collections.deque(maxlen=MEMORY_SIZE)

def select_action(state, epsilon):
    """ ϵ-greedy 探索策略 """
    if np.random.rand() < epsilon:
        return np.random.choice(range(action_size))  # 隨機探索
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(policy_net(state_tensor)).item()  # 選擇 Q 值最大動作

# 訓練 DQN
for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, EPSILON)
        next_state, reward, done, _ = env.step(action)

        memory.append((state, action, reward, next_state, done))

        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = target_net(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

            loss = loss_fn(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        total_reward += reward

    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())  # 更新 target network

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 儲存 DQN model
torch.save(policy_net, "dqn_model.pth")
print("DQN 訓練完成，模型已儲存！")
