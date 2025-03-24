import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tr_simple_custom_taxi_env import SimpleTaxiEnv
from IPython.display import clear_output

# 設定 device 為 GPU (若可用)
device = torch.device("cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    """輕量版 DQN"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # 減少神經元數量
        self.fc2 = nn.Linear(64,  128)  # 減少神經元數量    
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 輸出 6 個 Q-values

# 🎯 訓練超參數（輕量版）
GAMMA = 0.99          # 折扣因子
LR = 5e-4             # 學習率（較高，加速收斂）
EPSILON_START = 1.0   # 初始探索率
EPSILON_END = 0.15    # 最小探索率
EPSILON_DECAY = 0.9998 # 探索率衰減
MEMORY_SIZE = 7500    # 記憶庫大小（減少佔用記憶體）
BATCH_SIZE = 32       # 訓練批次大小（減少顯存需求）
TARGET_UPDATE = 10    # 每 10 個 episodes 更新目標網路
EPISODES = 10000       # 訓練回合數（減少訓練時間）
MAX_STEPS_PER_EPISODE = 10000  # 🚨 10000 步後自動結束
REPLAY_START = 500   # 記憶庫最少要有 500 條資料才能訓練

# 初始化環境
env = SimpleTaxiEnv()
state_dim = 16  # 觀察狀態為 16 維向量
action_dim = 6  

# 創建 DQN & 目標網路並移到 GPU
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)  
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START

def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            # 將 state 移到 GPU
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return torch.argmax(policy_net(state_tensor)).item()

def train():
    if len(memory) < REPLAY_START:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # 確保所有 tensor 都建立在 GPU 上
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0].detach()
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 🎮 訓練過程
reward_history = []

for episode in range(EPISODES):
    try:
        grid_size = random.randint(5, 10)  
        state, _ = env.reset(fixed_grid_size=grid_size)

        if episode == 0:
            print(f"✅ 第一回合開始，地圖大小: {grid_size}x{grid_size}")
            env.render_env()  

        total_reward = 0
        step_count = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            store_experience(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                break

        if step_count >= MAX_STEPS_PER_EPISODE:
            done = True
            print(f"⚠️ Episode {episode} 超過 {MAX_STEPS_PER_EPISODE} 步，自動結束")

        train()  
        reward_history.append(total_reward)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"📊 Episode {episode}, Reward: {total_reward}, Grid Size: {grid_size}, Epsilon: {epsilon:.3f}")
            torch.save(policy_net.state_dict(), "dqn_taxi_light64.pth")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        break  

# 儲存 DQN 模型
torch.save(policy_net.state_dict(), "dqn_taxi_light64.pth")
print("DQN 訓練完成，輕量化模型已儲存！")

# 📊 繪製獎勵趨勢
plt.plot(reward_history)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("DQN Lightweight Training Progress")
plt.show()
