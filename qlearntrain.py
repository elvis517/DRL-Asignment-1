import random
import numpy as np
from tr_simple_custom_taxi_env import SimpleTaxiEnv
import matplotlib.pyplot as plt
import pickle

# 超參數設定
ALPHA = 0.01             # 學習率
GAMMA = 0.99            # 折扣因子
EPSILON_START = 1.0     # 初始探索率
EPSILON_MIN = 0.1      # 最小探索率
EPSILON_DECAY = 0.9997  # 探索率衰減
EPISODES = 10000        # 訓練回合數
MAX_STEPS = 10000       # 每個回合最大步數

# 初始化環境
env = SimpleTaxiEnv()

# 建立 Q-table：以 dictionary 形式儲存，key 為 state（轉成字串），value 為長度為 6 的 numpy 陣列
q_table = {}

def get_q(state):
    """取得 state 的 Q 值，若不存在則初始化為全 0 陣列"""
    key = str(state)
    if key not in q_table:
        q_table[key] = np.zeros(6)
    return q_table[key]

def choose_action(state, epsilon):
    """ε-greedy 策略：以 epsilon 機率採取隨機動作，否則選擇 Q 值最高的動作"""
    # 確保該 state 已初始化
    _ = get_q(state)
    if random.random() < epsilon:
        return random.randint(0, 5)
    else:
        return int(np.argmax(get_q(state)))

reward_history = []
epsilon = EPSILON_START

for episode in range(EPISODES):
    grid_size = random.randint(5, 10)
    state, _ = env.reset(fixed_grid_size=grid_size)
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < MAX_STEPS:
        # 確保目前 state 初始化
        _ = get_q(state)
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        # 初始化下一個 state 的 Q 值
        _ = get_q(next_state)

        # Q-learning 更新公式：
        # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',:)) - Q(s,a)]
        state_key = str(state)
        q_table[state_key][action] += ALPHA * (reward + GAMMA * np.max(get_q(next_state)) - q_table[state_key][action])

        state = next_state
        total_reward += reward
        steps += 1

    # 衰減 epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    reward_history.append(total_reward)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Grid Size: {grid_size}, Epsilon: {epsilon:.3f}")

# 儲存 Q-table 到檔案中
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)
print("Q-table 已儲存至 q_table.pkl")

# 繪製獎勵趨勢圖
plt.plot(reward_history)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-learning Training Progress")
plt.show()
