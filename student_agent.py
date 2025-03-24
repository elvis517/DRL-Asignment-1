# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from tr_simple_custom_taxi_env import SimpleTaxiEnv as env
# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.


#     return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
#     # You can submit this random agent to evaluate the performance of a purely random strategy.

# import torch
# import numpy as np
# import random

# 載入 DQN 模型
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# 請確認模型檔案路徑正確
MODEL_PATH = "dqn_taxi_light14.pth"
state_dim = 14   # 根據你的 get_state() 回傳的維度
action_dim = 6   # 可用動作數量

class DQNAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 建立與訓練時相同的模型架構
        self.model = self.build_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.epsilon = 0.005  # 測試時較低的探索率

    def build_model(self):
        """建立與訓練時相同的 DQN 模型"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

    def select_action(self, obs):
        """選擇行動（使用 epsilon-greedy 策略）"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 隨機探索
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                return torch.argmax(self.model(obs_tensor)).item()  # 選擇 Q 值最高的動作

# 假設你已經有環境物件 env，且 env 有你定義的 get_state() 方法
# 例如：
# from your_custom_taxi_env import SimpleTaxiEnv
# env = SimpleTaxiEnv()
# env.reset()  之類的初始化動作

# 建立 agent 實例
agent = DQNAgent(state_dim, action_dim, MODEL_PATH)

def get_action():
    """
    從 env 中取得當前狀態 (使用你定義的 get_state() 方法)，
    並利用 agent 選擇最佳動作。
    """
    try:
        # 直接透過你自定義的 get_state() 取得狀態
        state = env.get_state()
        # 將狀態轉換成 NumPy 陣列，確保格式正確
        state = np.array(state, dtype=np.float32)
        # 檢查狀態維度是否正確 (應為 14)
        if state.shape[0] != state_dim:
            print(f"⚠️ state 維度錯誤: 期望 {state_dim}, 但收到 {state.shape[0]}，使用全零向量補位")
            state = np.zeros(state_dim, dtype=np.float32)
        
        # 使用 agent 選擇動作
        action = agent.select_action(state)
        return action

    except Exception as e:
        print(f"⚠️ 選擇動作時發生錯誤: {e}")
        # 發生錯誤時返回隨機動作
        return random.choice([0, 1, 2, 3, 4, 5])
    
