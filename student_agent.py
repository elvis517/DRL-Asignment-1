# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
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
MODEL_PATH = "dqn_taxi_light14.pth"

class DQNAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 建立與訓練時相同的模型架構
        self.model = self.build_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.epsilon = 0.005  # 測試時探索率較低

    def build_model(self):
        """建立與訓練時相同的 DQN 模型"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_dim)
        )

    def select_action(self, obs):
        """選擇行動（使用 epsilon-greedy 策略）"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 隨機探索
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                return torch.argmax(self.model(obs_tensor)).item()  # 選擇 Q 值最高的行動

# 初始化代理（根據環境狀態空間大小調整 state_dim）
state_dim = 14  # 這應該對應於 `SimpleTaxiEnv` 的 `get_state()` 維度
action_dim = 6  # 6 個可用行動（0,1,2,3,4,5）
agent = DQNAgent(state_dim, action_dim, MODEL_PATH)

def get_action(obs):
    """
    選擇最佳行動，如果 obs 格式錯誤或模型無法處理，則隨機選擇動作。
    """
    try:
        return agent.select_action(obs)
    except Exception as e:
        print(f"⚠️ 選擇動作時發生錯誤: {e}, 退回隨機行動")
        return random.choice([0, 1, 2, 3, 4, 5])  # 退回隨機選擇
