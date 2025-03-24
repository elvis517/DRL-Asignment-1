import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# 嘗試載入 DQN 模型
model_path = "dqn_model.pth"
if os.path.exists(model_path):
    model = torch.load(model_path)
    model.eval()  # 設置為評估模式
else:
    model = None

# 定義神經網絡架構
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q值

# 選擇動作
def get_action(obs):
    if model is None:
        return random.choice([0, 1, 2, 3, 4, 5])  # 沒有模型時，隨機選擇

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # 加 batch 維度
    with torch.no_grad():
        q_values = model(obs_tensor)
    return torch.argmax(q_values).item()  # 選擇 Q 值最大的動作
