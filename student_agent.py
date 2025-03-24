import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """輕量版 DQN，與訓練時相同的網路結構"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # 修改 fc3 輸入維度為 64 以符合 fc2 輸出
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 根據訓練時設定，state_dim 為 14, action_dim 為 6
state_dim = 16
action_dim = 6

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入訓練後的模型參數，檔名與路徑請依實際情況調整
policy_net = DQN(state_dim, action_dim)
policy_net.load_state_dict(torch.load("dqn_taxi_light128.pth"))
policy_net.eval()

def get_action(obs):
    """
    接收環境回傳的觀察 obs（16 維向量），並返回一個動作 (0~5)
    """
    # 確保 obs 為 numpy float32 向量
    obs = np.array(obs, dtype=np.float32)
    obs_tensor = torch.tensor(obs).unsqueeze(0)  # shape: (1, 14)
    with torch.no_grad():
        q_values = policy_net(obs_tensor)
    action = int(torch.argmax(q_values, dim=1).item())
    return action
