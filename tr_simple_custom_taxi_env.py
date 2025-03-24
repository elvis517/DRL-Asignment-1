import numpy as np
import random
from IPython.display import clear_output

import numpy as np
import random
from IPython.display import clear_output

import numpy as np
import random
from IPython.display import clear_output

class SimpleTaxiEnv():
    def __init__(self, grid_size=None, fuel_limit=5000):
        """
        自訂 Taxi 環境，支援不同大小的 grid，允許訓練與測試時使用固定 grid_size。
        """
        self.grid_size = grid_size if grid_size else random.randint(5, 10)
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False

        # 4 個固定的站點
        self.stations = [(0, 0), (0, self.grid_size - 1),
                         (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.destination = None
        self.obstacles = set()
        self.taxi_pos = None  # 確保 `taxi_pos` 屬性存在

    def reset(self, fixed_grid_size=None):
        """重設環境，可選擇固定 grid_size 以便測試一致性"""
        if fixed_grid_size:
            self.grid_size = fixed_grid_size
        else:
            self.grid_size = random.randint(5, 10)  # 訓練時隨機變換地圖大小

        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.obstacles.clear()

        # 產生障礙物，確保不影響 Taxi、乘客或目的地
        obstacle_count = random.randint(0, 5)
        available_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        available_positions -= set(self.stations)

        self.obstacles = set(random.sample(available_positions, obstacle_count))

        # 隨機設定 Taxi 位置，確保不在障礙物上
        available_positions -= self.obstacles
        
        self.taxi_pos = random.choice(list(available_positions))  # ✅ 確保 `taxi_pos` 被初始化
        # 隨機設定乘客與目的地
        self.passenger_loc = random.choice(self.stations)
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        return self.get_state(), {}

    def get_state(self):
        """返回當前環境的狀態，包含 Taxi-v3 格式，並擴充額外資訊 (含 grid_size)"""
        if self.taxi_pos is None:
            self.taxi_pos = (0, 0)

        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination

        passenger_look = int(self.taxi_pos == self.passenger_loc)
        destination_look = int(self.taxi_pos == self.destination)

        obstacle_north = int((taxi_row - 1, taxi_col) in self.obstacles or taxi_row == 0)
        obstacle_south = int((taxi_row + 1, taxi_col) in self.obstacles or taxi_row == self.grid_size - 1)
        obstacle_east  = int((taxi_row, taxi_col + 1) in self.obstacles or taxi_col == self.grid_size - 1)
        obstacle_west  = int((taxi_row, taxi_col - 1) in self.obstacles or taxi_col == 0)

        fuel_ratio = self.current_fuel / self.fuel_limit
        passenger_status = int(self.passenger_picked_up)

        # 🌟 **新增 grid_size，並標準化 (0~1)**，確保不同大小地圖能正確學習
        norm_grid_size = (self.grid_size-4) / 6.0  # 假設 grid_size 最大為 10，標準化到 0~1

        return (taxi_row, taxi_col, passenger_row, passenger_col, destination_row, destination_col, 
                obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
                passenger_look, destination_look, fuel_ratio, passenger_status)
        # return (taxi_row, taxi_col, passenger_row, passenger_col, destination_row, destination_col, 
        #         obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
        #         passenger_look, destination_look, fuel_ratio, passenger_status, norm_grid_size)


    def render_env(self):
        """顯示當前環境狀態"""
        clear_output(wait=True)
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        for x, y in self.obstacles:
            grid[x][y] = 'X'  # 障礙物
        
        for i, station in enumerate(self.stations):
            grid[station[0]][station[1]] = ['R', 'G', 'Y', 'B'][i]  # 車站

        tx, ty = self.taxi_pos
        grid[tx][ty] = '🚖' if not self.passenger_picked_up else '🚕'  # Taxi

        print(f"\nGrid Size: {self.grid_size}x{self.grid_size}, Fuel: {self.current_fuel}")
        for row in grid:
            print(" ".join(row))
        print("\n")

    def step(self, action):
        """執行動作，更新環境狀態"""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0

        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        # 檢查是否撞到障礙物或超出邊界
        if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
            reward -= 50  # ✅ 碰撞懲罰 -50
        else:
            self.taxi_pos = (next_row, next_col)
            if self.passenger_picked_up:
                self.passenger_loc = self.taxi_pos

        if action == 4:  # PICKUP
            if self.taxi_pos == self.passenger_loc:
                self.passenger_picked_up = True
                self.passenger_loc = self.taxi_pos  
            else:
                reward -= 50  # ✅ 錯誤 PICKUP 懲罰 -50

        elif action == 5:  # DROPOFF
            if self.passenger_picked_up:
                if self.taxi_pos == self.destination:
                    reward += 2000  # ✅ 成功送達乘客
                    return self.get_state(), reward, True, {}  # 遊戲結束
                else:
                    reward -= 50  # ✅ 錯誤 DROPOFF 懲罰 -50
                self.passenger_picked_up = False
                self.passenger_loc = self.taxi_pos
            else:
                reward -= 50  # ✅ 無乘客時執行 DROP 懲罰 -50

        reward -= 0.1  # ✅ 移動步數的懲罰 -0.1
        self.current_fuel -= 1

        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}  # ✅ 燃料耗盡時 -10 分並結束遊戲

        return self.get_state(), reward, False, {}
