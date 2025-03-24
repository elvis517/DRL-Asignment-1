import numpy as np
import random
from IPython.display import clear_output
import time
import importlib.util


class SimpleTaxiEnv():
    def __init__(self, grid_size=None, fuel_limit=5000):
        """
        自訂 Taxi 環境，支援不同大小的 grid，允許訓練與測試時使用固定 grid_size。
        """
        self.grid_size = grid_size if grid_size else random.randint(5, 10)
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False

        # 根據 grid_size 設定四個固定站點
        self.stations = [(0, 0), (0, self.grid_size - 1),
                         (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.destination = None
        self.obstacles = set()
        self.taxi_pos = None  # 確保 `taxi_pos` 屬性存在

    def reset(self, fixed_grid_size=None):
        """重設環境，可選擇固定 grid_size 以便測試一致性"""
        # 設定 grid_size 並依此更新站點
        if fixed_grid_size:
            self.grid_size = fixed_grid_size
        else:
            self.grid_size = random.randint(5, 10)
        self.stations = [(0, 0), (0, self.grid_size - 1),
                         (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.obstacles.clear()

        # 產生障礙物，避免影響到站點
        obstacle_count = random.randint(0, 5)
        available_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        available_positions -= set(self.stations)
        self.obstacles = set(random.sample(list(available_positions), obstacle_count))

        # 選擇 Taxi 起始位置（避開障礙物與站點）
        available_positions -= self.obstacles
        self.taxi_pos = random.choice(list(available_positions))
        
        # 隨機設定乘客位置與目的地（確保目的地與乘客位置不同）
        self.passenger_loc = random.choice(self.stations)
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        return self.get_state(), {}

    def get_state(self):
        """回傳目前環境狀態"""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        # 判斷四個方向是否有障礙或超出邊界
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        # 判斷乘客是否在 Taxi 附近或正好在 Taxi 位置
        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        # 判斷目的地是否在 Taxi 附近或正好在 Taxi 位置
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        # 回傳狀態 tuple，可依需求調整輸出資訊
        state = (taxi_row, taxi_col,
                 self.stations[0][0], self.stations[0][1],
                 self.stations[1][0], self.stations[1][1],
                 self.stations[2][0], self.stations[2][1],
                 self.stations[3][0], self.stations[3][1],
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                 passenger_look, destination_look)
        return state

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

        # 根據動作決定移動方向
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
            reward -= 50  # 碰撞懲罰 -50
        else:
            self.taxi_pos = (next_row, next_col)
            # 若已接乘客，乘客位置隨 Taxi 移動
            if self.passenger_picked_up:
                reward += 20
                self.passenger_loc = self.taxi_pos

        if action == 4:  # PICKUP
            if self.taxi_pos == self.passenger_loc:
                self.passenger_picked_up = True
                self.passenger_loc = self.taxi_pos  
            else:
                reward -= 50  # 錯誤 PICKUP 懲罰 -50

        elif action == 5:  # DROPOFF
            if self.passenger_picked_up:
                if self.taxi_pos == self.destination:
                    reward += 200  # 成功送達乘客獎勵 +200
                    return self.get_state(), reward, True, {}  # 遊戲結束
                else:
                    reward -= 20  # 錯誤 DROPOFF 懲罰 -50
                self.passenger_picked_up = False
                self.passenger_loc = self.taxi_pos
            else:
                reward -= 20  # 無乘客時執行 DROPOFF 懲罰 -50

        reward -= 0.1  # 每步移動懲罰 -0.1
        self.current_fuel -= 1

        # 檢查燃料是否耗盡
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}  # 燃料耗盡時額外扣 -10 並結束遊戲

        return self.get_state(), reward, False, {}


        return self.get_state(), reward, False, {}
def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")