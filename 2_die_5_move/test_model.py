import json
import numpy as np
from tensorflow.keras.models import load_model
from gym import Env
from gym.spaces import Discrete, Box
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def apply_pattern(board, pattern, position, direction):
    x, y = position
    pattern_width, pattern_height = pattern['width'], pattern['height']
    cells = pattern['cells']
    board_width, board_height = len(board[0]), len(board)

    lifted = []
    for i in range(pattern_height):
        for j in range(pattern_width):
            bx, by = x + j, y + i
            if 0 <= bx < board_width and 0 <= by < board_height:
                if cells[i][j] == 1:
                    lifted.append(board[by][bx])
                    board[by][bx] = -1

    def shift_board(board, direction):
        if direction == 0:  # Top
            for col in range(board_width):
                non_neg = [board[row][col] for row in range(board_height) if board[row][col] != -1]
                for row in range(board_height):
                    board[row][col] = non_neg[row] if row < len(non_neg) else -1
        elif direction == 1:  # Bottom
            for col in range(board_width):
                non_neg = [board[row][col] for row in range(board_height) if board[row][col] != -1]
                for row in range(board_height - 1, -1, -1):
                    board[row][col] = non_neg[len(non_neg) - (board_height - row)] if (board_height - row) <= len(non_neg) else -1
        elif direction == 2:  # Left
            for row in range(board_height):
                non_neg = [board[row][col] for col in range(board_width) if board[row][col] != -1]
                for col in range(board_width):
                    board[row][col] = non_neg[col] if col < len(non_neg) else -1
        elif direction == 3:  # Right
            for row in range(board_height):
                non_neg = [board[row][col] for col in range(board_width) if board[row][col] != -1]
                for col in range(board_width - 1, -1, -1):
                    board[row][col] = non_neg[len(non_neg) - (board_width - col)] if (board_width - col) <= len(non_neg) else -1

    shift_board(board, direction)

    empty_cells = [(r, c) for r in range(board_height) for c in range(board_width) if board[r][c] == -1]
    for idx, (r, c) in enumerate(empty_cells):
        board[r][c] = lifted[idx] if idx < len(lifted) else -1

    return board

def load_problem(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    board_start = [list(map(int, row)) for row in data['board']['start']]
    board_goal = [list(map(int, row)) for row in data['board']['goal']]
    patterns = [
        {
            'id': pattern['p'],
            'width': pattern['width'],
            'height': pattern['height'],
            'cells': [list(map(int, row)) for row in pattern['cells']]
        }
        for pattern in data['general']['patterns']
    ]
    return board_start, board_goal, patterns

# Hàm giải bài toán với mô hình đã huấn luyện
def test_model(env, agent):
    state = env.reset()
    solution = {"n": 0, "ops": []}

    for step in range(env.max_steps):
        action = agent.act(state)
        x, y, direction = env.decode_action(action)

        # Lưu hành động
        solution["ops"].append({
            "p": env.die['id'],  # ID khuôn duy nhất
            "x": x,
            "y": y,
            "s": direction  # 0 = top, 1 = bottom, 2 = left, 3 = right
        })
        solution["n"] += 1

        # Tiến hành bước
        next_state, _, done, _ = env.step(action)
        state = next_state

        if done:
            break

    return solution

class BoardEnvWithTwoDies(Env):
    def __init__(self, board, goal, dies, max_steps=5):
        """
        Môi trường RL cho bài toán 2 khuôn.

        Args:
            board: Bảng bắt đầu (numpy array hoặc list of lists).
            goal: Bảng mục tiêu (numpy array hoặc list of lists).
            dies: Danh sách các khuôn (list of dicts).
            max_steps: Số bước tối đa.
        """
        super(BoardEnvWithTwoDies, self).__init__()
        self.board = np.array(board)
        self.goal = np.array(goal)
        self.dies = dies  # Danh sách các khuôn
        self.num_dies = len(dies)  # Số khuôn
        self.width, self.height = self.board.shape
        self.max_steps = max_steps
        self.current_step = 0

        # Không gian hành động: (chọn khuôn, x, y, hướng)
        self.action_space = Discrete(self.num_dies * self.width * self.height * 4)
        # Không gian quan sát: trạng thái của bảng
        self.observation_space = Box(low=0, high=3, shape=(self.width, self.height), dtype=int)

    def reset(self):
        """Đặt lại môi trường về trạng thái ban đầu."""
        self.state = self.board.copy()
        self.current_step = 0
        return self.state

    def step(self, action):
        """Thực hiện một hành động và trả về trạng thái tiếp theo."""
        die_index, x, y, direction = self.decode_action(action)
        selected_die = self.dies[die_index]

        # Áp dụng khuôn và dịch chuyển
        new_board = self.state.copy()
        new_board = self.apply_pattern(new_board, selected_die, (x, y), direction)

        # Tính phần thưởng
        reward = -1
        done = False
        if np.array_equal(new_board, self.goal):
            reward = 100
            done = True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        self.state = new_board
        return new_board, reward, done, {}

    def decode_action(self, action):
        """Giải mã hành động thành (khuôn, x, y, hướng)."""
        num_cells = self.width * self.height * 4  # Số ô và hướng
        die_index = action // num_cells
        remainder = action % num_cells

        x = remainder // (self.height * 4)
        remainder = remainder % (self.height * 4)

        y = remainder // 4
        direction = remainder % 4

        return die_index, x, y, direction

    def apply_pattern(self, board, pattern, position, direction):
        """Áp dụng khuôn lên bảng và xử lý dịch chuyển."""
        x, y = position
        pattern_width, pattern_height = pattern['width'], pattern['height']
        cells = pattern['cells']
        board_width, board_height = board.shape

        # Áp dụng khuôn
        lifted = []
        for i in range(pattern_height):
            for j in range(pattern_width):
                bx, by = x + j, y + i
                if 0 <= bx < board_width and 0 <= by < board_height:
                    if cells[i][j] == 1:
                        lifted.append(board[by][bx])
                        board[by][bx] = -1  # Đánh dấu ô rỗng

        # Dịch chuyển các ô bị nâng lên
        board = self.shift_board(board, direction, lifted)
        return board

    def shift_board(self, board, direction, lifted):
        """Dịch chuyển các ô được nâng lên theo hướng."""
        board_width, board_height = board.shape
        if direction == 0:  # Top
            for col in range(board_width):
                column = [board[row][col] for row in range(board_height) if board[row][col] != -1]
                column = [-1] * (board_height - len(column)) + column
                for row in range(board_height):
                    board[row][col] = column[row]

        elif direction == 1:  # Bottom
            for col in range(board_width):
                column = [board[row][col] for row in range(board_height) if board[row][col] != -1]
                column = column + [-1] * (board_height - len(column))
                for row in range(board_height):
                    board[row][col] = column[row]

        elif direction == 2:  # Left
            for row in range(board_height):
                row_data = [board[row][col] for col in range(board_width) if board[row][col] != -1]
                row_data = [-1] * (board_width - len(row_data)) + row_data
                for col in range(board_width):
                    board[row][col] = row_data[col]

        elif direction == 3:  # Right
            for row in range(board_height):
                row_data = [board[row][col] for col in range(board_width) if board[row][col] != -1]
                row_data = row_data + [-1] * (board_width - len(row_data))
                for col in range(board_width):
                    board[row][col] = row_data[col]

        empty_cells = [(r, c) for r in range(board_height) for c in range(board_width) if board[r][c] == -1]
        for idx, (r, c) in enumerate(empty_cells):
            board[r][c] = lifted[idx] if idx < len(lifted) else -1

        return board


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.model = None

    def load_model(self, weights_file):
        self.model = load_model(weights_file)

    def act(self, state):
        q_values = self.model.predict(state[np.newaxis])
        return np.argmax(q_values[0])

def load_agent_for_testing(weights_file, state_shape, action_size):
    agent = DQNAgent(state_shape, action_size)
    agent.model = load_model(
        weights_file,
        compile=False  # Tải mô hình mà không biên dịch lại
    )
    agent.model.compile(optimizer=Adam(), loss="mean_squared_error")  # Biên dịch lại mô hình với hàm tổn thất "mse"
    return agent

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):  # Nếu là số nguyên của numpy
        return int(obj)
    elif isinstance(obj, np.floating):  # Nếu là số thực của numpy
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Nếu là mảng numpy
        return obj.tolist()
    elif isinstance(obj, list):  # Nếu là danh sách
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, dict):  # Nếu là từ điển
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:  # Các loại khác không cần chuyển đổi
        return obj
    
def save_solution_to_json(solution, filename):
    # Chuyển đổi các kiểu dữ liệu không tương thích
    solution = convert_numpy_types(solution)
    with open(filename, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    import argparse

    # Argument parser để chọn file trọng số và file bài toán
    file_w = r"...\RL_procon.h5"
    file_json = r"...\problem.json"
    file_out = r"...\out.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help=file_w)
    parser.add_argument("--problem", type=str, required=True, help=file_json)
    parser.add_argument("--output", type=str, default="solution.json", help=file_out)
    args = parser.parse_args()

    # Load bài toán từ file JSON
    board_start, board_goal, patterns = load_problem(args.problem)

    # Chọn khuôn duy nhất
    die = patterns[0]

    # Tạo môi trường RL
    env = BoardEnvWithTwoDies(board_start, board_goal, die, max_steps=5)

    # Load mô hình đã huấn luyện
    agent = load_agent_for_testing(args.weights, env.observation_space.shape, env.action_space.n)

    # Giải bài toán
    print("Solving the board...")
    solution = test_model(env, agent)

    # Lưu kết quả ra file JSON
    save_solution_to_json(solution, args.output)
    print(f"Solution saved to {args.output}")

'''
python test_model.py --weights RL_procon.h5 --problem problem.json --output out.json
'''