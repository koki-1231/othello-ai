# othello-ai
import numpy as np
import random
import copy
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

# 0. Global Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# A. Visualization and Helper Functions (No changes)

def plot_metrics(history, filename):
    """Plots training history (win rate and loss) and saves it as an image."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    episodes = history.get('episodes', [])
    losses = history.get('losses', [])
    eval_episodes = history.get('eval_episodes', [])
    win_rates = history.get('win_rates', [])
    
    if not episodes or not losses or not eval_episodes or not win_rates:
        print("History data is incomplete. Skipping plot generation.")
        return
        
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(episodes, losses, color='tab:blue', alpha=0.6, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Win Rate vs Random', color='tab:red')
    ax2.plot(eval_episodes, win_rates, color='tab:red', marker='o', linestyle='-', label='Win Rate')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.suptitle('Training Progress', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

def create_board_image(board, turn_number, player_name):
    """Generates an image representation of the board state."""
    cell_size, margin = 50, 25
    board_size = board.shape[0]
    img_size = board_size * cell_size + 2 * margin
    img = Image.new("RGB", (img_size, img_size), (0, 100, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        
    for r in range(board_size):
        for c in range(board_size):
            x0, y0 = margin + c * cell_size, margin + r * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], outline="black")
            stone = board[r, c]
            if stone != 0:
                color = "black" if stone == 1 else "white"
                draw.ellipse([x0 + 5, y0 + 5, x1 - 5, y1 - 5], fill=color, outline=color)
                
    text = f"Turn: {turn_number}\nPlayer: {player_name}"
    draw.text((margin, 2), text, font=font, fill=(255, 255, 255))
    return img


# B. Game Environment and Basic Agents (No major changes)

class OthelloEnv:
    def __init__(self):
        self.board_size = 8
        self.reset()
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.board[3, 3] = self.board[4, 4] = 1; self.board[3, 4] = self.board[4, 3] = -1
        self.current_player = 1; self.done = False
        return self.get_state()
    def get_state(self): return self.board.copy(), self.current_player
    def get_legal_moves(self, player):
        moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.is_valid_move(r, c, player): moves.append((r, c))
        return moves
    def is_valid_move(self, r, c, player):
        if self.board[r, c] != 0: return False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                r_temp, c_temp, flipped_found = r + dr, c + dc, False
                while 0 <= r_temp < self.board_size and 0 <= c_temp < self.board_size:
                    if self.board[r_temp, c_temp] == -player: flipped_found = True; r_temp += dr; c_temp += dc
                    elif self.board[r_temp, c_temp] == player and flipped_found: return True
                    else: break
        return False
    def step(self, action):
        if self.done: return self.get_state(), 0, True
        if action is None:
            self.current_player *= -1
            if not self.get_legal_moves(self.current_player): self.done = True
            return self.get_state(), 0, self.done
        r, c = action
        if not self.is_valid_move(r, c, self.current_player): self.done = True; return self.get_state(), -1, True
        self.board[r, c] = self.current_player
        self._flip_stones(r, c, self.current_player)
        self.current_player *= -1
        if not self.get_legal_moves(1) and not self.get_legal_moves(-1): self.done = True
        elif not self.get_legal_moves(self.current_player): self.current_player *= -1
        winner = 0
        if self.done:
            score = np.sum(self.board)
            if score > 0: winner = 1
            elif score < 0: winner = -1
        return self.get_state(), winner, self.done
    def _flip_stones(self, r, c, player):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                r_temp, c_temp, stones_to_flip = r + dr, c + dc, []
                while 0 <= r_temp < self.board_size and 0 <= c_temp < self.board_size:
                    if self.board[r_temp, c_temp] == -player: stones_to_flip.append((r_temp, c_temp)); r_temp += dr; c_temp += dc
                    elif self.board[r_temp, c_temp] == player and stones_to_flip:
                        for fr, fc in stones_to_flip: self.board[fr, fc] = player
                        break
                    else: break

class BaseAgent:
    def __init__(self, player): self.player = player
    def get_action(self, env): raise NotImplementedError

class HumanAgent(BaseAgent):
    def __init__(self, player, app_ref):
        super().__init__(player)
        self.app = app_ref
    def get_action(self, env):
        return self.app.get_human_move()

class RandomAgent(BaseAgent):
    def get_action(self, env):
        legal_moves = env.get_legal_moves(self.player)
        return random.choice(legal_moves) if legal_moves else None

class AlphaBetaAgent(BaseAgent):
    def __init__(self, player, depth=3):
        super().__init__(player); self.depth = depth
        self.WEIGHTS = np.array([[120,-20,20,5,5,20,-20,120],[-20,-40,-5,-5,-5,-5,-40,-20],[20,-5,15,3,3,15,-5,20],[5,-5,3,3,3,3,-5,5],
                                 [5,-5,3,3,3,3,-5,5],[20,-5,15,3,3,15,-5,20],[-20,-40,-5,-5,-5,-5,-40,-20],[120,-20,20,5,5,20,-20,120]])
    def get_action(self, env):
        time.sleep(0.5) # Add delay for better visualization
        legal_moves = env.get_legal_moves(self.player)
        if not legal_moves: return None
        best_move, best_score = None, -float('inf')
        alpha, beta = -float('inf'), float('inf')
        for move in legal_moves:
            next_env = self._get_next_env(env, move)
            score = self._alphabeta(next_env, self.depth - 1, alpha, beta, False)
            if score > best_score: best_score, best_move = score, move
            alpha = max(alpha, best_score)
        return best_move
    def _alphabeta(self, env, depth, alpha, beta, is_max):
        if depth == 0 or env.done: return self._evaluate_board(env.board)
        legal_moves = env.get_legal_moves(env.current_player)
        if not legal_moves:
            next_env = copy.deepcopy(env); next_env.current_player *= -1
            return self._alphabeta(next_env, depth, alpha, beta, not is_max)
        if is_max:
            max_eval = -float('inf')
            for move in legal_moves:
                eval = self._alphabeta(self._get_next_env(env, move), depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval); alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                eval = self._alphabeta(self._get_next_env(env, move), depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval); beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval
    def _evaluate_board(self, board): return np.sum(board * self.WEIGHTS) * self.player
    def _get_next_env(self, env, move):
        new_env = copy.deepcopy(env); new_env.step(move)
        return new_env


# C. DQN Implementation (No changes)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__(); self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1); self.fc1 = nn.Linear(64 * 8 * 8, 512); self.fc2 = nn.Linear(512, 64)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = x.view(x.size(0), -1); x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent(BaseAgent):
    def __init__(self, player, learning=True):
        super().__init__(player); self.learning = learning
        self.gamma, self.batch_size, self.replay_memory_size = 0.99, 256, 10000
        self.lr, self.target_update_frequency = 1e-4, 100
        self.epsilon_start, self.epsilon_end, self.epsilon_decay = 1.0, 0.01, 3000
        self.policy_net = QNetwork().to(DEVICE); self.target_net = QNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = collections.deque(maxlen=self.replay_memory_size); self.steps_done = 0
    def _state_to_tensor(self, board): return torch.tensor(board * self.player, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    def _action_to_index(self, action): return action[0] * 8 + action[1] if action else 64
    def remember(self, s, a, s_next, r, d):
        s_tensor = self._state_to_tensor(s); a_tensor = torch.tensor([[self._action_to_index(a)]], device=DEVICE, dtype=torch.long)
        s_next_tensor = self._state_to_tensor(s_next[0]) if not d else None; r_tensor = torch.tensor([r], device=DEVICE, dtype=torch.float32)
        self.replay_buffer.append((s_tensor, a_tensor, s_next_tensor, r_tensor))
    def get_action(self, env):
        time.sleep(0.5) # Add delay for better visualization
        legal_moves = env.get_legal_moves(self.player);
        if not legal_moves: return None
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        if self.learning: self.steps_done += 1
        if random.random() > eps or not self.learning:
            with torch.no_grad():
                q_values = self.policy_net(self._state_to_tensor(env.board))[0]
                legal_q = {m: q_values[self._action_to_index(m)].item() for m in legal_moves}
                return max(legal_q, key=legal_q.get)
        else: return random.choice(legal_moves)
    def learn(self):
        if len(self.replay_buffer) < self.batch_size: return None
        transitions = random.sample(self.replay_buffer, self.batch_size); s_batch, a_batch, s_next_batch, r_batch = zip(*transitions)
        s_batch = torch.cat(s_batch); a_batch = torch.cat(a_batch); r_batch = torch.cat(r_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, s_next_batch)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in s_next_batch if s is not None])
        state_action_values = self.policy_net(s_batch).gather(1, a_batch); next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + r_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()
    def load(self, path): self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
    def save(self, path): torch.save(self.policy_net.state_dict(), path)


# D. MCTS+NN Implementation 

class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__(); self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1); self.policy_fc = nn.Linear(64 * 8 * 8, 64)
        self.value_fc1 = nn.Linear(64 * 8 * 8, 256); self.value_fc2 = nn.Linear(256, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x_flat = x.view(x.size(0), -1)
        policy_log_probs = F.log_softmax(self.policy_fc(x_flat), dim=1); v = F.relu(self.value_fc1(x_flat))
        value = torch.tanh(self.value_fc2(v)); return policy_log_probs, value

class Node:
    def __init__(self, parent=None, prior_p=1.0): self.parent, self.children, self.N, self.W, self.Q, self.P = parent, {}, 0, 0, 0, prior_p
    def expand(self, act_p): [self.children.setdefault(act, Node(self, p)) for act, p in act_p.items()]
    def select(self, c): return max(self.children.items(), key=lambda an: an[1].get_ucb(c))
    def get_ucb(self, c): return self.Q + c * self.P * math.sqrt(self.parent.N) / (1 + self.N)
    def update(self, v): self.N += 1; self.W += v; self.Q = self.W / self.N
    def backprop(self, v): self.update(v); (self.parent and self.parent.backprop(-v))

class MCTS:
    def __init__(self, p_v_fn, c=1.0, n_sim=100): self.root, self.p_v_fn, self.c, self.n_sim = Node(), p_v_fn, c, n_sim
    def _playout(self, env):
        node, player = self.root, env.current_player
        while node.children: action, node = node.select(self.c); env.step(action)
        legal_moves = env.get_legal_moves(env.current_player)
        act_p, v = self.p_v_fn(env)
        if not env.done and legal_moves: node.expand({m: act_p[m[0]*8+m[1]] for m in legal_moves})
        node.backprop(-v if env.current_player != player else v)
    def get_move_probs(self, env, t=1e-3):
        for _ in range(self.n_sim): self._playout(copy.deepcopy(env))
        act_v = [(a, n.N) for a, n in self.root.children.items()]
        if not act_v: return None, None
        acts, visits = zip(*act_v); probs = F.softmax(torch.tensor(visits, dtype=torch.float32) / t, dim=0).numpy()
        return acts, probs
    def update_with_move(self, move): self.root = self.root.children.get(move, Node()); self.root.parent = None
        
class MCTSAgent(BaseAgent):
    def __init__(self, player, n_sim=100, learning=True, temp=1.0):
        super().__init__(player); self.learning, self.temp = learning, temp
        self.net = PolicyValueNetwork().to(DEVICE); self.mcts = MCTS(self._policy_value_fn, n_simulations=n_sim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3); self.replay_buffer = collections.deque(maxlen=10000)
    def _policy_value_fn(self, env):
        board = env.board * env.current_player; tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): log_p, v = self.net(tensor)
        return np.exp(log_p.cpu().numpy()[0]), v.item()
    def get_action(self, env, return_prob=False):
        time.sleep(0.5) # Add delay for better visualization
        legal_moves = env.get_legal_moves(self.player)
        if not legal_moves: return None
        acts, probs = self.mcts.get_move_probs(env, self.temp)
        if acts is None: return random.choice(legal_moves)
        move_probs = np.zeros(64); move_probs[[a[0]*8+a[1] for a in acts]] = probs
        action_idx = np.random.choice(len(acts), p=probs) if self.learning else np.argmax(probs)
        action = acts[action_idx]; self.mcts.update_with_move(action)
        return (action, move_probs) if return_prob else action
    def remember(self, s, p, z): self.replay_buffer.append((s,p,z))
    def learn(self, batch_size=256):
        if len(self.replay_buffer) < batch_size: return None
        s_b, p_b, z_b = zip(*random.sample(self.replay_buffer, batch_size))
        s_b, p_b, z_b = (torch.tensor(np.array(b), dtype=torch.float32).to(DEVICE) for b in (s_b, p_b, z_b))
        self.optimizer.zero_grad(); log_p_pred, v_pred = self.net(s_b.unsqueeze(1))
        loss = F.mse_loss(v_pred.view(-1), z_b) - torch.sum(p_b * log_p_pred) / batch_size
        loss.backward(); self.optimizer.step()
        return loss.item()
    def reset_mcts(self): self.mcts = MCTS(self._policy_value_fn, n_sim=self.mcts.n_simulations)
    def load(self, path): self.net.load_state_dict(torch.load(path, map_location=DEVICE))
    def save(self, path): torch.save(self.net.state_dict(), path)

# E. Training Logic

def evaluate_agent(agent, num_games=20):
    wins = 0; agent.learning = False
    for _ in range(num_games):
        env, test_agent_player = OthelloEnv(), random.choice([1, -1])
        agent.player = test_agent_player; opponent = RandomAgent(-test_agent_player)
        agents = {1: agent if test_agent_player == 1 else opponent, -1: agent if test_agent_player == -1 else opponent}
        while not env.done:
            action = agents[env.current_player].get_action(env)
            (board, player), winner, done = env.step(action)
        if winner == test_agent_player: wins += 1
    agent.learning = True
    return wins / num_games

def train(agent, num_episodes, eval_interval=100):
    history = {'episodes': [], 'losses': [], 'eval_episodes': [], 'win_rates': []}
    agent_type = 'dqn' if isinstance(agent, DQNAgent) else 'mcts'
    opponent = copy.deepcopy(agent); opponent.learning = False
    for i_episode in tqdm(range(num_episodes)):
        env = OthelloEnv()
        if agent_type == 'mcts': agent.reset_mcts()
        agents = {1: agent, -1: opponent} if random.random() > 0.5 else {-1: agent, 1: opponent}
        state, game_history = env.get_state(), []
        while not env.done:
            current_player = env.current_player; action = agents[current_player].get_action(env)
            pi = None
            if agent_type == 'mcts' and isinstance(action, tuple): action, pi = action
            game_history.append({'state': state[0], 'player': current_player, 'pi': pi})
            next_state, winner, done = env.step(action)
            if agent_type == 'dqn' and current_player == agent.player:
                reward = 0
                if done: reward = 1 if winner == agent.player else -1
                agent.remember(state[0], action, next_state, reward, done)
            state = next_state
        if agent_type == 'mcts':
            for transition in game_history:
                if transition['player'] == agent.player:
                    z = 1 if winner == agent.player else -1
                    normalized_state = transition['state'] * transition['player']
                    agent.remember(normalized_state, transition['pi'], z)
        loss = agent.learn()
        if loss: history['episodes'].append(i_episode); history['losses'].append(loss)
        if isinstance(agent, DQNAgent) and i_episode % agent.target_update_frequency == 0:
            opponent.policy_net.load_state_dict(agent.policy_net.state_dict())
        if i_episode > 0 and i_episode % eval_interval == 0:
            win_rate = evaluate_agent(agent, num_games=20)
            history['eval_episodes'].append(i_episode); history['win_rates'].append(win_rate)
            tqdm.write(f"Episode {i_episode}, Win Rate: {win_rate:.2f}")
    return history


# F. GUI Application

class OthelloGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Othello AI Battle")
        self.resizable(False, False)
        self.configure(bg='#F0F0F0')
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.game_thread, self.human_move = None, None
        self.human_move_event = threading.Event()
        self.photo_images = {}
        self.create_widgets()
        self.draw_board(OthelloEnv().board)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        board_frame = ttk.Frame(main_frame)
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.canvas = tk.Canvas(board_frame, width=440, height=440, bg="#006400", highlightthickness=0)
        self.canvas.pack(pady=10, padx=10)
        self.canvas.bind("<Button-1>", self.on_board_click)
        self.cell_size = 440 // 8

        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)

        p1_frame = ttk.LabelFrame(control_frame, text="Black (Player 1)")
        p1_frame.pack(pady=5, fill=tk.X)
        self.player1_combo = ttk.Combobox(p1_frame, values=["Human", "Random", "AlphaBeta", "DQN", "MCTS"], state="readonly")
        self.player1_combo.set("Human"); self.player1_combo.pack(pady=5, padx=5, fill=tk.X)

        p2_frame = ttk.LabelFrame(control_frame, text="White (Player -1)")
        p2_frame.pack(pady=5, fill=tk.X)
        self.player2_combo = ttk.Combobox(p2_frame, values=["Human", "Random", "AlphaBeta", "DQN", "MCTS"], state="readonly")
        self.player2_combo.set("DQN"); self.player2_combo.pack(pady=5, padx=5, fill=tk.X)

        self.record_var = tk.BooleanVar()
        record_check = ttk.Checkbutton(control_frame, text="Record AI vs AI match", variable=self.record_var)
        record_check.pack(pady=5)

        self.start_button = ttk.Button(control_frame, text="Start Game", command=self.start_game)
        self.start_button.pack(pady=10, fill=tk.X, ipady=5)
        
        self.status_label = ttk.Label(control_frame, text="Welcome to Othello!", wraplength=230, justify=tk.LEFT, font=('Arial', 10))
        self.status_label.pack(pady=10, fill=tk.X)
        
        ttk.Separator(control_frame).pack(fill='x', pady=10)

        self.train_button = ttk.Button(control_frame, text="Train Agents", command=self.open_train_window)
        self.train_button.pack(pady=5, fill=tk.X)
        self.show_plot_button = ttk.Button(control_frame, text="Show Learning Curve", command=self.show_plot)
        self.show_plot_button.pack(pady=5, fill=tk.X)
        self.show_video_button = ttk.Button(control_frame, text="Show AI Match Video", command=self.show_video)
        self.show_video_button.pack(pady=5, fill=tk.X)

    def draw_board(self, board, legal_moves=[]):
        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                x0, y0 = c * self.cell_size, r * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", fill="#006400")
                stone = board[r][c]
                if stone != 0:
                    color = "black" if stone == 1 else "white"
                    self.canvas.create_oval(x0+5, y0+5, x1-5, y1-5, fill=color, outline=color)
        if legal_moves:
             for r, c in legal_moves:
                x0, y0 = c * self.cell_size, r * self.cell_size
                self.canvas.create_oval(x0+18, y0+18, x0+self.cell_size-18, y0+self.cell_size-18, fill="gray", outline="")

    def on_board_click(self, event):
        if self.human_move_event.is_set(): return
        c, r = event.x // self.cell_size, event.y // self.cell_size
        self.human_move = (r, c)
        self.human_move_event.set()

    def get_human_move(self):
        self.human_move_event.clear()
        self.human_move_event.wait()
        return self.human_move

    def start_game(self):
        if self.game_thread and self.game_thread.is_alive(): return
        p1_type, p2_type = self.player1_combo.get(), self.player2_combo.get()
        record = self.record_var.get()
        if record and (p1_type == "Human" or p2_type == "Human"):
            messagebox.showwarning("Recording", "Recording is only available for AI vs AI matches.")
            self.record_var.set(False)
            record = False
        self.start_button.config(state=tk.DISABLED)
        self.game_thread = threading.Thread(target=self.game_loop, args=(p1_type, p2_type, record), daemon=True)
        self.game_thread.start()

    def get_agent(self, agent_type, player):
        if agent_type == "Human": return HumanAgent(player, self)
        if agent_type == "Random": return RandomAgent(player)
        if agent_type == "AlphaBeta": return AlphaBetaAgent(player, depth=3)
        model_path = f"{agent_type.lower()}_othello.pth"
        if not os.path.exists(model_path):
            messagebox.showerror("Model Not Found", f"Model for {agent_type} at '{model_path}' not found. Please train first.")
            return None
        if agent_type == "DQN": agent = DQNAgent(player, learning=False); agent.load(model_path); return agent
        if agent_type == "MCTS": agent = MCTSAgent(player, n_sim=100, learning=False); agent.load(model_path); return agent
        return None

    def game_loop(self, p1_type, p2_type, record):
        env = OthelloEnv()
        player1, player2 = self.get_agent(p1_type, 1), self.get_agent(p2_type, -1)
        if not player1 or not player2:
            self.after(0, lambda: self.start_button.config(state=tk.NORMAL)); return
        agents = {1: player1, -1: player2}
        names = {1: f"Black ({p1_type})", -1: f"White ({p2_type})"}
        frames, turn = [], 0

        while not env.done:
            player = env.current_player
            legal_moves = env.get_legal_moves(player)
            is_human = isinstance(agents[player], HumanAgent)
            self.after(0, self.draw_board, env.board, legal_moves if is_human else [])
            self.after(0, self.update_status, f"Turn: {names[player]}")
            if record: frames.append(create_board_image(env.board, turn, names[player]))
            
            if not legal_moves:
                self.after(0, self.update_status, f"{names[player]} passes."); time.sleep(1); action = None
            else:
                action = agents[player].get_action(env)
                if is_human and action not in legal_moves:
                     self.after(0, messagebox.showwarning, "Invalid Move", "That is not a legal move."); continue
            env.step(action); turn += 1
            if not is_human: time.sleep(0.5)

        self.after(0, self.draw_board, env.board)
        score = np.sum(env.board)
        if score == 0: result_msg = "Draw game."
        else: winner = names[1] if score > 0 else names[-1]; result_msg = f"Winner is {winner}!"
        self.after(0, self.update_status, f"Game Over!\n{result_msg}")
        
        if record and frames:
            frames.append(create_board_image(env.board, turn, "Game Over"))
            video_filename = f"{p1_type.lower()}_vs_{p2_type.lower()}.gif"
            imageio.mimsave(video_filename, frames, duration=1.0)
            self.after(0, messagebox.showinfo, "Recording Complete", f"Game video saved to {video_filename}")

        self.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def update_status(self, message): self.status_label.config(text=message)
    def open_train_window(self): TrainWindow(self)
    def show_plot(self):
        agent_type = tk.simpledialog.askstring("Input", "Enter agent type (dqn or mcts):", parent=self)
        if not agent_type: return
        filename = f"{agent_type.lower()}_learning_curve.png"
        if not os.path.exists(filename): messagebox.showerror("File Not Found", f"Plot file '{filename}' not found. Please train first."); return
        ImageViewer(self, filename, f"{agent_type.upper()} Learning Curve")
    def show_video(self):
        agent1 = tk.simpledialog.askstring("Input", "Enter player 1 agent type (e.g., dqn):", parent=self)
        if not agent1: return
        agent2 = tk.simpledialog.askstring("Input", "Enter player 2 agent type (e.g., alphabeta):", parent=self)
        if not agent2: return
        filename = f"{agent1.lower()}_vs_{agent2.lower()}.gif"
        if not os.path.exists(filename): messagebox.showerror("File Not Found", f"Video file '{filename}' not found. Please play and record the match first."); return
        VideoPlayer(self, filename, "AI Match Replay")
    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"): self.destroy()

class TrainWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master); self.title("Train Agents"); self.geometry("450x350")
        ttk.Label(self, text="Select Agent to Train:", font=('Arial', 12, 'bold')).pack(pady=10)
        self.agent_combo = ttk.Combobox(self, values=["DQN", "MCTS"], state="readonly"); self.agent_combo.set("DQN"); self.agent_combo.pack(pady=5)
        ttk.Label(self, text="Number of Episodes:").pack(pady=5)
        self.episodes_entry = ttk.Entry(self); self.episodes_entry.insert(0, "5000"); self.episodes_entry.pack(pady=5)
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=300, mode='determinate'); self.progress.pack(pady=20)
        self.train_button = ttk.Button(self, text="Start Training", command=self.start_training); self.train_button.pack(pady=10)
        self.log_text = tk.Text(self, height=5, state='disabled', wrap=tk.WORD); self.log_text.pack(pady=5, fill=tk.X, expand=True, padx=10)
    def log(self, message):
        self.log_text.config(state='normal'); self.log_text.insert(tk.END, message + '\n'); self.log_text.see(tk.END); self.log_text.config(state='disabled'); self.update_idletasks()
    def start_training(self):
        agent_type = self.agent_combo.get()
        try: num_episodes = int(self.episodes_entry.get())
        except ValueError: messagebox.showerror("Invalid Input", "Number of episodes must be an integer."); return
        self.train_button.config(state=tk.DISABLED)
        threading.Thread(target=self.training_thread, args=(agent_type, num_episodes), daemon=True).start()
    def training_thread(self, agent_type_str, num_episodes):
        self.log(f"--- Training {agent_type_str} Agent ---")
        agent = DQNAgent(player=1) if agent_type_str == 'DQN' else MCTSAgent(player=1, n_sim=50)
        history = train(agent, num_episodes, eval_interval=100)
        model_path = f"{agent_type_str.lower()}_othello.pth"; agent.save(model_path)
        self.log(f"Model saved to {model_path}")
        plot_path = f"{agent_type_str.lower()}_learning_curve.png"; plot_metrics(history, plot_path)
        self.log(f"Plot saved to {plot_path}"); self.log("Training complete!")
        self.train_button.config(state=tk.NORMAL)

class ImageViewer(tk.Toplevel):
    def __init__(self, master, image_path, title):
        super().__init__(master); self.title(title)
        img = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(img)
        ttk.Label(self, image=self.photo).pack(padx=10, pady=10)

class VideoPlayer(tk.Toplevel):
    def __init__(self, master, video_path, title):
        super().__init__(master); self.title(title); self.protocol("WM_DELETE_WINDOW", self.close)
        self.video = imageio.get_reader(video_path); self.delay = int(1000 / self.video.get_meta_data()['fps'])
        self.image_label = ttk.Label(self); self.image_label.pack()
        self._is_running = True
        self.thread = threading.Thread(target=self.stream, daemon=True)
        self.thread.start()
    def stream(self):
        while self._is_running:
            for image in self.video:
                if not self._is_running: break
                frame_image = ImageTk.PhotoImage(Image.fromarray(image))
                self.image_label.config(image=frame_image)
                self.image_label.image = frame_image
                time.sleep(self.delay / 1000)
    def close(self):
        self._is_running = False
        self.destroy()

if __name__ == '__main__':
    # GUIを起動するこちらの2行の#を付けて、無効にする
    # app = OthelloGUI()
    # app.mainloop()

    # トレーニング用のコードのコメントアウト有効にする
    print("GUIを無効化し、トレーニングモードで起動する。")

    # 1. 学習させるエージェントを選択 ('DQN' or 'MCTS')
    AGENT_TO_TRAIN = 'MCTS'  # ← ここを 'MCTS' に変更
    NUM_EPISODES = 10000     # 学習するゲームの回数
    EVAL_INTERVAL = 200      # 途中で性能を評価する間隔
    MODEL_SAVE_PATH = f"{AGENT_TO_TRAIN.lower()}_othello.pth"
    PLOT_SAVE_PATH = f"{AGENT_TO_TRAIN.lower()}_learning_curve.png"

    # 2. エージェントを作成
    if AGENT_TO_TRAIN == 'DQN':
        agent = DQNAgent(player=1, learning=True)
    elif AGENT_TO_TRAIN == 'MCTS':
        # n_simは1手あたりの探索回数で、大きいほど強くなるが学習時間が長くなる
        agent = MCTSAgent(player=1, n_sim=50, learning=True)
    else:
        raise ValueError("無効なエージェントタイプが指定されました。")

    print(f"{AGENT_TO_TRAIN}エージェントの学習を{NUM_EPISODES}エピソード開始します...")

    # 3. 学習プロセスを実行
    training_history = train(agent, num_episodes=NUM_EPISODES, eval_interval=EVAL_INTERVAL)

    # 4. 学習済みモデルと性能グラフを保存
    agent.save(MODEL_SAVE_PATH)
    print(f"学習が完了しました。モデルを {MODEL_SAVE_PATH} に保存しました。")

    plot_metrics(training_history, PLOT_SAVE_PATH)
    print(f"学習曲線のグラフを {PLOT_SAVE_PATH} に保存")
