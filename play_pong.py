import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ale_py
from gymnasium.wrappers import RecordVideo
import os

# --- Define the same Agent architecture ---
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(80 * 80, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# --- Preprocessing function (same as training) ---
def preprocess(obs, device):
    I = obs
    I = I[35:195]         # crop
    I = I[::2, ::2, 0]    # downsample by factor of 2
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    I = torch.tensor(I, dtype=torch.float32, device=device)
    return I.view(80 * 80)

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# --- Load model checkpoint ---
checkpoint_path = "checkpoints/pong_ep1800.pth"  # change to the one you want
agent = Agent().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
agent.load_state_dict(checkpoint["model_state_dict"])
agent.eval()
print(f"🎮 Loaded model from: {checkpoint_path}")
random_agent = Agent().to(device)  # for random play

# --- Initialize environment ---

# this is for video recording
os.makedirs("videos", exist_ok=True)
env = gym.make("ALE/Pong-v5", render_mode="rgb_array") 
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)

# env = gym.make("ALE/Pong-v5", render_mode="human")  
obs, info = env.reset()

f_prev = torch.tensor(0, device=device)
done = False

# --- Play one full episode ---
while not done:
    f_cur = preprocess(obs, device)
    f_in = f_cur - f_prev
    f_prev = f_cur

    with torch.no_grad():
        # Trained agent
        probs = agent(f_in)
        
        # Random play
        # probs = random_agent(f_in)
        
        # Stochastic play (sample) — or use argmax for deterministic
        m = Categorical(probs)
        action = m.sample().item()
        # print("action :", action)
        # print("probs :", probs)

    # 2 = UP, 3 = DOWN (like in training)
    obs, reward, terminated, truncated, info = env.step(2 if action == 0 else 3)
    done = terminated or truncated

env.close()
print("✅ Game finished!")
