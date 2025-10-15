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
        self.fc1 = nn.Linear(6, 200)  
        self.fc2 = nn.Linear(200, 3)         

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)          
        return F.softmax(x, dim=-1)  

# --- Preprocessing function (same as training) ---
D = 80

def preprocess(obs):
    I = obs
    I = I[34:194]  
    I = I[::2, ::2, 0]
    I[I == 144] = 0 
    I[I == 109] = 0  
    return I



def get_positions(obs):
    agent_paddle = []
    ball_pos = []
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i,j] == 236:
                ball_pos.append(torch.tensor([i/D,j/D]).float())
            elif obs[i,j] == 92:
                agent_paddle.append(torch.tensor([i/D,j/D]).float())
    
    while len(ball_pos) < 2:
        ball_pos.append(torch.tensor([0,0]).float())
    if len(ball_pos) > 2:
        ball_pos = ball_pos[:2]
    while len(agent_paddle) < 16:
        agent_paddle.append(torch.tensor([0,0]).float())
    agent_paddle = torch.stack(agent_paddle)
    ball_pos = torch.stack(ball_pos).float()
    agent_paddle = agent_paddle.to(device)
    ball_pos = ball_pos.to(device) 
    return agent_paddle, ball_pos

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# --- Load model checkpoint ---
checkpoint_path = "checkpoints/pong_ep1500.pth"  # change to the one you want
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

int_to_action = {0:0, 1:2, 2:3} # 0: stay, 1: up, 2: down

ball_pos_prev = torch.tensor(0).float().to(device)
done = False

# --- Play one full episode ---
while not done:
    frame = preprocess(obs)
    agent_paddle, ball_pos = get_positions(frame)

    with torch.no_grad():
        # Trained agent
        fps = env.metadata['render_fps']
        velocity = (ball_pos - ball_pos_prev)/fps

        bx = torch.mean(ball_pos[:,0]) 
        by = torch.mean(ball_pos[:,1])
        vx = torch.mean(velocity[:,0])
        vy = torch.mean(velocity[:,1])
        px = torch.mean(agent_paddle[:,0])
        py = torch.mean(agent_paddle[:,1])

        ball_pos_prev = ball_pos
        input = torch.tensor([bx, by, px, py, vx, vy], dtype=torch.float, device=device)
        probs = agent(input)

        # Random play
        # probs = random_agent(f_in)
        
        # Stochastic play (sample) — or use argmax for deterministic
        dist = Categorical(probs)
        action = dist.sample().item()
        action = int_to_action[action]
    
       

    obs, reward, terminated, truncated, info = env.step(action)
    # print("reward is : ", reward)
    done = terminated or truncated

env.close()
print("✅ Game finished!")
