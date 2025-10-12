import gymnasium as gym
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import ale_py

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(80 * 80, 200)  
        self.fc2 = nn.Linear(200, 2)         

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)          
        return F.softmax(x, dim=-1)  



def preprocess(obs):
    I = obs
    I = I[35:195]  
    I = I[::2, ::2, 0]
    I[I == 144] = 0 
    I[I == 109] = 0 
    I[I != 0] = 1   
    I = torch.tensor(I)
    I = I.view(80*80)
    I = I.float().to(device=device)
    return I

import os
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)


Episodes_number = 1000
gamma = 0.99
lr = 0.01
batch_size = 10

# if starting from scratch

# agent = Agent().to(device=device)
# optimizer = optim.Adam(agent.parameters(), lr=lr)

# if loading from checkpoint

checkpoint = torch.load("checkpoints/pong_ep1000.pth", map_location=device)
agent = Agent().to(device=device)
agent.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.Adam(agent.parameters(), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_episode = checkpoint['episode'] + 1


# start_episode = 1
end_episode = Episodes_number + start_episode + 1

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
observation, info = env.reset()

# buffers 
G_buffer = []
logs_buffer = []


for episode in range(start_episode, end_episode):
    obs, _ = env.reset()
    f_prev = torch.tensor(0).float()
    done = False
    Actions = []
    Rewards = []
    States = []

    if episode%5 == 0 : print("EPISODE #", episode)

    while not done : 
        f_cur = preprocess(obs)
        env.render()
        f_in = f_cur - f_prev
        f_prev = f_cur
        probs = agent(f_in)
        dist = Categorical(probs)
        action = dist.sample().item()
        obs, reward, terminated, truncated, info = env.step(2 if action == 0 else 3)
        done = truncated or terminated
        # print("the reward is : ", reward)
        States.append(f_in)
        Actions.append(torch.tensor(action, dtype=torch.int, device=device))
        Rewards.append(reward)

    # DiscountedReturns = []
    # for t in range(len(Rewards)):
    #     G = 0.0
    #     for k, r in enumerate(Rewards[t:]):
    #         G += (gamma**k)*r
    #     DiscountedReturns.append(G)



    G = 0
    DiscountedReturns = []
    for r in reversed(Rewards):
        G = r + gamma * G
        DiscountedReturns.insert(0, G)
    DiscountedReturns = torch.tensor(DiscountedReturns, dtype=torch.float, device=device)

    DiscountedReturns = (DiscountedReturns - DiscountedReturns.mean()) / DiscountedReturns.std()

    for State, Action, G in zip(States, Actions, DiscountedReturns):
        probs = agent(State)
        dist = torch.distributions.Categorical(probs=probs)    
        log_prob = dist.log_prob(Action)
        logs_buffer.append(log_prob)
        G_buffer.append(G)
        
    

    # loss = log_prob*G # this is previously (- log_prob*G)

    # tnajem toptimizi here
    # entropy = -(probs * probs.log()).sum()
    # loss = loss - 0.01 * entropy
    
    # upsate kol episode did not work 
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    
    if episode % batch_size == 0:
        optimizer.zero_grad()
        G_buffer = torch.stack(G_buffer)
        logs_buffer = torch.stack(logs_buffer)
        loss = -(logs_buffer * G_buffer).mean()
        loss.backward()
        optimizer.step()
        G_buffer = []
        logs_buffer = []
    

    if episode % 25 == 0:
        checkpoint_path = f"checkpoints/pong_ep{episode}.pth"
        torch.save({
            'episode': episode,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
    

    



