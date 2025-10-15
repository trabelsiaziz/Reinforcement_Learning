import gymnasium as gym
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import ale_py
import os


# ball position + paddle position + ball velocity
# 16 * 2 + 2 * 2 + 2 * 2 = 40 dimensional input to the network
# 16 for the paddle position (y-coordinates)
# 2 for the ball position (x and y coordinates)
# 2 for the ball velocity (x and y components)
# output = 3 (stay, up, down)

Episodes_number = 1000
gamma = 0.99
lr = 0.01
batch_size = 1


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(6, 200)  
        self.fc2 = nn.Linear(200, 3)         

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)          
        return F.softmax(x, dim=-1)  



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
                ball_pos.append(torch.tensor([i,j]).float())
            elif obs[i,j] == 92:
                agent_paddle.append(torch.tensor([i,j]).float())
    
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

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)



# if starting from scratch

# agent = Agent().to(device=device)
# optimizer = optim.Adam(agent.parameters(), lr=lr)

# if loading from checkpoint

checkpoint = torch.load("checkpoints/pong_ep700.pth", map_location=device)
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
int_to_action = {0:0, 1:2, 2:3} # 0: stay, 1: up, 2: down


for episode in range(start_episode, end_episode):
    obs, _ = env.reset()
    ball_pos_prev = torch.zeros(2,2).float().to(device)
    done = False
    Actions = []
    Rewards = []
    States = []

    if episode%5 == 0 : print("EPISODE #", episode)

    while not done : 
        frame = preprocess(obs)
        agent_paddle, ball_pos = get_positions(frame)
        # env.render()
        fps = env.metadata['render_fps']
        
        velocity = (ball_pos - ball_pos_prev)/fps
        # print("velocity is : ", velocity)
        bx = torch.mean(ball_pos[:,0]) 
        by = torch.mean(ball_pos[:,1])
        vx = torch.mean(velocity[:,0])
        vy = torch.mean(velocity[:,1])
        px = torch.mean(agent_paddle[:,0])
        py = torch.mean(agent_paddle[:,1])
    
        ball_pos_prev = ball_pos
        input = torch.tensor([bx, by, px, py, vx, vy], dtype=torch.float, device=device)
        # print("the input shape is : " , input.shape)
        probs = agent(input)
        # print("the probs are : ", probs)
        dist = Categorical(probs)
        action = dist.sample().item()
        obs, reward, terminated, truncated, info = env.step(int_to_action[action])
        done = truncated or terminated
        # print("the reward is : ", reward)
        States.append(input)
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
    

    if episode % 100 == 0:
        checkpoint_path = f"checkpoints/pong_ep{episode}.pth"
        torch.save({
            'episode': episode,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
    

    



