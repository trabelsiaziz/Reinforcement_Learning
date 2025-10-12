# Reinforcement Learning - Pong Game

A reinforcement learning project that trains an AI agent to play Atari Pong using Policy Gradient methods.

## 🎮 Demo

<video width="640" height="480" controls>
  <source src="videos/rl-video-episode-0.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 📋 Project Overview

This project implements a Policy Gradient reinforcement learning algorithm to train an agent to play the classic Atari Pong game. The agent learns through trial and error, gradually improving its gameplay performance.

## ⚡ Key Features

- **Policy Gradient Algorithm**: Uses REINFORCE algorithm for training
- **Neural Network Architecture**: Simple MLP network with 1 hidden layer
- **Atari Environment**: Built with OpenAI Gymnasium and ALE (Arcade Learning Environment)
- **Video Recording**: Captures gameplay videos for evaluation
- **Model Checkpointing**: Saves trained models for later use

## 🛠️ Files Structure

- `train_pong.py` - Main training script with Policy Gradient implementation
- `play_pong.py` - Script to play/test the trained agent
- `checkpoints/` - Directory containing saved model weights
- `videos/` - Directory containing recorded gameplay videos

## 🚀 Training Details

- **Hardware**: Trained on NVIDIA T4 GPU in Google Colab
- **Training Duration**: 4+ hours for 1800 episodes
- **Environment**: Atari Pong 
- **Preprocessing**: Frame downsampling to 80x80 grayscale images
- **Action Space**: 2 actions (UP, DOWN)

## 🎯 Results

The agent successfully learned to play Pong after extensive training. Check out the gameplay video above to see the trained agent (the player with green paddle) in action!

## 🔧 Usage

### Training
```bash
python train_pong.py
```

### Testing/Playing
```bash
python play_pong.py
```

## 📦 Dependencies

- PyTorch
- OpenAI Gymnasium
- ALE-py
- NumPy

## 🏆 Performance

The model checkpoint `pong_ep1800.pth` represents the agent's learned policy after 1800 training episodes, demonstrating competitive gameplay against the built-in AI opponent.