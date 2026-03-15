"""
Simple Training Script for InvertedPendulum-v5
===============================================
Minimal script to get started quickly on macOS.

Usage:
    python simple_train.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from pathlib import Path

# Configuration
TOTAL_TIMESTEPS = 10_00_0  # Adjust this for longer/shorter training
EVAL_FREQ = 100
N_EVAL_EPISODES = 10

# Setup directories (will be created in your current directory)
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

print("="*70)
print("InvertedPendulum-v5 Simple Training Script")
print("="*70)
print(f"Working directory: {BASE_DIR}")
print(f"Models will be saved to: {MODELS_DIR}")
print(f"Logs will be saved to: {LOGS_DIR}")

# Create environment with reset_noise_scale=0.1 as specified
print("\nCreating environment...")
env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Create evaluation environment
eval_env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

print("✓ Environment created")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

# Setup evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(MODELS_DIR),
    log_path=str(LOGS_DIR),
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    verbose=1
)

# Create PPO model
print("\nCreating PPO model with hyperparameters:")
print("  Learning rate: 3e-4")
print("  N steps: 2048")
print("  Batch size: 64")
print("  N epochs: 10")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=str(LOGS_DIR)
)

print("\n" + "="*70)
print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps")
print("="*70)
print("\nTraining progress:")

# Train the model
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True
)

# Save final model
print("\n" + "="*70)
print("Training Complete!")
print("="*70)

model.save(str(MODELS_DIR / 'final_model'))
env.save(str(MODELS_DIR / 'vec_normalize.pkl'))


print(f"\n✓ Final model saved to: {MODELS_DIR / 'final_model.zip'}")
print(f"✓ Best model saved to: {MODELS_DIR / 'best_model.zip'}")
print(f"✓ Normalization stats: {MODELS_DIR / 'vec_normalize.pkl'}")

# Test the trained model
print("\n" + "="*70)
print("Testing Trained Model")
print("="*70)

# Load best model
best_model = PPO.load(str(MODELS_DIR / 'best_model'))

# Create test environment
test_env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
test_env = Monitor(test_env)
test_env = DummyVecEnv([lambda: test_env])
test_env = VecNormalize.load(str(MODELS_DIR / 'vec_normalize.pkl'), test_env)
test_env.training = False
test_env.norm_reward = False

# Run test episodes
print("\nRunning 10 test episodes...")
test_rewards = []



for episode in range(10):
    obs = test_env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, _ = test_env.step(action)
        episode_reward += reward[0]
    
    test_rewards.append(episode_reward)
    print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")

print("\n" + "-"*70)
print("Test Results:")
print(f"  Mean reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
print(f"  Min reward: {np.min(test_rewards):.2f}")
print(f"  Max reward: {np.max(test_rewards):.2f}")
print("-"*70)

test_env.close()
env.close()

print("\n✅ All done!")
print(f"\nTo visualize training with TensorBoard, run:")
print(f"  tensorboard --logdir={LOGS_DIR}")
print(f"\nThen open: http://localhost:6006")