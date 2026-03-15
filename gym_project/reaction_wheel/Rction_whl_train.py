"""
Training Script for Custom Reaction Wheel Pendulum
==================================================
PARALLELIZED VERSION - Uses multiple CPU cores for faster training!
Same accuracy, much faster speed on M1 MacBook Air.

Usage:
    python train_reaction_wheel.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from Rction_whl_env_best import ReactionWheelPendulumEnv

# ============= PARALLEL CONFIGURATION =============
# M1 MacBook Air has 8 cores - use 4 for training (safe and stable)
N_ENVS = 4  # Number of parallel environments

# Configuration - SAME AS ORIGINAL (full accuracy)
TOTAL_TIMESTEPS = 500_000  # Full 1M timesteps
EVAL_FREQ = 10000
N_EVAL_EPISODES = 10

# Setup directories
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / 'models_reaction_wheel_v2'
LOGS_DIR = BASE_DIR / 'logs_reaction_wheel_v2'


def make_env():
    """Helper function to create environment"""
    def _init():
        env = ReactionWheelPendulumEnv(reset_noise_scale=0.15, render_mode=None)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=1000)
        return env
    return _init


# CRITICAL: This guard is required for multiprocessing on macOS
if __name__ == '__main__':
    
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("Reaction Wheel Pendulum - PARALLELIZED Training")
    print("="*70)
    print(f"🚀 Using {N_ENVS} parallel environments for speed boost!")
    print(f"Working directory: {BASE_DIR}")
    print(f"Models will be saved to: {MODELS_DIR}")
    print(f"Logs will be saved to: {LOGS_DIR}")

    # Create training environment
    print("\nCreating parallel training environments...")

    # PARALLEL MAGIC: SubprocVecEnv runs environments in separate processes
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Create evaluation environment (single env is fine for eval)
    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    print(f"✓ Created {N_ENVS} parallel environments")
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

    # Create PPO model - SAME HYPERPARAMETERS AS ORIGINAL
    print("\nCreating PPO model with hyperparameters:")
    print("  Learning rate: 3e-4")
    print("  N steps: 2048")
    print("  Batch size: 64")
    print("  N epochs: 15")

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
    print(f"Speed boost: ~{N_ENVS}x faster with {N_ENVS} parallel envs!")
    print(f"Estimated time: ~20-25 minutes (vs ~90 min without parallelization)")
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

    # Create test environment (single env for testing)
    def make_test_env():
        env = ReactionWheelPendulumEnv(reset_noise_scale=0.15, render_mode=None)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=1000)
        return env

    test_env = SubprocVecEnv([lambda: make_test_env()])
    test_env = VecNormalize.load(str(MODELS_DIR / 'vec_normalize.pkl'), test_env)
    test_env.training = False
    test_env.norm_reward = False

    # Run test episodes
    print("\nRunning 10 test episodes...")
    test_rewards = []
    test_lengths = []

    for episode in range(10):
        obs = test_env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, done, _ = test_env.step(action)
            episode_reward += reward[0]
            steps += 1
        
        test_rewards.append(episode_reward)
        test_lengths.append(steps)
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    print("\n" + "-"*70)
    print("Test Results:")
    print(f"  Mean reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Mean episode length: {np.mean(test_lengths):.1f} steps")
    print(f"  Min reward: {np.min(test_rewards):.2f}")
    print(f"  Max reward: {np.max(test_rewards):.2f}")
    print("-"*70)

    test_env.close()
    env.close()

    print("\n✅ All done!")
    print(f"\nTo visualize training with TensorBoard, run:")
    print(f"  tensorboard --logdir={LOGS_DIR}")
    print(f"\nThen open: http://localhost:6006")
    print(f"\n💡 You used {N_ENVS} parallel environments. Adjust N_ENVS (2-8) for different speed!")