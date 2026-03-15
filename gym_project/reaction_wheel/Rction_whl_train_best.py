"""
Training Script for Custom Reaction Wheel Pendulum
==================================================
IMPROVED VERSION - With LR decay and entropy regularization
Prevents catastrophic forgetting and ensures stable long-term training.

Usage:
    python train_reaction_wheel.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from gymnasium.wrappers import TimeLimit
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from Rction_whl_env_best import ReactionWheelPendulumEnv

# ============= PARALLEL CONFIGURATION =============
N_ENVS = 6  # Number of parallel environments

# Configuration
TOTAL_TIMESTEPS = 7_50_000  # Increased to 1M for better convergence
EVAL_FREQ = 10000
N_EVAL_EPISODES = 10

# Setup directories
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / 'models_reaction_wheel_new2'  # New version
LOGS_DIR = BASE_DIR / 'logs_reaction_wheel_new2'


def make_env():
    """Helper function to create environment"""
    def _init():
        env = ReactionWheelPendulumEnv(reset_noise_scale=0.15, render_mode=None)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=1000)
        return env
    return _init


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        final_value: Final learning rate (default: 0.0 means 10% of initial)
    
    Returns:
        schedule function that computes current learning rate
    """
    if final_value == 0.0:
        final_value = initial_value * 0.05  # End at 5% of initial value
    
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        We want LR to decrease from initial_value to final_value.
        """
        return final_value + (initial_value - final_value) * progress_remaining
    
    return func


# CRITICAL: This guard is required for multiprocessing on macOS
if __name__ == '__main__':
    
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("Reaction Wheel Pendulum - IMPROVED Training")
    print("="*70)
    print(f"🚀 Using {N_ENVS} parallel environments")
    print(f"📚 Learning rate: 3e-4 → 3e-5 (linear decay)")
    print(f"🎲 Entropy coefficient: 0.01 (prevents premature convergence)")
    print(f"⏱️  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Working directory: {BASE_DIR}")
    print(f"Models will be saved to: {MODELS_DIR}")
    print(f"Logs will be saved to: {LOGS_DIR}")

    # Create training environment
    print("\nCreating parallel training environments...")
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    print(f"✓ Created {N_ENVS} parallel environments")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Setup early stopping callback
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15,  # Stop if no improvement for 15 evals (150k steps)
        min_evals=20,  # Start checking after 20 evals (200k steps)
        verbose=1
    )

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(LOGS_DIR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        callback_after_eval=stop_callback,
        verbose=1
    )

    # Create PPO model with IMPROVED hyperparameters
    print("\nCreating PPO model with improved hyperparameters:")
    print("  Learning rate: 3e-4 → 3e-5 (linear decay)")
    print("  Entropy coefficient: 0.01 (maintains exploration)")
    print("  N steps: 2048")
    print("  Batch size: 64")
    print("  N epochs: 10")
    print("  Clip range: 0.2")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate= 3e-4, 
        n_steps=2048,
        batch_size=256,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=0.01,  # Entropy bonus - prevents premature convergence
        vf_coef=0.2,     # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping for stability
        verbose=1,
        tensorboard_log=str(LOGS_DIR)
    )

    print("\n" + "="*70)
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps")
    print("Improvements over previous version:")
    print("  ✓ Learning rate decay prevents late-stage instability")
    print("  ✓ Entropy regularization maintains exploration")
    print("  ✓ Early stopping prevents catastrophic forgetting")
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
    print("Testing Best Model")
    print("="*70)

    # Load best model
    best_model = PPO.load(str(MODELS_DIR / 'best_model'))

    # Create test environment
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
    
    # Success criteria
    avg_length = np.mean(test_lengths)
    if avg_length > 900:
        print(f"\n🎉 SUCCESS! Average episode length {avg_length:.0f} > 900 steps")
        print("   The pendulum is reliably balanced!")
    elif avg_length > 600:
        print(f"\n✓ GOOD! Average episode length {avg_length:.0f} > 600 steps")
        print("  The pendulum can balance, but may need more training for consistency")
    else:
        print(f"\n⚠️  NEEDS MORE TRAINING: Average length {avg_length:.0f} < 600 steps")
        print("   Consider training longer or adjusting hyperparameters")
    
    print("-"*70)

    test_env.close()
    env.close()

    print("\n✅ All done!")
    print(f"\nTo visualize training with TensorBoard, run:")
    print(f"  tensorboard --logdir={LOGS_DIR}")
    print(f"\nThen open: http://localhost:6006")
    
    print(f"\n💡 Training tips:")
    print(f"  - Check TensorBoard for 'eval/mean_ep_length' reaching 800-1000")
    print(f"  - Look for 'train/learning_rate' smoothly decreasing")
    print(f"  - Monitor 'train/entropy_loss' staying negative (good exploration)")