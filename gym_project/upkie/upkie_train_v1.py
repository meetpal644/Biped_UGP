#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for Upkie 6-DOF Balancing
=========================================

Floating-base wheeled humanoid balancing using PPO.

Usage:
    python train_upkie_balance.py
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from gymnasium.wrappers import TimeLimit

# Import your environment
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from upkie_env_gauss import UpkieBalanceEnv


# ============================================================
# CONFIGURATION
# ============================================================

N_ENVS = 6                     # Parallel environments
TOTAL_TIMESTEPS = 5_000_000    # Needs more than pendulum
EVAL_FREQ = 20000
N_EVAL_EPISODES = 5

BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "models_upkie_upright_5M"
LOGS_DIR = BASE_DIR / "logs_upkie_upright_5M"


# ============================================================
# ENV CREATION
# ============================================================

def make_env():
    def _init():
        env = UpkieBalanceEnv(reset_noise_scale=0.075, render_mode=None)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=2000)
        return env
    return _init


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def linear_schedule(initial_value: float, final_ratio: float = 0.333):
    final_value = initial_value * final_ratio

    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Upkie 6-DOF Balancing - PPO Training")
    print("=" * 70)

    print(f"🚀 Parallel environments: {N_ENVS}")
    print(f"⏱️  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"📁 Models: {MODELS_DIR}")
    print(f"📊 Logs: {LOGS_DIR}")

    # -----------------------------
    # Training Environment
    # -----------------------------
    print("\nCreating training environments...")
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # -----------------------------
    # Evaluation Environment
    # -----------------------------
    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # ============================================================
    # CALLBACKS
    # ============================================================

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15,
        min_evals=50,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(LOGS_DIR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    # ============================================================
    # PPO MODEL
    # ============================================================

    print("\nCreating PPO model...")

    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],     # Larger than pendulum
            vf=[256, 256]    # Stronger critic
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate= 2e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=7,
        gamma=0.998,            # Longer horizon than pendulum
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,         # Mild entropy
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(LOGS_DIR),
    )

    print("\nStarting training...")
    print("=" * 70)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    print("\nTraining complete.")
    model.save(str(MODELS_DIR / "final_model"))
    env.save(str(MODELS_DIR / "vec_normalize.pkl"))

    print(f"✓ Final model saved.")
    print(f"✓ Best model saved.")
    print(f"✓ Normalization stats saved.")

    # ============================================================
    # TEST BEST MODEL
    # ============================================================

    print("\nTesting best model...")

    best_model = PPO.load(str(MODELS_DIR / "best_model"))

    def make_test_env():
        env = UpkieBalanceEnv(reset_noise_scale=0.05, render_mode=None)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=2000)
        return env

    test_env = SubprocVecEnv([lambda: make_test_env()])
    test_env = VecNormalize.load(str(MODELS_DIR / "vec_normalize.pkl"), test_env)
    test_env.training = False
    test_env.norm_reward = False

    test_rewards = []
    test_lengths = []

    for episode in range(5):
        obs = test_env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, done, _ = test_env.step(action)
            ep_reward += reward[0]
            steps += 1

        test_rewards.append(ep_reward)
        test_lengths.append(steps)

        print(f"Episode {episode+1}: Reward={ep_reward:.2f}, Steps={steps}")

    print("\nTest Summary")
    print("-" * 50)
    print(f"Mean reward: {np.mean(test_rewards):.2f}")
    print(f"Mean episode length: {np.mean(test_lengths):.1f}")

    if np.mean(test_lengths) > 1500:
        print("\n🎉 Excellent balance stability!")
    elif np.mean(test_lengths) > 1000:
        print("\n✓ Good balance achieved.")
    else:
        print("\n⚠️ Needs more training or tuning.")

    test_env.close()
    env.close()

    print("\nDone.")