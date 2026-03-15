#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Continue Training Script for Upkie 6-DOF Balancing
==================================================

Loads previously trained PPO model and VecNormalize statistics,
then continues training with smaller learning rate for stabilization.

Usage:
    python continue_upkie_training.py
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

# Import environment
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from gym_project.upkie.upkie_env_gauss import UpkieBalanceEnv


# ============================================================
# CONFIGURATION
# ============================================================

N_ENVS = 6
CONTINUE_TIMESTEPS = 1_500_000
EVAL_FREQ = 20000
N_EVAL_EPISODES = 5

BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "models_upkie"
LOGS_DIR = BASE_DIR / "logs_upkie"


# ============================================================
# ENV CREATION
# ============================================================

def make_env():
    def _init():
        env = UpkieBalanceEnv(reset_noise_scale=0.05, render_mode=None)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=2000)
        return env
    return _init


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Upkie PPO - CONTINUED TRAINING (Stabilization Phase)")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load Training Environment + Normalization
    # ------------------------------------------------------------

    print("\nLoading training environments...")

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize.load(str(MODELS_DIR / "vec_normalize.pkl"), env)

    env.training = True
    env.norm_reward = True

    # ------------------------------------------------------------
    # Load Evaluation Environment
    # ------------------------------------------------------------

    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecNormalize.load(str(MODELS_DIR / "vec_normalize.pkl"), eval_env)

    eval_env.training = False
    eval_env.norm_reward = False

    print("✓ Normalization statistics loaded")

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=10,
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

    # ------------------------------------------------------------
    # Load Model
    # ------------------------------------------------------------

    print("\nLoading best model...")
    model = PPO.load(
        str(MODELS_DIR / "best_model"),
        env=env,
        tensorboard_log=str(LOGS_DIR),
    )

    print("✓ Model loaded")

    # ------------------------------------------------------------
    # Stabilization Hyperparameters
    # ------------------------------------------------------------
    model.learning_rate = lambda _: 5e-5
    model.clip_range = lambda _: 0.1
    model.ent_coef = 0.002

    print("\nStabilization hyperparameters:")
    print("  Learning rate: 5e-5")
    print("  Clip range: 0.1")
    print("  Entropy coef: 0.002")

    print("\nContinuing training for", CONTINUE_TIMESTEPS, "steps")
    print("=" * 70)

    model.learn(
        total_timesteps=CONTINUE_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=False,  # CRITICAL
    )

    # ------------------------------------------------------------
    # Save Updated Model
    # ------------------------------------------------------------

    print("\nSaving updated model...")
    model.save(str(MODELS_DIR / "final_model_continued"))
    env.save(str(MODELS_DIR / "vec_normalize.pkl"))

    print("✓ Updated model saved")
    print("✓ Normalization stats updated")

    # ------------------------------------------------------------
    # Testing Phase
    # ------------------------------------------------------------

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
        print("\n⚠️ Needs more stabilization.")

    test_env.close()
    env.close()

    print("\nDone.")