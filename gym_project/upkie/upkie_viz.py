#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-Time Visualization of Trained Upkie Balancing Agent
========================================================

No video recording. Pure live MuJoCo viewer.
"""

import numpy as np
import mujoco
import sys
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import Wrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "models_upkie_upright_5M"

DISABLE_TERMINATION = True
MAX_STEPS = 5000

# ---- Initial perturbations ----
INITIAL_PITCH_DEG = 25
INITIAL_PITCH_RATE = 0.0
INITIAL_FORWARD_VEL = 0.0


# ============================================================
# Termination Override
# ============================================================

class NoTerminationWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, False, info


# ============================================================
# Import environment
# ============================================================

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from upkie_env_gauss import UpkieBalanceEnv


# ============================================================
# Load Model
# ============================================================

print("=" * 70)
print("Upkie Real-Time Visualization")
print("=" * 70)

best_model_path = MODELS_DIR / "best_model.zip"
if not best_model_path.exists():
    print("❌ No trained model found.")
    exit(1)

model = PPO.load(str(best_model_path))
print("✓ Model loaded")


# ============================================================
# Create Human-Render Environment
# ============================================================

def make_env():
    env = UpkieBalanceEnv(
        reset_noise_scale=0.0,
        render_mode="human"  # IMPORTANT
    )
    if DISABLE_TERMINATION:
        env = NoTerminationWrapper(env)
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(str(MODELS_DIR / "vec_normalize.pkl"), env)
vec_env.training = False
vec_env.norm_reward = False

print("✓ Environment created (human render mode)")


# ============================================================
# Reset + Apply Custom Initial State
# ============================================================

vec_env.reset()

# unwrap to actual MuJoCo env
actual_env = vec_env.venv.envs[0]
while hasattr(actual_env, "env"):
    actual_env = actual_env.env

pitch_rad = np.deg2rad(INITIAL_PITCH_DEG)

quat = np.zeros(4)
mujoco.mju_axisAngle2Quat(
    quat,
    np.array([0, 1, 0], dtype=np.float64),
    pitch_rad
)

qpos = actual_env.data.qpos.copy()
qvel = actual_env.data.qvel.copy()

# set base orientation
qpos[3:7] = quat

# set base velocity
qvel[0] = INITIAL_FORWARD_VEL
qvel[3:6] = np.array([0, INITIAL_PITCH_RATE, 0])

actual_env.set_state(qpos, qvel)

obs = vec_env.normalize_obs(actual_env._get_obs().reshape(1, -1))

print(f"\nInitial pitch: {INITIAL_PITCH_DEG} deg")
print("Starting simulation...\n")


# ============================================================
# Real-Time Control Loop
# ============================================================

step = 0
total_reward = 0

try:
    while step < MAX_STEPS:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        step += 1

        # Explicit render call (important)
        actual_env.render()

        # Optional: control playback speed
        time.sleep(0.005)

        if step % 100 == 0:
            torso_id = actual_env.model.body("torso").id
            com_height = actual_env.data.subtree_com[torso_id][2]
            quat = actual_env.data.qpos[3:7]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            R = R.reshape(3, 3)

            g_body = R.T @ np.array([0, 0, -1])
            pitch = np.degrees(np.arcsin(g_body[0]))

            print(f"Step {step:4d} | Pitch: {pitch:+6.2f}°\n")
            print(f"COM Height: {com_height:.4f} m \n")
            print(f"gravity in body frame of the bot's z axis: {g_body[2]*9.81:.4f} m/s^2 \n")
        if done[0] and not DISABLE_TERMINATION:
            print("Terminated.")
            break

except KeyboardInterrupt:
    print("\nStopped by user")


# ============================================================
# Final Report
# ============================================================

quat = actual_env.data.qpos[3:7]
R = np.zeros(9)
mujoco.mju_quat2Mat(R, quat)
R = R.reshape(3, 3)

g_body = R.T @ np.array([0, 0, -1])
final_pitch = np.degrees(np.arcsin(g_body[0]))

print("\n" + "=" * 70)
print(f"Steps: {step}")
print(f"Total reward: {total_reward:.2f}")
print(f"Final pitch: {final_pitch:.2f}°")
print("=" * 70)

vec_env.close()