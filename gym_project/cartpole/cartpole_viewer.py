"""
Test Trained Model with Custom Initial Conditions - CORRECTLY READING STATE
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path

# ============= CONFIGURATION =============
BASE_DIRECTORY = Path.cwd()
MODEL_PATH = BASE_DIRECTORY / "models" / "best_model.zip"
VEC_NORMALIZE_PATH = BASE_DIRECTORY / "models" / "vec_normalize.pkl"

# SET YOUR INITIAL CONDITIONS: [cart_pos, pole_angle, cart_vel, pole_vel]
INITIAL_CONDITIONS = np.array([0, np.deg2rad(20), 0, 0], dtype=np.float64)  # -8 degrees

print("="*70)
print("Testing Trained Model with Custom Initial Conditions")
print("="*70)

# ============= LOAD MODEL =============
print("\nLoading model...")
model = PPO.load(MODEL_PATH)

# Create base environment
base_env = gym.make('InvertedPendulum-v5', render_mode='human', reset_noise_scale=0.0)
env = DummyVecEnv([lambda: base_env])
vec_normalize = VecNormalize.load(VEC_NORMALIZE_PATH, env)
vec_normalize.training = False
vec_normalize.norm_reward = False

print("✓ Model loaded")

print(f"\nInitial conditions:")
print(f"  Cart position: {INITIAL_CONDITIONS[0]:.3f} m")
print(f"  Pole angle: {INITIAL_CONDITIONS[1]:.3f} rad = {np.degrees(INITIAL_CONDITIONS[1]):.1f}°")

# ============= SET INITIAL CONDITIONS =============
vec_normalize.reset()
actual_env = vec_normalize.venv.envs[0].unwrapped

# Set initial state properly
actual_env.set_state(INITIAL_CONDITIONS[:2], INITIAL_CONDITIONS[2:])

# Get initial observation from the environment
obs = vec_normalize.normalize_obs(actual_env._get_obs().reshape(1, -1))

print("\nStarting simulation...\n")

# ============= RUN CONTROL LOOP =============
step = 0
total_reward = 0
max_steps = 1000

try:
    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_normalize.step(action)
        total_reward += reward[0]
        step += 1
        
        if step % 25 == 0:
            # Get CURRENT state from MuJoCo data (the actual simulation state)
            cart_pos = actual_env.data.qpos[0]
            pole_angle = actual_env.data.qpos[1]
            cart_vel = actual_env.data.qvel[0]
            pole_vel = actual_env.data.qvel[1]
            
            print(f"Step {step:4d} | "
                  f"Cart: {cart_pos:+.3f}m | "
                  f"Angle: {np.degrees(pole_angle):+.2f}° | "
                  f"CartVel: {cart_vel:+.3f} | "
                  f"Action: {float(action[0]):+.3f}")
        
        if done[0]:
            print(f"\nTerminated at step {step}")
            break
        
except KeyboardInterrupt:
    print(f"\n\nStopped by user")

# Final stats
cart_pos = actual_env.data.qpos[0]
pole_angle = actual_env.data.qpos[1]

print(f"\n{'='*70}")
print(f"Steps: {step} | Total reward: {total_reward:.2f}")
print(f"Final: Cart={cart_pos:.3f}m, Angle={np.degrees(pole_angle):.1f}°")
print(f"{'='*70}")

vec_normalize.close()
print("\n✓ Done!")