"""
Visualize Trained Reaction Wheel Pendulum Agent with Custom Perturbations
=========================================================================
Watch your trained agent balance the pendulum and test it with custom initial conditions!

Usage:
    python visualize_reaction_wheel.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import time
import numpy as np
from gymnasium import Wrapper

class NoTerminationWrapper(Wrapper):
    """Wrapper that disables all termination conditions."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Force terminated and truncated to False
        return obs, reward, False, False, info

# Import your custom environment
# First, make sure Python can find modules in the current directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from Rction_whl_env_best import ReactionWheelPendulumEnv

# ============= CONFIGURATION =============
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / 'models_reaction_wheel_best'

# SET YOUR INITIAL CONDITIONS: [pendulum_angle, pendulum_vel, wheel_vel, cos(wheel angle)]
# Example perturbations:
INITIAL_CONDITIONS = np.array([np.deg2rad(12), 0.05, 0.1, 0.0], dtype=np.float64)  # ~17 degrees tilt
# Try these:
# np.array([0.5, 0.0, 0.0, 0.0])  # ~28 degrees - harder
# np.array([-0.4, 0.0, 0.0, 0.0]) # -23 degrees - opposite side
# np.array([0.0, 0.0, 0.0, 0.0])  # Perfect upright

DISABLE_TERMINATION = True  # Set to True to see recovery attempts
MAX_STEPS = 2000  # Maximum steps to run

print("="*70)
print("Visualizing Trained Reaction Wheel Pendulum Agent")
if DISABLE_TERMINATION:
    print("Mode: NO TERMINATION - Watch it try to recover!")
else:
    print("Mode: Normal termination enabled")
print("="*70)

# Check if model exists
best_model_path = MODELS_DIR / 'best_model.zip'
if not best_model_path.exists():
    print(f"\n❌ Error: No trained model found at {best_model_path}")
    print("Please train a model first using train_reaction_wheel.py")
    exit(1)

print(f"\n✓ Loading model from: {best_model_path}")

# ============= LOAD MODEL =============
model = PPO.load(str(MODELS_DIR / 'best_model'))

# Create environment with rendering
print("✓ Creating environment with rendering...")

def make_env():
    env = ReactionWheelPendulumEnv(reset_noise_scale=0.0, render_mode='rgb_array')
    if DISABLE_TERMINATION:
        env = NoTerminationWrapper(env)
    return Monitor(env)

VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

def make_env_with_video():
    env = ReactionWheelPendulumEnv(reset_noise_scale=0.0, render_mode='rgb_array')
    if DISABLE_TERMINATION:
        env = NoTerminationWrapper(env)
    env = Monitor(env)
    env = RecordVideo(
        env,
        video_folder=str(VIDEO_DIR),
        name_prefix="reaction_wheel",
        episode_trigger=lambda x: True  # record every episode
    )
    return env

env = DummyVecEnv([make_env_with_video])
vec_normalize = VecNormalize.load(str(MODELS_DIR / 'vec_normalize.pkl'), env)
vec_normalize.training = False
vec_normalize.norm_reward = False

print("✓ Model loaded")
if DISABLE_TERMINATION:
    print("✓ Termination disabled - will run until max steps!")

print(f"\nInitial conditions:")
print(f"  Pendulum angle: {INITIAL_CONDITIONS[0]:.3f} rad = {np.degrees(INITIAL_CONDITIONS[0]):.1f}°")
print(f"  Pendulum velocity: {INITIAL_CONDITIONS[2]:.3f} rad/s")

print("\n" + "="*70)
print("Starting simulation... (Press Ctrl+C to stop)")
print("="*70)

# ============= SET INITIAL CONDITIONS =============
vec_normalize.reset()
actual_env = vec_normalize.venv.envs[0].env

# Unwrap to get to the actual MuJoCo environment
while hasattr(actual_env, 'env'):
    actual_env = actual_env.env

# Set initial state
actual_env.set_state(INITIAL_CONDITIONS[:2], INITIAL_CONDITIONS[2:])
obs = vec_normalize.normalize_obs(actual_env._get_obs().reshape(1, -1))

# ============= RUN CONTROL LOOP =============
step = 0
total_reward = 0

try:
    while step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_normalize.step(action)
        total_reward += reward[0]
        step += 1
        
        # Small delay to make it watchable (100 FPS -> 10ms per frame)
        # time.sleep(0.01)
        
        if step % 25 == 0:
            pendulum_angle = actual_env.data.qpos[0]
            wheel_angle = actual_env.data.qpos[1]
            pendulum_vel = actual_env.data.qvel[0]
            wheel_vel = actual_env.data.qvel[1]
            
            print(f"Step {step:4d} | "
                  f"Angle: {np.degrees(pendulum_angle):+6.2f}° | "
                  f"Vel: {pendulum_vel:+.2f}rad/s | "
                  f"Wheel: {wheel_vel:+6.2f}rad/s | "
                  f"Action: {float(action[0]):+.3f}")
        
        # Check if terminated (only if termination is enabled)
        if done[0] and not DISABLE_TERMINATION:
            print(f"\nEpisode terminated at step {step}")
            break
        
        # Safety stop if pendulum spins wildly
        pendulum_angle = actual_env.data.qpos[0]
        if abs(pendulum_angle) > 2 * np.pi:
            print(f"\nPendulum rotated beyond 360° - stopping")
            break
        
except KeyboardInterrupt:
    print(f"\n\n⚠️ Stopped by user")

# Final stats
pendulum_angle = actual_env.data.qpos[0]
pendulum_vel = actual_env.data.qvel[0]

print(f"\n{'='*70}")
print(f"Steps: {step} | Total reward: {total_reward:.2f}")
print(f"Final: Angle={np.degrees(pendulum_angle):.1f}°, Velocity={pendulum_vel:.2f}rad/s")

if abs(pendulum_angle) < 0.1:
    print("✓ Controller successfully balanced/recovered!")
else:
    print("✗ Controller failed to balance")
    
print(f"{'='*70}")

vec_normalize.close()

print("\n✅ Visualization complete!")
print("\nTip: Edit INITIAL_CONDITIONS at the top of this script to test different perturbations!")