"""
Custom Gymnasium Environment for 6-DOF Upkie Balancing
======================================================
Floating-base humanoid-like wheeled robot balancing task.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import os


class UpkieBalanceEnv(MujocoEnv):
    """
    ## Description
    A floating-base 6-DOF wheeled biped robot (Upkie-like).
    The goal is to dynamically balance upright using torque control.

    ## Observation Space (21D)
    - gravity vector in body frame (3)
    - base angular velocity (3)
    - base linear velocity (3)
    - joint positions (6)
    - joint velocities (6)

    ## Action Space (6D)
    - Joint torques:
        [left_hip, right_hip,
         left_knee, right_knee,
         left_wheel, right_wheel]

    Actions are normalized in [-1,1] and internally scaled to torque limits.

    ## Rewards
    - COM horizontal stability (Gaussian)
    - COM velocity penalty
    - Base angular velocity penalty
    - Torque penalty

    ## Termination
    - Robot falls (gravity misaligned)
    - Torso height too low
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,  # approx 1/(0.002*5)
    }

    def __init__(self, reset_noise_scale=0.01, **kwargs):

        self.reset_noise_scale = reset_noise_scale

        xml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "upkie.xml"
        )

        # Observation: 21-dimensional
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config={"distance": 2.5, "elevation": -20},
            **kwargs
        )

        # Torque limits (from XML)
        self.torque_limits = np.array(
            [16.0, 16.0, 16.0, 16.0, 1.7, 1.7],
            dtype=np.float32
        )

        # Normalized action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        # Nominal COM height (upright)
        self.target_com_height = 0.4169979364138086

        # Standard deviation for vertical COM tolerance
        self.height_std = 0.03  # 3 cm tolerance
        self.prev_vx_com = 0.0
    # ============================================================
    # STEP
    # ============================================================

    def step(self, action):

        # Scale normalized action to physical torque
        torque = np.clip(action, -1.0, 1.0) * self.torque_limits

        self.do_simulation(torque, self.frame_skip)

        observation = self._get_obs()
        reward = self._get_reward(observation, torque)
        terminated = self._is_terminated(observation)
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, {}

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_obs(self):

        # ----------------------------
        # Base orientation
        # ----------------------------
        quat = self.data.qpos[3:7]

        R_flat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R_flat, quat)
        R = R_flat.reshape(3, 3)

        gravity_world = np.array([0.0, 0.0, -1.0])
        g_body = R.T @ gravity_world

        # ----------------------------
        # Base velocities
        # ----------------------------
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]

        # ----------------------------
        # Joint states
        # ----------------------------
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]

        observation = np.concatenate([
            g_body,
            base_ang_vel,
            base_lin_vel,
            joint_pos,
            joint_vel
        ])

        return observation

    # ============================================================
    # REWARD
    # ============================================================

    def _get_reward(self, observation, torque):

        g = 9.81
        dt = self.model.opt.timestep
        g_body = observation[0:3]
        pitch = np.arcsin(g_body[0])
        # -------------------------------------------------
        # Whole-robot COM
        # -------------------------------------------------
        root_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso"
        )

        com = self.data.subtree_com[root_id]
        com_vel = self.data.subtree_linvel[root_id]

        x_com, y_com, z_com = com
        vx_com, vy_com, vz_com = com_vel

        # -------------------------------------------------
        # Finite-difference acceleration (simple & stable)
        # -------------------------------------------------
        ax_com = (vx_com - self.prev_vx_com) / dt
        self.prev_vx_com = vx_com

        # -------------------------------------------------
        # Support midpoint (sagittal only)
        # -------------------------------------------------
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")

        left_pos = self.data.xpos[left_id]
        right_pos = self.data.xpos[right_id]

        support_mid_x = 0.5 * (left_pos[0] + right_pos[0])

        # -------------------------------------------------
        # ZMP (LIPM approximation)
        # -------------------------------------------------
        x_zmp = x_com - (z_com / g) * ax_com
        zmp_error = x_zmp - support_mid_x

        # -------------------------------------------------
        # Height regulation
        # -------------------------------------------------
        height_error = z_com - self.target_com_height

        # -------------------------------------------------
        # Angular velocity
        # -------------------------------------------------
        base_ang_vel = observation[3:6]

        # -------------------------------------------------
        # Reward components
        # -------------------------------------------------

        # ZMP stability (main term)
        r_zmp = np.exp(-(zmp_error / 0.05) ** 2)
        

        # # Height regulation
        r_height = np.exp(-(height_error / self.height_std) ** 2)

        # # Horizontal velocity damping
        r_velocity = -0.5 * (vx_com)**2
        
        # #pitch angle penalty
        r_pitch = -100 * (pitch)**2
        # # Angular velocity damping
        r_angular = -0.1* np.sum(base_ang_vel**2)

        # # Torque penalty
        r_torque = -0.0001 * np.sum(torque**2)
        # -------------------------------------------------
        # Combined reward
        # -------------------------------------------------
        reward = (6*r_height + 3*r_zmp + r_pitch + r_velocity + r_angular + r_torque)

        # -------------------------------------------------
        # Fall detection
        # -------------------------------------------------
        if z_com < 0.25:
            return -100.0

        if not np.isfinite(reward):
            return -100.0

        return reward
            # ============================================================
            # TERMINATION
            # ============================================================

    def _is_terminated(self, observation):

        g_body = observation[0:3]
        height = self.data.qpos[2]

        # Too tilted
        if g_body[2] > -0.7:
            return True

        # Too low
        if height < 0.25:
            return True

        return False

    # ============================================================
    # RESET
    # ============================================================

    def reset_model(self):

        qpos = self.init_qpos + self.np_random.uniform(
            low=-self.reset_noise_scale,
            high=self.reset_noise_scale,
            size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=-self.reset_noise_scale,
            high=self.reset_noise_scale,
            size=self.model.nv
        )

        self.set_state(qpos, qvel)
        self.prev_vx_com = 0.0
        return self._get_obs()


# ============================================================
# REGISTER ENVIRONMENT
# ============================================================

gym.register(
    id="UpkieBalance-v0",
    entry_point=__name__ + ":UpkieBalanceEnv",
    max_episode_steps=2000,
)