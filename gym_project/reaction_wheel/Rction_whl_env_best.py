"""
Custom Gymnasium Environment for Reaction Wheel Inverted Pendulum
==================================================================
This environment simulates a pendulum balanced using a reaction wheel.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

class ReactionWheelPendulumEnv(MujocoEnv):
    """
    ## Description
    A pendulum is mounted on a base. At the top of the pendulum is a reaction wheel.
    By spinning the wheel, you create torque that keeps the pendulum balanced upright.
    
    ## Observation Space
    The observation is a 4-dimensional vector:
    - pendulum_angle: Angle of the pendulum from vertical (radians)
    - pendulum_velocity: Angular velocity of the pendulum
    - wheel_velocity: Angular velocity of the reaction wheel
    - cos(pendulum_angle): Cosine of pendulum angle (helps the network)
    
    ## Action Space
    The action is 1-dimensional:
    - wheel_torque: Torque applied to the reaction wheel (controls how fast it spins)
    
    ## Rewards
    You get reward for:
    - Keeping the pendulum upright (angle close to 0)
    - Minimizing angular velocity (smooth control)
    - Small penalty for using too much torque (energy efficiency)
    
    ## Episode Termination
    Episode ends when:
    - Pendulum falls beyond ±30 degrees (~0.52 radians)
    - Maximum timesteps reached (1000 steps = 2 seconds at 0.002s timestep)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,  # 1/0.002 = 500 fps but frame skip = 5, so 500/5 = 100
    }
    
    def __init__(self, reset_noise_scale, **kwargs):
        """
        Initialize the environment.
        
        Args:
            reset_noise_scale: How much random noise to add when resetting
                              (smaller = easier, starts closer to upright)
        """
        self.reset_noise_scale = reset_noise_scale
        
        # Get the path to our XML file
        xml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'Rction_whl.xml'
        )
        
        # observation_space and action_space will be set by MujocoEnv
        # but we need to know their sizes
        observation_space = Box(
            low=np.array([-np.pi/2, -30.0, -75.0, -1.0]),
            high=np.array([np.pi/2, 30.0, 75.0, 1.0]),
            shape=(4,),  # 4 observations
            dtype=np.float64
        )
        
        # Initialize the MuJoCo environment
        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip=5,  # How many physics steps per action
            observation_space=observation_space,
            default_camera_config={"distance": 2.0, "elevation": -20.0},
            **kwargs
        )
        
        # Action space: torque on the reaction wheel
        self.action_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(1,),
            dtype=np.float32
        )
    
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action: The torque to apply to the reaction wheel
            
        Returns:
            observation: Current state
            reward: Reward for this step
            terminated: Whether episode is over (fell down)
            truncated: Whether hit max timesteps
            info: Extra information
        """
        # Apply the action (torque to wheel)
        self.do_simulation(action, self.frame_skip)
        
        # Get the current state
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._get_reward(observation, action)
        
        # Check if episode should end (pendulum fell)
        terminated = self._is_terminated(observation)
        
        # Truncated is handled by TimeLimit wrapper automatically
        truncated = False
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        # Return the standard Gymnasium tuple
        return observation, reward, terminated, truncated, {}
    
    def _get_obs(self):
        """
        Get the current observation from the simulation.
        
        Returns:
            numpy array of [angle, angular_vel, wheel_vel, cos(angle)]
        """
        # Get joint positions and velocities from MuJoCo
        pendulum_angle = self.data.qpos[0]  # First joint position
        pendulum_velocity = self.data.qvel[0]  # First joint velocity
        wheel_angle = self.data.qpos[1]
        wheel_velocity = self.data.qvel[1]  # Second joint velocity (wheel)
        
        # Return observation vector
        return np.array([
            pendulum_angle,
            pendulum_velocity,
            wheel_velocity,
            np.cos(pendulum_angle)  
        ], dtype=np.float64)
    
    def _get_reward(self, observation, action):
        """
        Calculate the reward for the current state and action.
        
        Higher reward = better balance
        
        Args:
            observation: Current state
            action: Action taken
            
        Returns:
            float: reward value
        """
        pendulum_angle = observation[0]
        pendulum_velocity = observation[1]
        wheel_velocity = observation[2]
        
        # === COST COMPONENTS ===
        
        reward = 1.0
        # survival component
        # === 2. Smooth Angle Penalty (The most important part) ===

        reward -= 10.0 * (pendulum_angle ** 2)

        reward -= 0.001 * (pendulum_velocity**2)
        reward -= 0.00005 * (wheel_velocity**2)
        reward -= 0.0001 * np.sum(action** 2)
    
        if abs(pendulum_angle) > 0.52:
            reward = -50.0
        return reward
    
    def _is_terminated(self, observation):
        """
        Check if the episode should end.
        
        Episode ends if pendulum falls beyond ±30 degrees.
        
        Args:
            observation: Current state
            
        Returns:
            bool: True if episode should end
        """
        pendulum_angle = observation[0]
        wheel_velocity = observation[2]
        pendulum_velocity = observation[1]
        
        # End if pendulum falls beyond ±30 degrees (0.523 radians), starts falling too quickly or the wheel starts spinning uncontrollably
        if abs(pendulum_angle) > 0.523:
            return True
        if abs(wheel_velocity) > 100:
            return True
        if abs(pendulum_velocity)> 50:
            return True
        
        return False
    
    def reset_model(self):
        """
        Reset the simulation to initial state.
        
        Called automatically by reset().
        
        Returns:
            observation: Initial state
        """
        # Random initial state (small noise around upright position)
        qpos = self.init_qpos + self.np_random.uniform(
            low=-self.reset_noise_scale,
            high=self.reset_noise_scale,
            size=self.model.nq  # Number of position coordinates
        )
        
        qvel = self.init_qvel + self.np_random.uniform(
            low=-self.reset_noise_scale,
            high=self.reset_noise_scale,
            size=self.model.nv  # Number of velocity coordinates
        )
        
        # Set the state in MuJoCo
        self.set_state(qpos, qvel)
        
        return self._get_obs()


# Register the environment so you can use it with gym.make()
gym.register(
    id='ReactionWheelPendulum-v0',
    entry_point=__name__ + ':ReactionWheelPendulumEnv',
    max_episode_steps=1000,  # 2 seconds at 0.002s timestep
    reward_threshold=975.0,  # Consider solved if avg reward > 950
)