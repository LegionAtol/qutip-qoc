import gymnasium as gym
from gymnasium import spaces
import qutip as qu
from qutip.qip.operations import *
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class GymQubitEnv(gym.Env):
    def __init__(self):
        super(GymQubitEnv, self).__init__()
        
        self.w = 2 * np.pi * 3.9  # (GHz)

        # time for mesolve()
        self.step_time = 0.5    #0.09

        # threshold for trace distance to consider the target operator reached
        self.trace_distance_threshold = 0.01 

        self.current_step_in_the_episode = 0
        self.max_steps = 100    # max episode length (max number of steps for one episode)

        # Reward parameters
        self.C1 = 1    
        self.step_penalty = 1  # step penalty parameter

        self.target_operator = hadamard_transform()

        self.state = None
        self.final_state = None

        # Hamiltonians
        self.H_0 = self.w / 2 * qu.sigmaz()
        self.H_1 = qu.sigmax()
        self.H_tot = None

        self.N = self.target_operator.shape[0]
        self.dim = 2 * self.N**2  # (2*N^2 dim, real and imaginary part)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.dim,))

        # for debugging
        self.episode_reward = 0 
        self.rewards = []   # contains the cumulative reward of each episode
        self.trace_distances = []    # contains the final trace distance of each episode
        self.lowest_trace_distance = float('inf')  # track the lowest trace distance achieved
        self.lowest_trace_distance_episode = 0  # track the episode where lowest trace distance is achieved
        self.episode_actions = []  # Track actions for the current episode
        self.actions = []  # Contains the actions taken in each episode
        self.num_of_terminated = 0  # number of episodes terminated
        self.num_of_truncated = 0   # number of truncated episodes
        self.episode_steps = [] # number of steps for each episodes
        
    def step(self, action):
        action = action[0]  # action is an array -> extract its value with [0]
        alpha = action * 15

        self.H_tot = self.H_0 + alpha * self.H_1

        # Update unitary operator using mesolve
        result = qu.mesolve(self.H_tot, self.state, [0, self.step_time])
        self.state = result.states[-1]  # update unitary operator

        # Calculate the trace distance
        trace_distance = qu.metrics.tracedist(self.state, self.target_operator)

        # Reward based on trace distance
        reward = self.C1 * (1 - trace_distance) - self.step_penalty
        self.current_step_in_the_episode += 1
        terminated = trace_distance <= self.trace_distance_threshold    # if the goal is reached
        truncated = self.current_step_in_the_episode >= self.max_steps  # if the episode ended without reaching the goal

        # for debugging
        self.episode_reward += reward
        self.episode_actions.append(action)
        if terminated or truncated:
            self.final_state = self.state   # for debug
            self.trace_distances.append(trace_distance) # keep the final trace distance
            if trace_distance < self.lowest_trace_distance:
                self.lowest_trace_distance = trace_distance  # update lowest trace distance
                self.lowest_trace_distance_episode = len(self.rewards) + 1  # update the episode number (since rewards are appended after reset)
            self.rewards.append(self.episode_reward) # keep the episode rewards
            self.episode_reward = 0  # Reset the episode reward
            self.episode_steps.append(self.current_step_in_the_episode) # Keep the number of steps used for this episode
            self.current_step_in_the_episode = 0  # Reset the step counter
            self.actions.append(self.episode_actions.copy()) # Append actions of the episode to the actions list
            self.episode_actions = []  # Reset the actions for the new episode
        if terminated:
            self.num_of_terminated += 1
        elif truncated:
            self.num_of_truncated += 1

        observation = self._get_obs()
        
        return observation, reward, bool(terminated), bool(truncated), {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.state = qu.qeye(2)  # Use identity operator as the initial state
        return self._get_obs(), {}

    def _get_obs(self):
        real_part = np.real(self.state.full())
        imag_part = np.imag(self.state.full())
        obs = np.concatenate((real_part.flatten(), imag_part.flatten()))
        return obs.astype(np.float32) # Gymnasium expects the observation to be of type float32


if __name__ == '__main__':
    env = GymQubitEnv()

    # Check if the environment follows Gym API
    check_env(env, warn=True)

    # Create the model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=80000) #100000
 
    # For debugging
    print("\n Summary of the training:")
    for i, (r, d) in enumerate(zip(env.rewards, env.trace_distances), start=1):
        print(f"Trace distance for episode {i}: {d}")
        if i % 50 == 0:
            avg_reward = np.mean(env.rewards[i-50:i])
            avg_trace_distance = np.mean(env.trace_distances[i-50:i])
            print(f"Episode {i}, Avg reward of last 50 episodes: {avg_reward}")
            print(f"Episode {i}, Avg trace distance of last 50 episodes: {avg_trace_distance}\n")

    print(f"Lowest trace distance achieved during training: {env.lowest_trace_distance}")
    print(f"Lowest trace distance was achieved in episode: {env.lowest_trace_distance_episode}")
    print(f"Number of: Terminated episodes {env.num_of_terminated}, Truncated episodes {env.num_of_truncated}")
    print(f"Number of steps used in each episode {env.episode_steps}")
    
    # Plot actions of some episodes
    num_episodes = len(env.actions)
    indices = [9, 19, num_episodes - 11, num_episodes - 1]  # 10th, 20th, (final-10)th, final episodes
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    for i, idx in enumerate(indices):
        steps = np.arange(len(env.actions[idx]))  # Create an array of step indices
        actions = env.actions[idx]  # Extract action values from the array  
        axs[i].step(steps, actions, where='post')
        axs[i].set_title(f'Episode {idx + 1}')
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel('Action')
    plt.tight_layout()
    plt.show()

    # Test 
    init_state = qu.basis(2, 0)
    final_state = env.final_state * init_state

    print("final state", env.final_state)
    trace_distance = qu.metrics.tracedist(env.final_state, env.target_operator)
    print(f"Trace distance between final state and target_operator: {trace_distance}")

    # Visualize on the Bloch sphere
    b = qu.Bloch()
    b.add_states(init_state)
    b.add_states(final_state)   
    fig = plt.figure()
    b.fig = fig
    b.render()
    plt.show()
