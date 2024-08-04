import gymnasium as gym
from gymnasium import spaces
import qutip as qu
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class GymQubitEnv(gym.Env):
    def __init__(self):
        super(GymQubitEnv, self).__init__()

        self.dim = 4  # dimension of Hilbert space for two qubits

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Continuous action space from -1 to +1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)  # Observation space for two qubits (4 real and 4 imaginary parts)

        # time for step
        self.step_time = 0.01

        # threshold for fidelity to consider the target state reached
        self.fidelity_threshold = 0.99 

        self.current_step_in_the_episode = 0
        self.max_steps = 100  # max episode length

        # Reward parameters
        self.C1 = 1    
        self.step_penalty = 1  # step penalty parameter

        # Target state |10>
        self.target_state = qu.tensor(qu.basis(2, 1), qu.basis(2, 0))
        self.state = None  # actual state
        self.final_state = None #for debug

        sx_sx = qu.tensor(qu.sigmax(), qu.sigmax())
        sy_sy = qu.tensor(qu.sigmay(), qu.sigmay())
        sz_sz = qu.tensor(qu.sigmaz(), qu.sigmaz())
        i_sx, sx_i = qu.tensor(qu.sigmax(), qu.qeye(2)), qu.tensor(qu.qeye(2), qu.sigmax())
        i_sy, sy_i = qu.tensor(qu.sigmay(), qu.qeye(2)), qu.tensor(qu.qeye(2), qu.sigmay())
        i_sz, sz_i = qu.tensor(qu.sigmaz(), qu.qeye(2)), qu.tensor(qu.qeye(2), qu.sigmaz())

        # Drift H
        self.H_0 = 0.5*(sx_sx + sy_sy + sz_sz)
        # Control H
        self.H_1 = [sx_i, sy_i, i_sx, i_sy]

        # For debugging
        self.episode_reward = 0 
        self.rewards = []  # contains the cumulative reward of each episode
        self.fidelities = []  # contains the final fidelity of each episode
        self.highest_fidelity = 0  # track the highest fidelity achieved
        self.highest_fidelity_episode = 0  # track the episode where highest fidelity is achieved
        self.episode_actions = []  # Track actions for the current episode
        self.actions = []  # Contains the actions taken in each episode
        self.num_of_terminated = 0  # number of episodes terminated
        self.num_of_truncated = 0  # number of truncated episodes
        self.episode_steps = []  # number of steps for each episode
        
    def step(self, action):
        #action = action[0]  # action is an array -> extract its value with [0]
        
        #action = ((action + 1) / 2 * (13 - 9)) + 9  # Scale action from [-1, 1] to [9, 13]
        action = action * 15
        #print(action)

        H = self.H_0
        for i, alpha in enumerate(action):
            H += alpha * self.H_1[i]
       
        result = qu.mesolve(H, self.state, [0, self.step_time])
        self.state = result.states[-1]  # result.states returns a list of state vectors (kets), is a Qobj object. let's take the last one.
        #print(f"state: {self.state}")

        fidelity = qu.metrics.fidelity(self.state, self.target_state)
        reward = self.C1 * fidelity - self.step_penalty
        self.current_step_in_the_episode += 1
        terminated = fidelity >= self.fidelity_threshold  # if the goal is reached
        truncated = self.current_step_in_the_episode >= self.max_steps  # if the episode ended without reaching the goal

        reward = float(reward.item())  # Ensure the reward is a float

        # For debugging
        self.episode_reward += reward
        self.episode_actions.append(action)
        if terminated or truncated:
            self.final_state = self.state
            self.fidelities.append(fidelity)  # keep the final fidelity
            if fidelity > self.highest_fidelity:
                self.highest_fidelity = fidelity  # update highest fidelity
                self.highest_fidelity_episode = len(self.rewards) + 1  # update the episode number
            self.rewards.append(self.episode_reward)  # keep the episode rewards
            self.episode_reward = 0  # Reset the episode reward
            self.episode_steps.append(self.current_step_in_the_episode)  # Keep the number of steps used for this episode
            self.current_step_in_the_episode = 0  # Reset the step counter
            self.actions.append(self.episode_actions.copy())  # Append actions of the episode to the actions list
            self.episode_actions = []  # Reset the actions for the new episode
        if terminated:
            self.num_of_terminated += 1
        elif truncated:
            self.num_of_truncated += 1

        observation = self._get_obs()
        
        return observation, reward, bool(terminated), bool(truncated), {"state": self.state}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.state = self.create_init_state()

        return self._get_obs(), {}

    # if state=(p q)' with p = a + i*b and q = c + i*d -> return [a, b, c, d]
    def _get_obs(self):
        rho = self.state.full().flatten()  # to have state vector as NumPy array and flatten into one dimensional array
        obs = np.concatenate((np.real(rho), np.imag(rho)))
        return obs.astype(np.float32)  # Gymnasium expects the observation to be of type float32

    def create_init_state(self, noise=False, random=False):
        init_state = qu.tensor(qu.basis(2, 1), qu.basis(2, 1)).unit()  # |11>
        return init_state


if __name__ == '__main__':
    env = GymQubitEnv()

    # Check if the environment follows Gym API
    check_env(env, warn=True)

    # Create the model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=160000)
 
    # For debugging
    print("\n Summary of the training:")
    for i, (r, f) in enumerate(zip(env.rewards, env.fidelities), start=1):
        print(f"Fidelity for episode {i}: {f}")
        if i % 50 == 0:
            avg_reward = np.mean(env.rewards[i-50:i])
            avg_fidelity = np.mean(env.fidelities[i-50:i])
            print(f"Episode {i}, Avg reward of last 50 episodes: {avg_reward}")
            print(f"Episode {i}, Avg fidelity of last 50 episodes: {avg_fidelity}\n")

    print(f"Highest fidelity achieved during training: {env.highest_fidelity}")
    print(f"Highest fidelity was achieved in episode: {env.highest_fidelity_episode}")
    print(f"Number of: Terminated episodes {env.num_of_terminated}, Truncated episodes {env.num_of_truncated}")
    print(f"Number of steps used in each episode {env.episode_steps}")
    
    # Plot actions of some episodes
    num_episodes = len(env.actions)
    indices = [9, 19, num_episodes - 11, num_episodes - 1]  # 10th, 20th, (final-10)th, final episodes
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    for i, idx in enumerate(indices):
        steps = np.arange(len(env.actions[idx]))  # Create an array of step indices
        actions = env.actions[idx]  # Extract action values from the array  
        # Plot each action as a constant value over its interval
        axs[i].step(steps, actions, where='post')
        axs[i].set_title(f'Episode {idx + 1}')
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel('Action')
    plt.tight_layout()


    # Test the model
    final_state = env.final_state
    print(f"final_state : {final_state}")

    target_state = env.target_state
    fidelity = qu.metrics.fidelity(final_state, target_state)
    print(f"final fidelity(final_state, |10>): {fidelity}")
