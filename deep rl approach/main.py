# My second approach deep reinforcement learning 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple 3D block-building environment.
class BlockBuildEnv:
    # this will initialize a minecraft-like 3d environment for block building
    def __init__(self, grid_size=3, goal_structure=None):
        self.N = grid_size
        if goal_structure is None:
            goal_structure = np.zeros((grid_size, grid_size, grid_size), dtype=int)
            goal_structure[0, 0, :] = 1  
        self.goal = goal_structure
        self.state = np.zeros_like(self.goal, dtype=int)
        self.max_steps = int(self.goal.sum())  
        self.step_count = 0

    # this method will reset the environment to the initial state
    def reset(self):
        self.state.fill(0)
        self.step_count = 0
        return self._get_observation()

    # this will construct the observation for the agenr
    def _get_observation(self):
        # this will show current state as s flat vector of 0/1
        current_flat = self.state.flatten()  
        # goal structure as flat vector of 0/1
        goal_flat = self.goal.flatten()      
        # Combine them (for example, first part is current, second part is goal)
        obs = np.concatenate([current_flat, goal_flat]).astype(np.float32)
        return obs

    # This code will take an action in the environment
    def step(self, action):
        x, y, z = self._index_to_xyz(action)
        reward = 0.0
        done = False

        #checking validity of actions

        if not (0 <= x < self.N and 0 <= y < self.N and 0 <= z < self.N):
            reward -= 1.0  
        else:
            if self.state[x, y, z] == 0:
                if self.goal[x, y, z] == 1:
                    supported = (z == 0) or (self.goal[x, y, z-1] == 0) or (self.state[x, y, z-1] == 1)
                    if not supported:
                        # If not supported, treat it as an invalid placement (block would "float" - disallowed)
                        reward -= 1.0  # penalty for trying to place a block without support
                    else:
                        # Valid placement: place the block
                        self.state[x, y, z] = 1
                        reward += 1.0  # reward for correctly placing a needed block
                else:
                    # The agent placed a block where it's NOT needed in the goal
                    self.state[x, y, z] = 1  # we allow it to place (could be considered a mistake)
                    reward -= 0.5  
            else:
                # The chosen cell already has a block
                reward -= 0.5  # slight penalty for wasting a move 

        self.step_count += 1
        # Check if the structure is complete (i.e., current state matches goal)
        if np.array_equal(self.state, self.goal):
            reward += 5.0  # big reward for completing the structure
            done = True
        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_observation()
        return obs, reward, done, {}

    # This will convert a flat index into (x,y,z) coordinates in the 3D grid
    def _index_to_xyz(self, index):
        # We flatten in order x + N*y + N*N*z
        z = index // (self.N * self.N)
        remainder = index % (self.N * self.N)
        y = remainder // self.N
        x = remainder % self.N
        return x, y, z

    @property
    def observation_space_size(self):
        # Observation are the  two grids concatenated (current and goal), each of size N^3.
        return 2 * (self.N ** 3)

    @property
    def action_space_size(self):
        # One discrete action for each cell in the grid
        return self.N ** 3

# This is the policy of the rl agent
class BlockBuildPolicy(nn.Module):
    def __init__(self, obs_size, action_size):
        super(BlockBuildPolicy, self).__init__()
        hidden_size = 128
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
    
    # Forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)      
        value = self.value_head(x).squeeze(-1)  
        return logits, value

    # this method will choose an action based on the current state (for training, we sample from the policy distribution).
    def act(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0) 
        logits, value = self.forward(state_t)
        action_probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return int(action.item()), log_prob.squeeze().detach(), value.squeeze().detach()

    # The following method will compute log probabilities and values for given states and actions (for PPO update).
    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()  # entropy of the action distribution (for exploration bonus)
        return log_probs, values, entropy


env = BlockBuildEnv(grid_size=3)  
obs_size = env.observation_space_size 
action_size = env.action_space_size

policy = BlockBuildPolicy(obs_size, action_size)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# discount factor
gamma = 0.99        
# number of update epochs per batch of experience
ppo_epochs = 4      
# PPO clip parameter
clip_epsilon = 0.2  
# number of timesteps to run (and collect) per update
batch_size = 1000   
train_iterations = 100  

for it in range(train_iterations):
    batch_states = []
    batch_actions = []
    batch_log_probs = []
    batch_values = []
    batch_rewards = []
    batch_done = []

    state = env.reset()
    episode_rewards = []
    for t in range(batch_size):
        action, log_prob, value = policy.act(state)
        next_state, reward, done, _ = env.step(action)
        batch_states.append(state)
        batch_actions.append(action)
        batch_log_probs.append(log_prob)
        batch_values.append(value)
        batch_rewards.append(reward)
        batch_done.append(done)
        state = next_state
        episode_rewards.append(reward)
        if done:
            state = env.reset()
            episode_rewards = []
    batch_states_t = torch.FloatTensor(batch_states)
    batch_actions_t = torch.LongTensor(batch_actions)
    batch_log_probs_t = torch.FloatTensor([lp.item() for lp in batch_log_probs])
    batch_values_t = torch.FloatTensor([v.item() for v in batch_values])
    batch_rewards_t = torch.FloatTensor(batch_rewards)
    batch_done_t = torch.FloatTensor(batch_done)

    returns = []
    G = 0.0
    for reward, done_flag, value in zip(reversed(batch_rewards), reversed(batch_done), reversed(batch_values)):
        # If episode is done, reset cumulative reward
        if done_flag:
            G = 0.0
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    advantages = returns - batch_values_t

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy.train()
    for epoch in range(ppo_epochs):
        new_log_probs, values, entropy = policy.evaluate_actions(batch_states_t, batch_actions_t)
        ratios = torch.exp(new_log_probs - batch_log_probs_t)
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))
        value_loss = F.mse_loss(values, returns)
        entropy_bonus = torch.mean(entropy)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (it + 1) % 10 == 0:
        total_reward = returns.sum().item()
        avg_reward = batch_rewards_t.mean().item()
        print(f"Iteration {it+1}/{train_iterations}: Avg reward per step = {avg_reward:.3f}")

# testing the learned policy:
state = env.reset()
done = False
steps = 0
print("\nTesting trained policy on one episode:")
while not done and steps < env.max_steps:
    state_t = torch.FloatTensor(state).unsqueeze(0)
    logits, _ = policy.forward(state_t)
    action = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()
    state, _, done, _ = env.step(action)
    x, y, z = env._index_to_xyz(action)
    print(f"Placed block at {(x, y, z)}")
    steps += 1


print("Final structure matches goal: ", np.array_equal(env.state, env.goal))
print("Goal structure (1 indicates a block):\n", env.goal)
print("Built structure:\n", env.state)
