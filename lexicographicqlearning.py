import os
import platform
from time import sleep, time
import numpy as np
#import unicurses
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm 

RENDER_REFRESH_TIME = 0.02
NON_RENDER_REFRESH_TIME = 0.008


#stdscr = unicurses.initscr()
#unicurses.noecho()
#unicurses.cbreak()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        # self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, action_dim)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class LexicographicQTableLearner:

    # based on: Martina Panfili, Antonio Pietrabissa, Guido Oddi & Vincenzo Suraci (2016) 
    # A lexicographic approach to constrained MDP admission control, International Journal of Control,
    # 89:2, 235-247, DOI: 10.1080/00207179.2015.1068955

    def __init__(self, env, model_name, constraints):
        """ Constructor for LexicographicQTableLearner.

        Parameters: 
        env (gym environment): Open AI gym (Toy Text) environment of the game.
        model_name (str): Name of the Open AI gym game

        Returns: None
        """
        self.env = env
        self.model_name = model_name
        action_size = self.env.action_space.n
        state_size = self.env.observation_space.shape[0]
        constraint_size = len(constraints)
        self.constraints = constraints
        self.qtable_constraints = np.zeros((constraint_size, state_size, action_size))
        self.qtable = np.zeros((state_size, action_size))
        self.avg_time = 0
        self.steps_per_episode = 1000
        self.gamma = None
    

    def _clear_screen(self):
        """Clears the terminal screen according to OS.

        Parameters: None

        Returns: None
        """
        os.system('cls') if platform.system() == \
            'Windows' else os.system('clear')

    def _render_train_logs(self, episode, total_episodes, epsilon, step, action, reward, done, done_count):
        """Rendering text logs on console window.

        Parameters:
        episode (int): Running episode number.
        total_episodes (int): The total number of gameplay episodes.
        epsilon (int): Exploration Exploitation tradeoff rate for running episode.
        step (int): Step count for current episode.
        action (int): Action taken for the current state of environment.
        reward (int): Reward for the action taken in the environment.
        done (bool): Flag to know where the episode is finished or not.
        done_count (int): Counter for how many time the agent finished the episode before timeout.

        Returns: None
        """

        # Clear Screen
        self._clear_screen()
        #unicurses.clear()

        # Printing Logs
        print(f'Model Name     :\t{self.model_name}')
        print(f'Q - Table Shape:\t{self.qtable.shape}')
        print(f'Q - Table Constraints Shape:\t{self.qtable_constraints.shape}')
        print(f'Episode Number :\t{episode}/{total_episodes}')
        print(f'Episode Epsilon:\t{epsilon}')
        print(f'Episode Step   :\t{step+1}/{self.steps_per_episode}')
        print(f'Episode Action :\t{action}')
        print(f'Episode Reward :\t{reward}')
        print(f'Episode Done ? :\t{"Yes" if done else "No"}')
        print(f'Done Count     :\t{done_count}')
        #unicurses.refresh()

    def _render_train_env(self):
        """Renders the environment."""
        print()
        self.env.render()

    def _render_train_time(self, episode_left, episode_t,  step_t, done, step_end, render):
        """ Calculates and renders time metrics for training.

        Parameters:
        episode_left (int): Number of episodes left out of total episodes.
        episode_t (int): Running episode time in seconds.
        step_t (int): Running episode step in seconds.
        done (bool): Flag to know where the episode is finished or not.
        step_end (bool): Flag to know if running step is last step of episode limit.
        render (bool): Flag to render the training environment if possible.
        wait (float): Waiting time to continue process. (default:0.02)

        Returns: None
        """
        if self.avg_time == 0:
            self.avg_time = episode_t
        elif done or step_end:
            self.avg_time = (self.avg_time+episode_t)/2

        time_left = int(self.avg_time*episode_left)
        time_left = (time_left//60, time_left % 60)
        print()
        print(f'Time Left            :\t{time_left[0]} mins  {time_left[1]} secs')
        print(f'Average Episode Time :\t{np.round(self.avg_time,4)} secs')
        print(f'Current Episode Time :\t{np.round(episode_t,4)} secs')
        print(f'Current Step Time    :\t{np.round(step_t,4)} secs')
        #unicurses.refresh()
        #sleep(RENDER_REFRESH_TIME if render else NON_RENDER_REFRESH_TIME)

    def train(self, train_episodes=10000, lr_init=0.7, gamma=0.9, render=False):
        """ Calling this method will start the training process.

        Parameters: 
        train_episodes (int): The total number of gameplay episodes to learn from for agent. (default:10000)
        lr (float): Learning Rate used by the agent to update the Q-Table after each episode. (default:0.7)
        gamma (float): Discount Rate used by the agent in Bellman's Equation. (default:0.6)
        render (bool): Flag to render the training environment if possible. (default:False)

        Returns: None
        """
        (epsilon, max_epsilon, min_epsilon, decay_rate) = (1.0, 1.0, 0.01, 0.0001)
        self.gamma = gamma
        done_count = 0
        lr = lr_init
        t_episode = 0
        reward_plot = np.zeros(((len(self.constraints)+1), train_episodes))
        for episode in range(train_episodes):
            t_s_episode = time()

            curr_state = self.env.reset()
            curr_step = 0
            episode_done = False
            lr = lr_init/(episode+1)
            for curr_step in range(self.steps_per_episode):
                t_s_step = time()
                # Exploration Exploitation Tradeoff for the current step.
                ee_tradeoff = np.random.random()
                # Choosing QTable to be used based on which constraint is not satisfied
                c = 0
                constraint_violated = False
                avail_actions = range(self.env.action_space.n)
                for c in range(len(self.constraints)):
                    #print(curr_state)
                    #print(self.qtable_constraints[c][curr_state, :])
                    curr_avail_actions = copy.deepcopy(avail_actions)
                    for action in curr_avail_actions:
                        if self.qtable_constraints[c][curr_state, action] > self.constraints[c]:
                            try:
                                curr_avail_actions.remove(action)
                            except:
                                pass
                    if len(curr_avail_actions) == 0:
                    #if max(self.qtable_constraints[c][curr_state, :]) > self.constraints[c]:
                        constraint_violated = True
                        break
                    else:
                        avail_actions = curr_avail_actions
                selected_q_table = None
                if constraint_violated:
                    selected_q_table = self.qtable_constraints[c]
                else:
                    selected_q_table = self.qtable
                # Choosing action based on tradeoff. Random action or action from QTable.
                if ee_tradeoff > epsilon:
                    action_min = avail_actions[0]
                    value_min = selected_q_table[curr_state, avail_actions[0]]
                    # Choose only among available actions, in order to do not harm previous constraints
                    for action in avail_actions:
                        if selected_q_table[curr_state, action] < value_min:
                            value_min = selected_q_table[curr_state, action]
                            action_min = action
                    curr_action = action_min
                else:
                    curr_action = self.env.action_space.sample()
                # Taking an action, reward will contain the reward of the classic QTable, while info will contain the reward of all the Constraint QTables
                new_state, reward, episode_done, info = self.env.step(
                    curr_action)
                
                if reward == 1:
                    reward = reward - gamma #eq 11-12 paper   
                # Keeping track of done count
                done_count += 1 if episode_done else 0
                # Rendering Logs
                if episode%100 == 0:
                    self._render_train_logs(episode, train_episodes, epsilon, curr_step, curr_action,
                                        reward, episode_done, done_count)
                # Rendering environment
                if render:
                    self._render_train_env()
                # Updating only the the QTable corresponding to the UE class and the general QTable using Bellman Equation
                self.qtable[curr_state, curr_action] = \
                    (1-lr)*self.qtable[curr_state, curr_action] + lr*(reward + gamma * min(self.qtable[new_state, :]))
                reward_plot[0][episode] += reward
                for c in range(len(self.constraints)):
                    reward_constr = info[c]
                    if reward_constr == 1:
                        reward_constr = 1 - gamma 
                    elif reward_constr == -1:
                        continue # update only if UE is of class i (i.e., reward 1 or 0), if reward = -1 skip it  
                    self.qtable_constraints[c][curr_state, curr_action] = \
                        (1-lr)*self.qtable_constraints[c][curr_state, curr_action] + lr*(reward_constr + gamma * min(self.qtable_constraints[c][new_state, :]))
                    reward_plot[c+1][episode] += reward_constr
                
                # Environment state change
                curr_state = new_state

                # Step Time Calculation
                t_step = time() - t_s_step
                if episode % 100 == 0:
                    self._render_train_time(train_episodes-episode, t_episode,
                                        t_step, episode_done, self.steps_per_episode-1 == curr_step, render)

                if episode_done:
                    break

            # Updating Epsilon for Exploration Exploitation Tradeoff
            epsilon = min_epsilon + \
                (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

            # Episode Time Calculation
            t_episode = time()-t_s_episode
        self.env.close()
        self.training_reward_plot = reward_plot

    def _render_test_logs(self, episode, total_episodes, step, episode_reward, step_reward, done, done_count):
        """Rendering text logs on console window.

        Parameters:
        episode (int): Running episode number.
        total_episodes (int): The total number of gameplay episodes.
        step (int): Step count for current episode.
        episode_reward (int): Reward for the action taken in the environment for current episode.
        step_reward (int): Reward for the action taken in the environment for current step.
        done (bool): Flag to know where the episode is finished or not.
        done_count (int): Counter for how many time the agent finished the episode before timeout.

        Returns: None
        """

        # Clear Screen
        self._clear_screen()
        #unicurses.clear()

        # Printing Logs
        print(f'Model Name     :\t{self.model_name}')
        print(f'Episode Number :\t{episode}/{total_episodes}')
        print(f'Episode Step   :\t{step+1}/{self.steps_per_episode}')
        print(f'Episode Reward :\t{episode_reward}')
        print(f'Step Reward    :\t{step_reward}')
        print(f'Episode Done ? :\t{"Yes" if done else "No"}')
        print(f'Done Count     :\t{done_count}')
        #unicurses.refresh()

    def _render_test_env(self, render):
        """Renders the environment

        Parameters:
        render (bool): Flag to render the environment or not.

        Returns: None
        """

        if render:
            self.env.render()
            sleep(RENDER_REFRESH_TIME)
        else:
            #sleep(NON_RENDER_REFRESH_TIME)
            pass

    def test(self, test_episodes=200, render=False):
        """ Testing method to know our environment performance.

        Parameters:
        test_episodes (int): Total number of episodes to evaluate performance.
        render (bool): Flag to render the training environment if possible. (default:False)

        Returns: None
        """
        self.env.reset()
        # Collecting the rewards over time.
        rewards = list()
        constraint_rewards = list()
        done_count = 0
        for episode in range(test_episodes):
            state = self.env.reset()
            # Reward for current episode.
            total_rewards = 0
            total_constraints_rewards = np.zeros(len(self.constraints))
            for _ in range(self.steps_per_episode):
                # Selecting the best action from the appropriate QTable
                c = 0
                constraint_violated = False
                avail_actions = range(self.env.action_space.n)
                for c in range(len(self.constraints)):
                    #print(curr_state)
                    #print(self.qtable_constraints[c][curr_state, :])
                    curr_avail_actions = copy.deepcopy(avail_actions)
                    for action in curr_avail_actions:
                        if self.qtable_constraints[c][state, action] > self.constraints[c]:
                            try:
                                curr_avail_actions.remove(action)
                            except:
                                pass
                    if len(curr_avail_actions) == 0:
                    #if max(self.qtable_constraints[c][curr_state, :]) > self.constraints[c]:
                        constraint_violated = True
                        break
                    else:
                        avail_actions = curr_avail_actions
                selected_q_table = None
                if constraint_violated:
                    selected_q_table = self.qtable_constraints[c]
                else:
                    selected_q_table = self.qtable             
                action_min = avail_actions[0]
                value_min = selected_q_table[state, avail_actions[0]]
                # Choose only among available actions, in order to do not harm previous constraints
                for action in avail_actions:
                    if selected_q_table[state, action] < value_min:
                        value_min = selected_q_table[state, action]
                        action_min = action
                action = action_min
                # Performing the action.
                new_state, reward, done, info = self.env.step(action)                
                total_rewards += reward
                for _ in range(len(info)):
                    if info[_] == -1:
                        total_constraints_rewards[_] += 0
                    else:
                        total_constraints_rewards[_] += info[_]
                # Printing logs
                if episode % 100 == 0:
                    self._render_test_logs(
                        episode, test_episodes, _, total_rewards, reward, done, done_count)
                # Render Environment
                #self._render_test_env(render)

                if done:
                    rewards.append(total_rewards)
                    constraint_rewards.append(total_constraints_rewards)
                    done_count += 1
                    break
                # Changing states for next step
                state = new_state
            if done_count == 0:
                rewards.append(total_rewards)
                constraint_rewards.append(total_constraints_rewards)
        self.env.close()
        #unicurses.addstr(f"\n\nScore over time: \t{sum(rewards)/test_episodes}\n")
        return rewards, constraint_rewards

    def set_refresh_time(self, time, render):
        """ Sets the refresh time for render mode TRUE or FALSE.

        Parameters:
        time (float): Refresh time in seconds.
        render (bool): Render flag for which time parameter is to be set.

        Returns: None
        """
        if render:
            RENDER_REFRESH_TIME = time
        else:
            NON_RENDER_REFRESH_TIME = time


    def save_dqn_model(self, policy_net, target_net, optimizer, model_name=None, model_path='saved_models'):
        """Save the DQN model parameters and optimizer state.
    
        Parameters:
            policy_net (DQNetwork): The policy network
            target_net (DQNetwork): The target network
            optimizer (torch.optim): The optimizer
            model_name (str): Name for the saved model files
            model_path (str): Directory to save the model files
        """
        if not model_name:
            model_name = self.model_name

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        # Save networks and optimizer state
        checkpoint = {
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
    
        path = os.path.join(model_path, f"{model_name}_dqn.pt")
        torch.save(checkpoint, path)
        print(f'DQN model saved at location: {path}')


    def load_dqn_model(self, state_dim=7, action_dim=None, model_name=None, path="saved_models", evaluate=False):
        """Load a saved DQN model.
    
        Parameters:
            state_dim (int): State dimension
            action_dim (int): Action dimension (if None, uses env.n_action)
            model_name (str): Name of the model to load
            path (str): Directory containing the model
            evaluate (bool): If True, sets model to evaluation mode
    
        Returns:
            tuple: (policy_net, target_net, optimizer)
        """
        if action_dim is None:
            action_dim = self.env.n_action
        
        # Initialize networks
        policy_net = DQNetwork(state_dim, action_dim)
        target_net = DQNetwork(state_dim, action_dim)
        optimizer = optim.Adam(policy_net.parameters())
    
        model_path = os.path.join(path, f"{model_name}.pt")
    
        if not os.path.isfile(model_path):
            print(f'Model file not found at location: {model_path}')
            return policy_net, target_net, optimizer
    
        checkpoint = torch.load(model_path)
    
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        if evaluate:
            policy_net.eval()
            target_net.eval()
    
        print(f'DQN model loaded from: {model_path}')
        return policy_net, target_net, optimizer


    def round_robin(self, time_interval, n_episode):
        """
        run our proposed RL, Round-Robin and Greedy algorithms should be passed as `learner`
        """
        avg_return = 0
        li = []
        for eps in range(n_episode):
            curr_state = self.env.reset()
            done = False
            iter_num = 0
            total_return = 0
            # this loop should terminate once `done` is True
            while (not done):
                print()
                print("--------------")
                print(curr_state) # current_power, chnl_cap, connected_users, n_drop, sat_pos
                print(f"{len(self.env.env.ue_list)} connecting users")

                # dummy action
                action = (iter_num // time_interval) % self.env.n_action + 25

                new_state, reward, done, info = self.env.step(action)
                curr_state = new_state

                print(f"[iter {iter_num}] reward: {reward}, done: {done}")
                iter_num += 1
                total_return += reward
            avg_return += total_return
            li.append(total_return)
        print(li)
        avg_return /= n_episode
        return avg_return


    def Qlearning(self, time_interval, n_episode, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Implement Q-learning for the given environment.
        
        Parameters:
        - time_interval: The time step interval for actions.
        - n_episode: Number of episodes for training.
        - alpha: Learning rate.
        - gamma: Discount factor.
        - epsilon: Exploration rate for epsilon-greedy policy.
        """
        avg_return = 0
        li = []  # List to store returns for each episode

        # Initialize Q-table (state-action values)
        q_table = {}  # A dictionary for simplicity, can be replaced by numpy array for discrete states

        for eps in range(n_episode):
            curr_state = self.env.reset()
            done = False
            iter_num = 0
            total_return = 0

            while not done:
                print()
                print("--------------")
                print(curr_state)  # Log current state
                print(f"{len(self.env.env.ue_list)} connecting users")

                # Convert the state to a hashable format for Q-table (e.g., tuple or string)
                state_key = tuple(curr_state) if isinstance(curr_state, list) else curr_state

                # Ensure the state exists in the Q-table
                if state_key not in q_table:
                    q_table[state_key] = [0] * self.env.n_action

                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.env.n_action)  # Explore
                else:
                    action = np.argmax(q_table[state_key])  # Exploit

                # Perform the action in the environment
                new_state, reward, done, info = self.env.step(action)

                # Convert new state to hashable format
                new_state_key = tuple(new_state) if isinstance(new_state, list) else new_state

                # Ensure the new state exists in the Q-table
                if new_state_key not in q_table:
                    q_table[new_state_key] = [0] * self.env.n_action

                # Update Q-value using Q-learning formula
                q_table[state_key][action] += alpha * (
                    reward + gamma * max(q_table[new_state_key]) - q_table[state_key][action]
                )

                # Update state and rewards
                curr_state = new_state
                iter_num += 1
                total_return += reward

                print(f"[iter {iter_num}] reward: {reward}, done: {done}, action: {action}")

            avg_return += total_return
            li.append(total_return)

        print(li)
        avg_return /= n_episode
        return avg_return
    


    def DQN(self, time_interval, n_episode, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999, # 0.995
        learning_rate=0.001, replay_buffer_size=10000, batch_size=64, target_update_freq=10, 
        save_interval=100, model_name=None, model_path='saved_models', verbose=True):
                                                # 512
        # Get state and action dimensions

        def preprocess_state(state):
            flat_state = []
            for elem in state:
                if isinstance(elem, list):  # If element is a list, flatten it further
                    flat_state.extend(preprocess_state(elem))
                elif isinstance(elem, tuple):  # If element is a tuple, expand it
                    flat_state.extend(elem)
                else:  # Otherwise, append the numeric value directly
                    flat_state.append(elem)
            return flat_state
        

        # new_state [33, 9999.966541062533, 1, 861, 3911784.7182674943, 2076474.8654186632, 5383403.519263018]
        # Note, just hand write. following the above state dimension
        state_dim = 6 # change to spherical coord, only need theta and phi angles
        action_dim = self.env.n_action
        accumulated_rewards = []
        # Initialize networks
        policy_net = DQNetwork(state_dim, action_dim)
        target_net = DQNetwork(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()  # Target network is not trained directly

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Replay buffer
        replay_buffer = deque(maxlen=replay_buffer_size)

        # Helper function to select an action
        def select_action(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.randint(0, action_dim) + 0 # Random action
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                return torch.argmax(q_values).item() + 0

        # Training loop
        avg_return = 0
        li = []  # Store returns for each episode

        for eps in tqdm(range(n_episode), desc="Training Progress"):
            
            curr_state = preprocess_state(self.env.reset())
            total_return = 0  # Accumulated reward for the current episode
            done = False
            iter_num = 0
            
            while not done:
                # Select action using epsilon-greedy policy
                action = select_action(curr_state, epsilon)
                
                # Take action and observe the environment
                new_state, reward, done, info = self.env.step(action)
                new_state = preprocess_state(new_state)
                
                # Store transition in replay buffer
                replay_buffer.append((curr_state, action, reward, new_state, done))

                # Sample a mini-batch from replay buffer
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions).unsqueeze(1)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones).unsqueeze(1)

                    # Compute Q(s, a) from the policy network
                    q_values = policy_net(states).gather(1, actions)

                    # Compute target Q-values using the target network
                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q_values = rewards + gamma * next_q_values * (1 - dones)

                    # Compute loss and backpropagate
                    loss = loss_fn(q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        pass

                # Update state and accumulate reward
                curr_state = new_state
                total_return += reward  # Accumulate reward for the current episode
                iter_num += 1

            # Update epsilon (decay exploration)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Update target network periodically
            if eps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Save model periodically with episode number in filename
            if eps % save_interval == 0 and eps > 0:
                episode_model_name = f"{model_name}_episode_{eps}" if model_name else f"dqn_model_episode_{eps}"
                self.save_dqn_model(policy_net, target_net, optimizer, episode_model_name, model_path)
                # Also save the latest version
                latest_model_name = f"{model_name}_latest" if model_name else "dqn_model_latest"
                self.save_dqn_model(policy_net, target_net, optimizer, latest_model_name, model_path)

            avg_return += total_return
            li.append(total_return)
            # same meaning
            accumulated_rewards.append(total_return)
            # Print accumulated reward for the current episode
            print(f"Episode {eps}, Accumulated Reward: {total_return}, Epsilon: {epsilon}")

            # Save graph every 100 episodes
            if eps % 100 == 0 and eps != 0:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(accumulated_rewards)), accumulated_rewards, label='Accumulated Reward per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Accumulated Reward')
                plt.title('Accumulated Reward vs. Episode')
                plt.legend()
                plt.grid()
                plt.savefig(f"DQN_reward_episode_{eps}.png")
                plt.close()

        # Save final model
        final_model_name = f"{model_name}_final" if model_name else "dqn_model_final"
        self.save_dqn_model(policy_net, target_net, optimizer, final_model_name, model_path)

        avg_return /= n_episode
        print(f"Average return across all episodes: {avg_return}")


         # Plot the accumulated rewards
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_episode), accumulated_rewards, label='Accumulated Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Accumulated Reward')
        plt.title('Accumulated Reward vs. Episode')
        plt.legend()
        plt.grid()
        plt.show()

    

        return avg_return