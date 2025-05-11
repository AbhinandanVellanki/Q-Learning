# Author: Vibhakar Mohta (vmohta@cs.cmu.edu)

import numpy as np
import matplotlib.pyplot as plt

class RandomAgent():        
    def __init__(self):
        self.available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
    # Choose a random action
    def get_action(self, state, explore=True):
        # randomly choose an action
        action = np.random.choice(self.available_actions)
        return action

class QAgent():
    def __init__(self, environment, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize the Q-table with random values
        self.q_table = dict()
        for i in range(self.environment.height):
            for j in range(self.environment.width):
                # each position in the grid is a state, define dictionary for each state
                self.q_table[(i, j)] = dict()
                for action in self.environment.actions:
                    # each action is a key in the dictionary for each state
                    self.q_table[(i, j)][action] = np.random.uniform(0, 1) # random value between 0 and 1
        # Initialize the terminal states
        for terminal_state in self.environment.terminal_states:
            # set the Q-value for terminal states to 0
            self.q_table[terminal_state] = dict()
            for action in self.environment.actions:
                self.q_table[terminal_state][action] = 0 
                

    def get_action(self, state, explore=True):
        """
        Returns the optimal action from Q-Value table. 
        Performs epsilon greedy exploration if explore is True (during training)
        If multiple optimal actions, chooses random choice.
        """
        if np.random.uniform(0,1) < self.epsilon and explore:
            # Choose a random action with epsilon probability
            action = np.random.choice(self.environment.actions)
        else:
            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
            action = np.random.choice(best_actions)
        return action
    
    def update(self, state, reward, next_state, action):
        """
        Updates the Q-value tables using Q-learning
        """
        if next_state in self.environment.terminal_states:
        # No future rewards for terminal states
            max_q_next = 0
        else:
            max_q_next = max(self.q_table[next_state].values())
        
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * max_q_next - self.q_table[state][action])
    
    def visualize_q_values(self, title, SCALE=100):
        """
        In the grid, plot the arrow showing the best action to take in each state.
        """
        plt.figure()
        # Display grid (red as bomb, yellow as gold)
        img = np.zeros((self.environment.height*SCALE, self.environment.width*SCALE, 3)) + 0.01
        for bomb_location in self.environment.bomb_locations:
            img[bomb_location[0]*SCALE:bomb_location[0]*SCALE+SCALE, bomb_location[1]*SCALE:bomb_location[1]*SCALE+SCALE] = [1, 0, 0]
        img[self.environment.gold_location[0]*SCALE:self.environment.gold_location[0]*SCALE+SCALE, self.environment.gold_location[1]*SCALE:self.environment.gold_location[1]*SCALE+SCALE] = [1, 1, 0]
        
        # Display best actions (print as text), show as blue arrow
        for i in range(self.environment.height):
            for j in range(self.environment.width):
                # skip terminal states
                if (i, j) in self.environment.terminal_states:
                    continue
                state = (i, j)
                best_action = self.get_action(state)
                if best_action == 'UP':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, 0, -40, color='blue', head_width=10)
                elif best_action == 'DOWN':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, 0, 40, color='blue', head_width=10)
                elif best_action == 'LEFT':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, -40, 0, color='blue', head_width=10)
                elif best_action == 'RIGHT':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, 40, 0, color='blue', head_width=10)
        
        plt.imshow(img)
        plt.xticks(np.arange(0, self.environment.width*SCALE, SCALE), np.arange(0, self.environment.width, 1))
        plt.yticks(np.arange(0, self.environment.height*SCALE, SCALE), np.arange(0, self.environment.height, 1))
        plt.title(title)
        plt.savefig(f'{title}.png')