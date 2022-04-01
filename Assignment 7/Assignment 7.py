import numpy as np
import matplotlib.pyplot as plt


# n×m gridworld
class GridWorld:
    # value function for the entire gridworld
    value = 0

    def __init__(self, size, terminal={}, inaccessible={}, default_reward = -1):
        self.n = size[0]
        self.m = size[1]
        self.terminal = terminal
        self.inaccessible = inaccessible
        self.default_reward = default_reward

    # issuing the next state based on a given action.
    def next_state(self, action):
        pass
    
    # issuing a reward based on the state the agent is in.
    def get_reward(self, state):
        if state in self.terminal.keys():
            return self.terminal.state
        else:
            return self.default_reward

# agent    
class Agent:

    def __init__(self, world):
        self.world = world

    # based on the current state will decide which action to take next (greedy action selection)
    def action(self, state):
        pass

    # value iteration
    def value_iter(self):
        pass




if __name__ == '__main__':

    # Table 1
    size = (4, 4)
    inaccessible_states = [(1, 1), (1, 2), (2, 3)] # color=Gray
    terminal_states = {(0, 3): 0, (3, 3): 0} # color=Blue, Blue

    gw_1 = GridWorld(size, terminal_states, inaccessible_states)


    # Table 2
    size = (5, 6)
    inaccessible_states = [(0, 1), (1, 1), (2, 1), (2, 3), (2, 4), (2, 5), ((3, 1))] # color=Gray
    terminal_states = {(0, 5): 10, (3, 3): -10} # color=Green, Blue

    gw_2 = GridWorld(size, terminal_states, inaccessible_states)

