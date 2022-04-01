# n×m gridworld
class GridWorld:
    # value function for the entire gridworld
    value = 0

    def __init__(self, n, m, terminal={}, inaccessible={}, default_reward = -1):
        self.n = n
        self.m = m
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