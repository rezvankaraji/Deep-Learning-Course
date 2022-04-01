import numpy as np

MAX_ITER = 10000

# n×m gridworld
class GridWorld:
    # origin point of the grid_world
    ORIGIN = (0, 0)

    def __init__(self, size, terminal={}, inaccessible={}, default_reward = -1):
        self.size = size # n x m
        self.terminal = terminal
        self.inaccessible = inaccessible
        self.default_reward = default_reward

    # returning the next state based on a given action and current state.
    # input action = (vertical_move, horizontal_move)
    def next_state(self, current_state, action):
        # define next state
        next = (current_state[0]+action[0], current_state[1]+action[1])
        # check validity and accessibility of the next state
        if self.is_valid(next) and self.is_accessible(next):
            return next
        return False
    
    # returning the reward for an state.
    def get_reward(self, state):
        # check validity and accessibility of the state
        if self.is_valid(state) and self.is_accessible(state):        
            if state in self.terminal.keys():
                return self.terminal[state]
            else:
                return self.default_reward
        return False

    # for a state, determins with action in a list of moves is possible
    def possible_actions(self, state, moves):
        # keys=possible actions, values=next state with that action
        actions = {}
        for move in moves:
            next_state = self.next_state(state, move)
            if next_state:
                actions[move] = next_state
        return actions

    # check if a state is in the grid limits
    def is_valid(self, state):
        if (0 <= state[0] < self.size[0]) and (0 <= state[1] < self.size[1]):
            return True
        return False

    # check if a state is not in inaccessible list
    def is_accessible(self, state):
        if state in self.inaccessible:
            return False
        return True

    # check if a state is a terminal
    def is_terminal(self, state):
        if state in self.terminal:
            return True
        return False

    # returns all states in the GridWorld
    def get_states(self):
        return [(x, y) for x in range(self.size[0]) for y in range(self.size[1])]

    def get_origin(self):
        return GridWorld.ORIGIN

# agent    
class Agent:

    def __init__(self, env, starting_state=None, possible_moves = None):
        self.env = env
        self.current_state = self.env.get_origin()
        self.possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] # one cell to right, left, down, up

        if starting_state:
            self.current_state = starting_state

        if possible_moves:
            self.possible_moves = possible_moves

    # value iteration
    def value_iter(self, gamma=0.9, theta=1e-400):
        V = np.zeros(self.env.size) # to save the values of each state
        A = np.empty(self.env.size,dtype=str) # to save the action of each state

        converged = False
        i = 0
        while not converged:
            # keep the values of V
            old_V = V
            # For each state in env
            for state in self.env.get_states():

                if self.env.is_terminal(state):
                    V[state] = 0
                    A[state] = 'T'

                elif not self.env.is_accessible(state):
                    A[state] = '-'

                else:
                    new_v = []
                    new_a = []
                    # get the possible actions and their corresponding destinations based on current state and possible moves of agent
                    possible_actions = self.env.possible_actions(state, self.possible_moves)
                    for action in possible_actions:
                        next_state = possible_actions[action]
                        reward = self.env.get_reward(next_state)
                        new_v.append(reward + gamma * V[next_state])
                        new_a.append(action)

                    best_index = np.argmax(new_v)
                    V[state] = np.round(new_v[best_index],1)

                    mask = {(0, 1):'R', (0, -1):'L', (1, 0):'D', (-1, 0):'U'}
                    A[state] = mask[new_a[best_index]]

            # calculate the maximum amount of the change happend to an state in this iteration
            max_change = np.max(np.abs(np.subtract(old_V, V)))
            converged = True if max_change < theta else False
            i += 1

            if i > MAX_ITER:
                break

        return V, A
           

if __name__ == '__main__':

    # Table 1
    size = (4, 4)
    inaccessible_states = [(1, 1), (1, 2), (2, 3)] # color=Gray
    terminal_states = {(0, 3): 0, (3, 3): 0} # color=Blue, Blue

    env_1 = GridWorld(size, terminal_states, inaccessible_states)


    # Table 2
    size = (5, 6)
    inaccessible_states = [(0, 1), (1, 1), (2, 1), (2, 3), (2, 4), (2, 5), ((3, 1))] # color=Gray
    terminal_states = {(0, 5): 10, (4, 5): -10} # color=Green, Blue

    env_2 = GridWorld(size, terminal_states, inaccessible_states)


    # Agent
    agent_1 = Agent(env_1)
    V_1, A_1 = agent_1.value_iter()
    print('Table 1')
    print('value: \n',np.matrix(V_1))
    print('action: \n',np.matrix(A_1))
    print()

    agent_2 = Agent(env_2)
    V_2, A_2 = agent_2.value_iter()
    print('Table 2')
    print('value: \n',np.matrix(V_2))
    print('action: \n',np.matrix(A_2))
