# imports
import numpy as np
import pandas as pd

# constant definitions
EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1

GRID_HEIGHT = 20
GRID_WIDTH = 10

TRAIN_EPISODES = 1000

# all possible sequence of actions
# for shape names see https://www.quora.com/What-are-the-different-blocks-in-Tetris-called-Is-there-a-specific-name-for-each-block
# TODO: generate all combinations of action sequences for each shape
ACTIONS = dict()
ACTIONS['O'] = [['left'] * 4, ['left'] * 3, ['left'] * 2, ['left'],
                [],
                ['right'] * 4, ['right'] * 3, ['right'] * 2, ['right']
                ]
ACTIONS['I'] = [['left'] * 5, ['left'] * 4, ['left'] * 3, ['left'] * 2, ['left'],
                [],
                ['right'] * 4, ['right'] * 3, ['right'] * 2, ['right'],
                ['rotate'] + ['left'] * 5, ['rotate'] + ['left'] * 4, ['rotate'] + ['left'] * 3,
                ['rotate'] + ['left'] * 2, ['rotate'] + ['left'],
                ['rotate'],
                ['rotate'] + ['right']
                ]
ACTIONS['L'] = [['left'] * 4, ['left'] * 3, ['left'] * 2, ['left'],
                [],
                ['right'] * 4, ['right'] * 3, ['right'] * 2, ['right'],
                ['rotate'] + ['left'] * 4, ['rotate'] + ['left'] * 3, ['rotate'] + ['left'] * 2, ['rotate'] + ['left'],
                ['rotate'],
                ['rotate'] + ['right'] * 3, ['rotate'] + ['right'] * 2, ['rotate'] + ['right'],
                ['rotate'] * 2 + ['left'] * 4, ['rotate'] * 2 + ['left'] * 3, ['rotate'] * 2 + ['left'] * 2,
                ['rotate'] * 2 + ['left'],
                ['rotate'] * 2,
                ['rotate'] * 2 + ['right'] * 4, ['rotate'] * 2 + ['right'] * 3, ['rotate'] * 2 + ['right'] * 2,
                ['rotate'] * 2 + ['right'],
                ['rotate'] * 3 + ['left'] * 4, ['rotate'] * 3 + ['left'] * 3, ['rotate'] * 3 + ['left'] * 2,
                ['rotate'] * 3 + ['left'],
                ['rotate'] * 3,
                ['rotate'] * 3 + ['right'] * 3, ['rotate'] * 3 + ['right'] * 2, ['rotate'] * 3 + ['right']
                ]
ACTIONS['J'] = [['left'] * 4, ['left'] * 3, ['left'] * 2, ['left'],
                [],
                ['right'] * 4, ['right'] * 3, ['right'] * 2, ['right'],
                ['rotate'] + ['left'] * 4, ['rotate'] + ['left'] * 3, ['rotate'] + ['left'] * 2, ['rotate'] + ['left'],
                ['rotate'],
                ['rotate'] + ['right'] * 3, ['rotate'] + ['right'] * 2, ['rotate'] + ['right'],
                ['rotate'] * 2 + ['left'] * 4, ['rotate'] * 2 + ['left'] * 3, ['rotate'] * 2 + ['left'] * 2,
                ['rotate'] * 2 + ['left'],
                ['rotate'] * 2,
                ['rotate'] * 2 + ['right'] * 4, ['rotate'] * 2 + ['right'] * 3, ['rotate'] * 2 + ['right'] * 2,
                ['rotate'] * 2 + ['right'],
                ['rotate'] * 3 + ['left'] * 4, ['rotate'] * 3 + ['left'] * 3, ['rotate'] * 3 + ['left'] * 2,
                ['rotate'] * 3 + ['left'],
                ['rotate'] * 3,
                ['rotate'] * 3 + ['right'] * 3, ['rotate'] * 3 + ['right'] * 2, ['rotate'] * 3 + ['right']
                ]
ACTIONS['S'] = [['left'] * 4, ['left'] * 3, ['left'] * 2, ['left'],
                [],
                ['right'] * 3, ['right'] * 2, ['right'],
                ['rotate'] + ['left'] * 4, ['rotate'] + ['left'] * 3, ['rotate'] + ['left'] * 2, ['rotate'] + ['left'],
                ['rotate'],
                ['rotate'] + ['right'] * 4, ['rotate'] + ['right'] * 3, ['rotate'] + ['right'] * 2,
                ['rotate'] + ['right'],
                ]
ACTIONS['T'] = [['left'] * 4, ['left'] * 3, ['left'] * 2, ['left'],
                [],
                ['right'] * 3, ['right'] * 2, ['right'],
                ['rotate'] + ['left'] * 4, ['rotate'] + ['left'] * 3, ['rotate'] + ['left'] * 2, ['rotate'] + ['left'],
                ['rotate'],
                ['rotate'] + ['right'] * 4, ['rotate'] + ['right'] * 3, ['rotate'] + ['right'] * 2,
                ['rotate'] + ['right'],
                ['rotate'] * 2 + ['left'] * 4, ['rotate'] * 2 + ['left'] * 3, ['rotate'] * 2 + ['left'] * 2,
                ['rotate'] * 2 + ['left'],
                ['rotate'] * 2,
                ['rotate'] * 2 + ['right'] * 3, ['rotate'] * 2 + ['right'] * 2, ['rotate'] * 2 + ['right'],
                ['rotate'] * 3 + ['left'] * 4, ['rotate'] * 3 + ['left'] * 3, ['rotate'] * 3 + ['left'] * 2,
                ['rotate'] * 3 + ['left'],
                ['rotate'] * 3,
                ['rotate'] * 3 + ['right'] * 4, ['rotate'] * 3 + ['right'] * 3, ['rotate'] * 3 + ['right'] * 2,
                ['rotate'] * 3 + ['right']
                ]
ACTIONS['Z'] = [['left'] * 4, ['left'] * 3, ['left'] * 2, ['left'],
                [],
                ['right'] * 3, ['right'] * 2, ['right'],
                ['rotate'] + ['left'] * 4, ['rotate'] + ['left'] * 3, ['rotate'] + ['left'] * 2, ['rotate'] + ['left'],
                ['rotate'],
                ['rotate'] + ['right'] * 4, ['rotate'] + ['right'] * 3, ['rotate'] + ['right'] * 2,
                ['rotate'] + ['right'],
                ]

SHAPES = ['O', 'I', 'J', 'L', 'S', 'T', 'Z']

# all possible shapes

# variable definitions

# Q(st, at)
# st = [s1,...,s10], 0<=si<=4
# at = [0,1,2,3,4,5] = [left, right, turn1, turn2, turn3, no move']
# Q_values
Q_values = dict()
Q_values['O'] = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 5, 5, len(ACTIONS['O'])))
Q_values['I'] = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 5, 5, len(ACTIONS['I'])))
Q_values['J'] = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 5, 5, len(ACTIONS['J'])))
Q_values['L'] = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 5, 5, len(ACTIONS['L'])))
Q_values['S'] = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 5, 5, len(ACTIONS['S'])))
Q_values['T'] = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 5, 5, len(ACTIONS['T'])))
Q_values['Z'] = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 5, 5, len(ACTIONS['J'])))

# test matrix
test_mat = np.array(pd.read_csv('testData/test_mat.csv', header=None))
terminal_mat = np.array(pd.read_csv('testData/terminal_mat.csv', header=None))


# function definitions
def is_terminal_state(st):
    """
    :param st: 20X10 matrix of current tetris board

    :return: boolean if matrix represents a terminal state of the Tetris game
    """
    if any(st[0]) == 1:
        return True
    else:
        return False


def encode_state(st):
    """
    :param st: 20X10 representation of current tetris board as list
    :return: 1x10 representation of top 4 rows of current tetris board as numpy array
    """
    # convert to numpy array
    st = np.array(st)
    # locate highest block in each column
    reduced_state = []
    max_height = 1

    for i in range(GRID_WIDTH):
        col = st[:, i][::-1]
        highest_in_column = 0
        for j, _ in enumerate(col):
            if _ != 0:
                highest_in_column = j + 1
        reduced_state.append(highest_in_column)
        max_height = max(max_height, highest_in_column)

    baseline = max(max_height - 4, 0)
    reduced_state = [max(x - baseline, 0) for x in reduced_state]

    return reduced_state


def get_next_state(st, shape, action_index):
    """
    returns the complete 10x20 board after shape is dropped after taking action_index
    :param st:
    :param shape:
    :param action_index:
    :return:
    """
    old_st = st
    new_st = st
    # TODO: update state based on action_index and shape
    return new_st


def get_reward(old_board, new_board):
    # TODO: calculate reward based on difference between old and new boards
    reward = 0
    return reward


def get_new_random_shape():
    return np.random.choice(SHAPES)


def get_next_action(reduced_state, shape):
    if np.random.random() < EPSILON:
        # return the action that maximizes Q value of given reduced state
        return np.argmax(Q_values[shape][tuple(reduced_state)])
    else:
        # return a random action
        return np.random.randint(len(ACTIONS[shape]))


def train():
    for episode in range(TRAIN_EPISODES):
        # start a new board
        st = np.zeros((20, 10))
        while not is_terminal_state(st):
            # reduced state representation of st by encoding based on its top 4 lines:
            old_reduced_state = encode_state(st)

            # generate a random piece of tetriminoe
            shape = get_new_random_shape()

            # get a sequence of action from list of ACTIONS
            action_index = get_next_action(old_reduced_state, shape)

            # get the next state based on current state and action chosen
            new_st = get_next_state(st, shape, action_index)
            new_reduced_state = encode_state(new_st)
            reward = get_reward(st, new_st)

            # update Q value based on temporal difference
            old_q_value = Q_values[shape][tuple(old_reduced_state), action_index]
            temporal_difference = reward + (GAMMA * np.max(Q_values[shape][tuple(new_reduced_state)])) - old_q_value
            new_q_value = old_q_value + ALPHA * temporal_difference
            Q_values[shape][tuple(new_reduced_state)] = new_q_value

            # transition into new st
            st = new_st
