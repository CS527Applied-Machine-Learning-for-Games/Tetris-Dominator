# imports
import numpy as np
import pandas as pd
from collections import Counter
from copy import deepcopy

# constant definitions
EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1

GRID_HEIGHT = 20
GRID_WIDTH = 10

TRAIN_EPISODES = 1000

# all possible sequence of actions
# for shape names see https://www.quora.com/What-are-the-different-blocks-in-Tetris-called-Is-there-a-specific-name-for-each-block
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

SHAPE_STARTING_COORDS = dict()
SHAPE_STARTING_COORDS['O'] = [(19, 4), (19, 5), (18, 4), (18, 5)]
SHAPE_STARTING_COORDS['I'] = [(19, 5), (18, 5), (17, 5), (16, 5)]
SHAPE_STARTING_COORDS['L'] = [(19, 4), (18, 4), (17, 4), (17, 5)]
SHAPE_STARTING_COORDS['J'] = [(17, 4), (19, 5), (18, 4), (17, 5)]
SHAPE_STARTING_COORDS['S'] = [(18, 4), (19, 5), (18, 5), (19, 6)]
SHAPE_STARTING_COORDS['Z'] = [(19, 4), (19, 5), (18, 5), (18, 6)]
SHAPE_STARTING_COORDS['T'] = [(19, 4), (19, 5), (18, 5), (19, 6)]

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


def get_rotated_coordinates(shape, n):
    """
    n: number of rotations
    """
    if n == 0:
        return deepcopy(SHAPE_STARTING_COORDS[shape])
    elif shape == 'O':
        return deepcopy(SHAPE_STARTING_COORDS[shape])
    elif shape == 'I':
        if n == 1:
            return [(19, 4), (19, 5), (19, 6), (19, 7)]
        else:
            raise Exception
    elif shape == 'L':
        if n == 1:
            return [(18, 4), (18, 5), (19, 6), (18, 6)]
        elif n == 2:
            return [(19, 4), (19, 5), (18, 5), (17, 5)]
        elif n == 3:
            return [(19, 4), (18, 4), (19, 5), (19, 6)]
        else:
            raise Exception
    elif shape == 'J':
        if n == 1:
            return [(19, 4), (19, 5), (19, 6), (18, 6)]
        elif n == 2:
            return [(19, 4), (18, 4), (17, 4), (19, 5)]
        elif n == 3:
            return [(19, 4), (18, 4), (18, 5), (18, 6)]
        else:
            raise Exception
    elif shape == 'S':
        if n == 1:
            return [(19, 4), (18, 4), (18, 5), (17, 5)]
        else:
            raise Exception
    elif shape == 'Z':
        if n == 1:
            return [(18, 4), (17, 4), (19, 5), (18, 5)]
        else:
            raise Exception
    elif shape == 'T':
        if n == 1:
            return [(19, 4), (18, 4), (17, 4), (18, 5)]
        elif n == 2:
            return [(18, 4), (19, 5), (18, 5), (18, 6)]
        elif n == 3:
            return [(18, 4), (19, 5), (18, 5), (17, 5)]
        else:
            raise Exception
    else:
        raise Exception


def get_terminal_position_before_drop(shape, action):
    """return coordinates of <shape> after it has moved through the sequence of <action>"""
    # create a counter for number of rotations and left/right moves
    action_counter = Counter(action)
    n_left = action_counter['left']
    n_rotate = action_counter['rotate']
    n_right = action_counter['right']

    terminal_position_before_drop = get_rotated_coordinates(shape, n_rotate)
    if n_left:
        # move left n_left times
        for i in range(len(terminal_position_before_drop)):
            (x, y) = terminal_position_before_drop[i]
            terminal_position_before_drop[i] = (x, y - n_left)
        # move right n_right times
    if n_right:
        for i in range(len(terminal_position_before_drop)):
            (x, y) = terminal_position_before_drop[i]
            terminal_position_before_drop[i] = (x, y + n_right)

    return terminal_position_before_drop


def get_next_state(st, shape, action_index):
    """
    gets next state based on current state, and terminal configuration of tetris piece before it is dropped
    """
    # make a deepcopy of new state so that we still have the configurations of the old state to compare
    new_st = deepcopy(st)

    action = ACTIONS[shape][action_index]
    terminal_position_before_drop = get_terminal_position_before_drop(shape, action)

    # <col:lowest_coord_in_col> for shape
    shape_bottom_coords = dict()
    for row, col in terminal_position_before_drop:
        try:
            shape_bottom_coords[col] = min(shape_bottom_coords[col], row)
        except KeyError:
            shape_bottom_coords[col] = row

    # <col:highest_coord_in_col> for board
    board_top_height = dict()
    for col in range(len(st[0])):
        if np.where(st[:, col] == 1)[0].size == 0:
            board_top_height[col] = 0
        else:
            board_top_height[col] = len(st) - 1 - np.where(st[:, col] == 1)[0][0]

    # determine minimum gap between shape and current tetris board, and corresponding column
    min_gap = 20
    for col in shape_bottom_coords.keys():
        if shape_bottom_coords[col] - board_top_height[col] < min_gap:
            min_gap = shape_bottom_coords[col] - board_top_height[col]

    # bring all columns of tetriminoe down by min_gap
    for i, j in terminal_position_before_drop:
        # convert to vertical coordinates for st
        i = 19 - i
        # fill board
        new_st[i + min_gap - 1][j] = 1

    # detect any complete lines and cancel them if any
    new_st = new_st[np.where(np.count_nonzero(new_st, axis=1) < 10)]

    if len(new_st) != len(st):
        lines_cancelled = len(st) - len(new_st)
        new_st = np.insert(new_st, 0, [np.zeros(10)]*lines_cancelled, 0)
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
