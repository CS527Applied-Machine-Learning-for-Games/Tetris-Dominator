import numpy as np
from sys import maxsize as MAXSIZE
from copy import deepcopy
from collections import Counter


GRID_HEIGHT = 20
GRID_WIDTH = 10

def encode_state(st, n):
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

    baseline = max(max_height - n, 0)
    reduced_state = [max(x - baseline, 0) for x in reduced_state]

    return reduced_state

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
SHAPE_STARTING_COORDS['O'] = [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-2, 5)]
SHAPE_STARTING_COORDS['I'] = [(GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-3, 5), (GRID_HEIGHT-4, 5)]
SHAPE_STARTING_COORDS['L'] = [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-3, 4), (GRID_HEIGHT-3, 5)]
SHAPE_STARTING_COORDS['J'] = [(GRID_HEIGHT-3, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-3, 5)]
SHAPE_STARTING_COORDS['S'] = [(GRID_HEIGHT-2, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-1, 6)]
SHAPE_STARTING_COORDS['Z'] = [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-2, 6)]
SHAPE_STARTING_COORDS['T'] = [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-1, 6)]

#%% testing phase
def get_next_action_state_test(st, shape, gene = [-0.01578,-0.2882,-0.1177,0.01883,-0.9869]):
    # decides next best action for st given the shape and gene (heuristic vector)
    # also returns state after this action
    max_score = -MAXSIZE
    arg_max = 0
    for action_index in range(len(ACTIONS[shape])):
        new_st = get_next_state(st, shape, action_index)
        feature_vals = get_features(new_st)
        heuristic_score = np.dot(gene, feature_vals)
        if heuristic_score > max_score:
            arg_max = action_index
            max_score = heuristic_score
    new_st = get_next_state(st, shape, arg_max)
    new_st, lines_cancelled = update_board(new_st)
    
    return arg_max

def get_next_state(st, shape, action_index):
    # a slightly modified version of getting next state vs. the Q-learning method. Namely the fitness function has changed
    # returns new_st before any line cancellations
    
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
            board_top_height[col] = -1
        else:
            board_top_height[col] = len(st) - 1 - np.where(st[:, col] == 1)[0][0]

    # determine minimum gap between shape and current tetris board, and corresponding column
    min_gap = GRID_HEIGHT
    for col in shape_bottom_coords.keys():
        if shape_bottom_coords[col] - board_top_height[col] < min_gap:
            min_gap = shape_bottom_coords[col] - board_top_height[col]

    # bring all columns of tetriminoe down by min_gap
    for i, j in terminal_position_before_drop:
        # convert to vertical coordinates for st
        i = GRID_HEIGHT - 1 - i
        # fill board
        new_st[i + min_gap - 1][j] = 1
    
    return new_st 

def get_features(st):
    # extracts features of the board defined in the EDD
    # current feature space:
    # genes[i] = [aggregate_height, bumpiness, complete_lines, aggregate_holes]
    
    max_height, aggregate_height, bumpiness, complete_lines, aggregate_holes =  0, 0, 0, 0, 0
    
    heights = []
    prev_height = None
    
    for col in range(GRID_WIDTH):
        try:
            height = np.max(np.nonzero(np.flip(st[:, col]))) + 1
        except ValueError:
            height = 0
        if col == 0:
            prev_height = height
        else:
            bumpiness += abs(height - prev_height)
            prev_height = height
        heights.append(height)
        holes = height - sum(st[:,col])
        aggregate_holes += holes    
        max_height = max(max_height, height)
    
    aggregate_height = sum(heights)
    complete_lines = sum(np.count_nonzero(st, axis = 1) == GRID_WIDTH)
    
    return [max_height**2, aggregate_height, bumpiness, complete_lines, aggregate_holes]

def update_board(st):
    # returns board after complete lines have been deleted
    
    # detect any complete lines and cancel them if any
    st = st[np.where(np.count_nonzero(st, axis=1) < 10)]
    
    # lines cancelled
    lines_cancelled = 0
    if len(st) != GRID_HEIGHT:
        lines_cancelled = GRID_HEIGHT - len(st)
        st = np.insert(st, 0, [np.zeros(10)]*lines_cancelled, 0)
        
    return st, lines_cancelled

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
            return [(GRID_HEIGHT-1, 5), (GRID_HEIGHT-1, 6), (GRID_HEIGHT-1, 7), (GRID_HEIGHT-1, 8)]
        else:
            raise Exception
    elif shape == 'L':
        if n == 1:
            return [(GRID_HEIGHT-2, 4), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-1, 6), (GRID_HEIGHT-2, 6)]
        elif n == 2:
            return [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-3, 5)]
        elif n == 3:
            return [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-1, 6)]
        else:
            raise Exception
    elif shape == 'J':
        if n == 1:
            return [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-1, 6), (GRID_HEIGHT-2, 6)]
        elif n == 2:
            return [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-3, 4), (GRID_HEIGHT-1, 5)]
        elif n == 3:
            return [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-2, 6)]
        else:
            raise Exception
    elif shape == 'S':
        if n == 1:
            return [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-3, 5)]
        else:
            raise Exception
    elif shape == 'Z':
        if n == 1:
            return [(GRID_HEIGHT-2, 4), (GRID_HEIGHT-3, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5)]
        else:
            raise Exception
    elif shape == 'T':
        if n == 1:
            return [(GRID_HEIGHT-1, 4), (GRID_HEIGHT-2, 4), (GRID_HEIGHT-3, 4), (GRID_HEIGHT-2, 5)]
        elif n == 2:
            return [(GRID_HEIGHT-2, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-2, 6)]
        elif n == 3:
            return [(GRID_HEIGHT-2, 4), (GRID_HEIGHT-1, 5), (GRID_HEIGHT-2, 5), (GRID_HEIGHT-3, 5)]
        else:
            raise Exception
    else:
        raise Exception