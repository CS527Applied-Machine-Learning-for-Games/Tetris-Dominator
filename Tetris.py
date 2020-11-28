import numpy as np
from copy import copy, deepcopy
from collections import Counter
from random import shuffle

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


class Tetris(object):

    def __init__(self, height=20, width=10):
        self.height = height
        self.width = width

        self.board = np.zeros((20, 10))
        self.score = 0
        self.fitness = 0
        self.pieces = 0
        self.cleared_lines = 0

        self.bag = copy(SHAPES)
        shuffle(self.bag)

    def get_random_piece(self):
        piece = self.bag.pop()
        if self.bag:
            return piece
        else:
            self.bag = copy(SHAPES)
            shuffle(self.bag)
            return piece

    def reset(self):
        self.board = np.zeros((20, 10))
        self.score = 0
        self.fitness = 0
        self.pieces = 0
        self.cleared_lines = 0

        self.bag = copy(SHAPES)
        shuffle(self.bag)
        return self.get_features(self.board)

    def is_terminal(self):
        if any(self.board[0]) == 1:
            return True
        else:
            return False

    def get_rotated_coordinates(self, shape, n):
        if n == 0:
            return deepcopy(SHAPE_STARTING_COORDS[shape])
        elif shape == 'O':
            return deepcopy(SHAPE_STARTING_COORDS[shape])
        elif shape == 'I':
            if n == 1:
                return [(self.height - 1, 5), (self.height - 1, 6), (self.height - 1, 7), (self.height - 1, 8)]
            else:
                raise Exception
        elif shape == 'L':
            if n == 1:
                return [(self.height - 2, 4), (self.height - 2, 5), (self.height - 1, 6), (self.height - 2, 6)]
            elif n == 2:
                return [(self.height - 1, 4), (self.height - 1, 5), (self.height - 2, 5), (self.height - 3, 5)]
            elif n == 3:
                return [(self.height - 1, 4), (self.height - 2, 4), (self.height - 1, 5), (self.height - 1, 6)]
            else:
                raise Exception
        elif shape == 'J':
            if n == 1:
                return [(self.height - 1, 4), (self.height - 1, 5), (self.height - 1, 6), (self.height - 2, 6)]
            elif n == 2:
                return [(self.height - 1, 4), (self.height - 2, 4), (self.height - 3, 4), (self.height - 1, 5)]
            elif n == 3:
                return [(self.height - 1, 4), (self.height - 2, 4), (self.height - 2, 5), (self.height - 2, 6)]
            else:
                raise Exception
        elif shape == 'S':
            if n == 1:
                return [(self.height - 1, 4), (self.height - 2, 4), (self.height - 2, 5), (self.height - 3, 5)]
            else:
                raise Exception
        elif shape == 'Z':
            if n == 1:
                return [(self.height - 2, 4), (self.height - 3, 4), (self.height - 1, 5), (self.height - 2, 5)]
            else:
                raise Exception
        elif shape == 'T':
            if n == 1:
                return [(self.height - 1, 4), (self.height - 2, 4), (self.height - 3, 4), (self.height - 2, 5)]
            elif n == 2:
                return [(self.height - 2, 4), (self.height - 1, 5), (self.height - 2, 5), (self.height - 2, 6)]
            elif n == 3:
                return [(self.height - 2, 4), (self.height - 1, 5), (self.height - 2, 5), (self.height - 3, 5)]
            else:
                raise Exception
        else:
            raise Exception

    def get_terminal_position_before_drop(self, shape, action):
        """return coordinates of <shape> after it has moved through the sequence of <action>"""
        # create a counter for number of rotations and left/right moves
        action_counter = Counter(action)
        n_left = action_counter['left']
        n_rotate = action_counter['rotate']
        n_right = action_counter['right']

        terminal_position_before_drop = self.get_rotated_coordinates(shape, n_rotate)
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

    def get_next_state(self, shape, action_index):
        # returns new_st before any line cancellations

        # make a deepcopy of new state so that we still have the configurations of the old state to compare
        new_board = deepcopy(self.board)

        action = ACTIONS[shape][action_index]
        terminal_position_before_drop = self.get_terminal_position_before_drop(shape, action)
        # <col:lowest_coord_in_col> for shape
        shape_bottom_coords = dict()
        for row, col in terminal_position_before_drop:
            try:
                shape_bottom_coords[col] = min(shape_bottom_coords[col], row)
            except KeyError:
                shape_bottom_coords[col] = row

        # <col:highest_coord_in_col> for board
        board_top_height = dict()
        for col in range(len(self.board[0])):
            if np.where(self.board[:, col] == 1)[0].size == 0:
                board_top_height[col] = -1
            else:
                board_top_height[col] = self.height - 1 - np.where(self.board[:, col] == 1)[0][0]

        # determine minimum gap between shape and current tetris board, and corresponding column
        min_gap = self.height
        for col in shape_bottom_coords.keys():
            if shape_bottom_coords[col] - board_top_height[col] < min_gap:
                min_gap = shape_bottom_coords[col] - board_top_height[col]

        # bring all columns of tetriminoe down by min_gap
        for i, j in terminal_position_before_drop:
            # convert to vertical coordinates for st
            i = self.height - 1 - i
            # fill board
            new_board[i + min_gap - 1][j] = 1

        return new_board

    def update_board(self, shape, action_index):
        new_board = self.get_next_state(shape, action_index)

        # returns board after complete lines have been deleted

        # detect any complete lines and cancel them if any
        new_board = new_board[np.where(np.count_nonzero(new_board, axis=1) < 10)]

        # lines cancelled
        lines_cancelled = 0
        if len(new_board) != self.height:
            lines_cancelled = self.height - len(new_board)
            new_board = np.insert(new_board, 0, [np.zeros(10)] * lines_cancelled, 0)

        old_fitness = self.fitness

        self.board = new_board
        self.cleared_lines += lines_cancelled
        score = 100 * 2 ** (lines_cancelled - 1) if lines_cancelled else 0
        self.score += score
        self.fitness += score / 100
        if self.is_terminal():
            self.fitness -= 2  # 2 punishment for dying
        else:
            self.fitness += 1  # reward 1 pts for surviving

        return self.fitness - old_fitness

    def get_features(self, board):
        # extracts features of the board defined in the EDD
        # current feature space:
        # genes[i] = [aggregate_height, bumpiness, complete_lines, aggregate_holes]

        max_height, aggregate_height, bumpiness, complete_lines, aggregate_holes = 0, 0, 0, 0, 0

        heights = []
        prev_height = None

        for col in range(self.width):
            try:
                height = np.max(np.nonzero(np.flip(board[:, col]))) + 1
            except ValueError:
                height = 0
            if col == 0:
                prev_height = height
            else:
                bumpiness += abs(height - prev_height)
                prev_height = height
            heights.append(height)
            holes = height - sum(board[:, col])
            aggregate_holes += holes
            max_height = max(max_height, height)

        aggregate_height = sum(heights)
        complete_lines = sum(np.count_nonzero(self.board, axis=1) == self.height)

        return [max_height, aggregate_height, bumpiness, complete_lines, aggregate_holes]
