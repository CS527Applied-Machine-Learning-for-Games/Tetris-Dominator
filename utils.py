import numpy as np

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
