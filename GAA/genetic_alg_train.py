#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:04:45 2020

@author: mzp06256
"""
#%% imports
import numpy as np
import pandas as pd
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt
from sys import maxsize as MAXSIZE
from time import time
from random import shuffle, random, choice, sample, seed, randint
from datetime import datetime

#%% constant definitions 
INT_GENERATIONS = 50
INT_POPULATION = 100
FLOAT_MUTATION_RATE = 0.05
INT_NUM_FEATURES = 5
INT_GAME_PER_GEN = 5

GRID_HEIGHT = 20
GRID_WIDTH = 10

REPLACE_RATIO = 0.5
REPRODUCE_RATIO = 0.1

# all possible sequence of actions for shape names see
# https://www.quora.com/What-are-the-different-blocks-in-Tetris-called-Is-there-a-specific-name-for-each-block
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
#%% method definitions & imports

    
def normalize(v):
    norm_v = np.linalg.norm(v)
    return [v[i]/norm_v for i in range(len(v))]

def is_terminal_state(st):
    """
    :param st: matrix of current tetris board

    :return: boolean if matrix represents a terminal state of the Tetris game
    """
    if any(st[0]) == 1:
        return True
    else:
        return False
    

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



def initialize_genes(pop_size):
    # initializes pop_size number of genes, each with length = n where n = length of feature space used
    # current feature space:
    # genes[i] = [aggregate_height, bumpiness, complete_lines, aggregate_holes]
    genes = []
    for i in range(pop_size):
        gene = normalize([ -random(), -random(), -random(), random(), -random()])
        genes.append(gene)
    return genes

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
        
def get_next_action_state_train(st, shape, gene):
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
    
    return arg_max, new_st, lines_cancelled
            
def crossover(p1, p2, f1, f2, method = None):
    if not method:
        # average between p1 and p2 
        child = []
        for i in range(len(p1)):
            child.append((p1[i] + p2[i])/2)
        return normalize(child)

    elif method == 'SBC':
        # Simulated Binary Crossover
        eta = 2
        u = random()
        if u <= 0.5:
            beta = (2*u)**(1/(eta+1))
        else:
            beta = 1/(2*(1-u))**(1/(eta+1))
        
        x1 = [0.5*((1+beta)*p1[i] + (1-beta)*p2[i]) for i in range(len(p1))]
        x2 = [0.5*((1-beta)*p1[i] + (1+beta)*p2[i]) for i in range(len(p2))]
        return x1, x2
    
    elif method == 'randCut':
        index = randint(0, len(p1))
        x1 = p1[:index] + p2[index:]
        x2 = p2[:index] + p1[index:]
        return x1, x2
    
def mutate(gene):
    # gives a random 20% variation one of its weights
    i = choice([_ for _ in range(len(gene))])
    gene[i] += (random()*0.4 - 0.2)
    return normalize(gene)

def get_random_shape():
    return choice(SHAPES)


#%% training phase
genes = initialize_genes(INT_POPULATION)
for generation in range(INT_GENERATIONS):
    fitness = [[0 for _ in range(INT_GAME_PER_GEN)] for __ in range(INT_POPULATION)]
    s = datetime.now()
    for i, gene in enumerate(genes):
        print('generation ' + str(generation) +  '.gene ' + str(i) + ' playing...')
        # fitness = score per round
        
        # each gene plays game INT_GAME_PER_GEN times
        for j in range(INT_GAME_PER_GEN):
            score = 0
            count = 0
            st = np.zeros((GRID_HEIGHT, GRID_WIDTH))
            while (not is_terminal_state(st)) and count < 500:
                shapes = list('OIJLSTZ')
                # seed(s)
                shuffle(shapes)
                # seed(datetime.now())
                for shape in shapes:
                    count += 1
                    action_index, new_st,lines_cancelled = get_next_action_state_train(st, shape, gene)
                    if lines_cancelled:
                        score += 100*2**(lines_cancelled-1)
                    st = new_st
                    if is_terminal_state(st) or count > 500:
                        break
                    # print(shape, new_st)
            fitness[i][j] = score
    mean_fitness = [np.mean(fitness[i]) for i in range(len(genes))]
    genes = [(genes[k], mean_fitness[k]) for k in range(INT_POPULATION)]
    
    # end of games. replace bottom of generation with offsprings reproduced using crossover of fittest genes
    
    # print fitness of this generation
    print('\n' + 'gen. ' + str(generation) + ' fitness = ' + str(sorted(mean_fitness)))
    
    # sort in ascending order based on fitness and discard last 50%
    genes.sort(key = lambda x: -x[1])
    
    reproducing_parents = genes[:int(INT_POPULATION*REPRODUCE_RATIO)]
    parent_pairs = []
    for i in range(len(reproducing_parents)):
        for j in range(i+1, len(reproducing_parents)):
            parent_pairs.append((reproducing_parents[i], reproducing_parents[j]))
            
    # parent_pairs = sample(parent_pairs, int(INT_POPULATION*REPLACE_RATIO))
    shuffle(parent_pairs)
    offsprings = []
    for gene1, gene2 in parent_pairs:
        p1, f1 = gene1
        p2, f2 = gene2
        x1, x2 = crossover(p1, p2, f1, f2, 'SBC')
        
        # mutation
        if random() < FLOAT_MUTATION_RATE:
            x1 = mutate(x1)
            x2 = mutate(x2)
        offsprings.append(x1)
        offsprings.append(x2)
    offsprings = sample(offsprings, int(INT_POPULATION*REPLACE_RATIO) )
    genes = [genes[i][0] for i in range(len(genes))]
    genes[-len(offsprings):] = offsprings        
    

#%% testing phase
w1 = [-0.01578,-0.2882,-0.1177,0.01883,-0.9869]
w2 = [-0.02465912687228323,-0.9185600317928028,-0.07957083651311785,0.04704280996011445,-0.38316757086296743]
def get_next_action_state_test(st, shape, gene = w2):
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
    
    return arg_max, new_st, lines_cancelled

SCORES = []
for i in range(100):
    score = 0
    count = 0
    st = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    while (not is_terminal_state(st)):
        shapes = list('OIJLSTZ')
        shuffle(shapes)
        for shape in shapes:
            count += 1
            action_index, new_st,lines_cancelled = get_next_action_state_test(st, shape)
            if lines_cancelled:
                score += 100*2**(lines_cancelled-1)
            st = new_st
            if is_terminal_state(st):
                break
            # print(shape, new_st)
    SCORES.append(score)
    print('round ' + str(i) + ' score = ' + str(score))
print('average score of 100 rounds = ' + str(np.mean(SCORES)))