#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:31:12 2020

@author: mzp06256
"""

import argparse
import torch
import matplotlib.pyplot as plt
from Tetris import Tetris, ACTIONS, SHAPES, SHAPE_STARTING_COORDS
from dqn_agent import DeepQNetwork
from tensorboardX import SummaryWriter
from collections import deque
import numpy as np
from copy import deepcopy

from random import random, sample, randint


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_initial', type=float, default=1)
    parser.add_argument('--eps_final', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_decay_epochs', type=int, default=2000)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--replay_mem_size', type=int, default=30000, help='Number of epochs between testing phases')
    parser.add_argument('--log_path', type=str, default='log/')
    parser.add_argument('--save_path', type=str, default='trained_models/')
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()

    return args


def train(args):
    scores = []
    torch.manual_seed(42)

    writer = SummaryWriter(args.log_path)
    env = Tetris()
    model = DeepQNetwork()
    fixed_q_model = deepcopy(model)
    fixed_q_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    st = torch.FloatTensor(env.get_features(env.board))

    replay_mem = deque(maxlen=args.replay_mem_size)
    epoch = 0

    while epoch < args.num_epochs:
        eps = args.eps_final + (
                max(args.num_decay_epochs - epoch, 0) * (args.eps_initial - args.eps_final) / args.num_decay_epochs)
        random_action = random() < eps
        shape = env.get_random_piece()
        next_states = []
        for action_index in range(len(ACTIONS[shape])):
            next_states.append(torch.FloatTensor(env.get_features(env.get_next_state(shape, action_index))))
        next_states = torch.stack(next_states)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random_action:
            chosen_index = randint(0, len(ACTIONS[shape]) - 1)
        else:
            chosen_index = np.argmax(predictions).item()

        next_state = next_states[chosen_index, :]
        reward = env.update_board(shape, chosen_index)
        replay_mem.append([st, reward, next_state, env.is_terminal()])

        if env.is_terminal():
            final_score = env.score
            final_fitness = env.fitness
            final_cleared_lines = env.cleared_lines
            scores.append(final_score)
            st = torch.FloatTensor(env.reset())
        else:
            st = next_state
            continue

        # after a game ends check replay memory
        if len(replay_mem) < args.replay_mem_size / 10:
            continue

        epoch += 1
        # sample a batch of state transitions and their respective rewards
        batch = sample(replay_mem, min(len(replay_mem), args.batch_size))

        state_batch, reward_batch, next_state_batch, is_terminal_batch = [], [], [], []

        for i in range(len(batch)):
            state_batch.append(batch[i][0])
            reward_batch.append(batch[i][1])
            next_state_batch.append(batch[i][2])
            is_terminal_batch.append(batch[i][3])

        state_batch = torch.stack(state_batch)
        next_state_batch = torch.stack(next_state_batch)
        reward_batch = torch.tensor(np.array(reward_batch).reshape(-1, 1))

        q_values = model(state_batch)

        with torch.no_grad():
            next_pred_batch = fixed_q_model(next_state_batch)

        y_batch = []
        for i in range(len(next_pred_batch)):
            if is_terminal_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + args.gamma * next_pred_batch[i])
        y_batch = torch.tensor(np.array(y_batch).reshape(-1, 1), dtype=torch.float32)

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            fixed_q_model = deepcopy(model)
        print('Epoch {}/{}, score: {}, lines cleared: {}, fitness: {}, loss: {}'.format(
            epoch,
            args.num_epochs,
            final_score,
            final_cleared_lines,
            final_fitness,
            loss
        ))

        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

    torch.save(model, "{}tetris_model".format(args.save_path))
    return scores


if __name__ == "__main__":
    parsed_args = get_args()
    SCORES = train(parsed_args)
    plt.plot(SCORES)
