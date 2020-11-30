import argparse
import torch
from Tetris import Tetris, ACTIONS
from dqn_agent import DeepQNetwork
from collections import deque
import numpy as np

from random import random, sample, randint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--eps_initial', type=float, default=1)
    parser.add_argument('--eps_final', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_decay_epochs', type=int, default=2000, help='Number of epochs for eps to decay')
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--save_path', type=str, default='trained_models/')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(42)
    SCORES = []

    env = Tetris()
    model = DeepQNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    st = torch.FloatTensor(env.get_features(env.board))

    epoch = 0

    while epoch < args.num_epochs:
        eps = args.eps_final + (
                max(args.num_decay_epochs - epoch, 0) * (args.eps_initial - args.eps_final) / args.num_decay_epochs)
        shape = env.get_random_piece()
        next_states = []
        for action_index in range(len(ACTIONS[shape])):
            next_states.append(torch.FloatTensor(env.get_features(env.get_next_state(shape, action_index))))
        next_states = torch.stack(next_states)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random() < eps:
            # exploration
            chosen_index = randint(0, len(ACTIONS[shape]) - 1)
        else:
            # exploitation
            chosen_index = np.argmax(predictions).item()

        next_state = next_states[chosen_index, :]
        reward = env.update_board(shape, chosen_index)

        if env.is_terminal():
            final_score = env.score
            final_fitness = env.fitness
            final_cleared_lines = env.cleared_lines
            SCORES.append(final_score)

            q_value = model(st)
            q_target = reward
            q_target = torch.tensor(np.array(q_target).reshape(-1, 1), dtype=torch.float32)
            opt.zero_grad()
            loss = criterion(q_value, q_target)
            loss.backward()
            opt.step()

            st = torch.FloatTensor(env.reset())

        else:
            q_value = model(st)
            model.eval()
            with torch.no_grad():
                next_pred = model(next_state)
            model.train()
            q_target = reward + args.gamma * next_pred
            q_target = torch.tensor(np.array(q_target).reshape(-1, 1), dtype=torch.float32)
            opt.zero_grad()
            loss = criterion(q_value, q_target)
            loss.backward()
            opt.step()

            st = next_state
            continue

        # begin training
        epoch += 1

        print('Epoch {} of {}, score: {}, lines cleared: {}, fitness: {}, loss: {}'.format(
            epoch,
            args.num_epochs,
            final_score,
            final_cleared_lines,
            final_fitness,
            loss
        ))

        if epoch > args.num_decay_epochs and epoch % 100 == 0:
            torch.save(model, "{}tetris_model_{}_epochs".format(
                args.save_path,
                str(epoch)
            ))

