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
    parser.add_argument('--num_decay_epochs', type=int, default=4000, help='Number of epochs for eps to decay')
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--replay_mem_size', type=int, default=50000, help='Max length of reply buffer queue')
    parser.add_argument('--save_path', type=str, default='trained_models/')
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(42)

    env = Tetris()
    model = DeepQNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    replay_buffer = deque(maxlen=args.replay_mem_size)

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
        replay_buffer.append([st, reward, next_state, env.is_terminal()])

        if env.is_terminal():
            final_score = env.score
            final_fitness = env.fitness
            final_cleared_lines = env.cleared_lines
            st = torch.FloatTensor(env.reset())

            # we do not start training until the replay buffer is filled
            if len(replay_buffer) < args.replay_mem_size/10:
                continue
        else:
            st = next_state
            continue

        # begin training
        epoch += 1
        # sample a batch of state transitions and their respective rewards
        batch = sample(replay_buffer, min(len(replay_buffer), args.batch_size))

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
        model.eval()
        with torch.no_grad():
            next_pred_batch = model(next_state_batch)
        model.train()

        # calculate target Q values from Bellman equation using next state
        y_batch = []
        for i in range(len(next_pred_batch)):
            if is_terminal_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + args.gamma * next_pred_batch[i])
        y_batch = torch.tensor(np.array(y_batch).reshape(-1, 1), dtype=torch.float32)

        opt.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        opt.step()

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

