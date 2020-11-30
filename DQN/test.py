from train import get_args
from Tetris import Tetris, ACTIONS
import torch
import numpy as np


def test(args):
    torch.manual_seed(43)
    model = torch.load('{}tetris_model_4100_epochs'.format(args.save_path))

    model.eval()
    env = Tetris()
    env.reset()

    while True:
        next_states = []
        shape = env.get_random_piece()
        for action_index in range(len(ACTIONS[shape])):
            next_states.append(torch.FloatTensor(env.get_features(env.get_next_state(shape, action_index))))
        next_states = torch.stack(next_states)

        predictions = model(next_states)[:, 0]
        chosen_index = torch.argmax(predictions).item()

        env.update_board(shape, chosen_index)

        if env.is_terminal():
            break

    print('Total score: {}'.format(env.score))
    return env.score


if __name__ == "__main__":
    parsed_args = get_args()
    SCORES = []
    for i in range(20):
        SCORES.append(test(parsed_args))
    print('average score: {}'.format(np.mean(SCORES)))