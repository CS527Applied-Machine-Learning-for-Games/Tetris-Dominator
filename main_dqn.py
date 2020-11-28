from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask import request
import utils
import numpy as np
from Tetris import Tetris
import torch

app = Flask(__name__)
app._static_folder = "./templates/static"
CORS(app, resources=r'/*')

GRID_HEIGHT = 20
GRID_WIDTH = 10

torch.manual_seed(42)
model = torch.load('{}tetris_model_4600_epochs'.format('trained_models/'))
model.eval()
env = Tetris()
env.reset()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tetris/test')
def hello():
    return 'Hello World!'


@app.route('/tetris/next', methods=['POST'])
def get_next_step():
    params = request.get_json()
    board = params[0]
    st = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0:
                st[i][j] = 1
    shape = params[1]["flag"]

    env.board = st

    next_states = []
    for action_index in range(len(utils.ACTIONS[shape])):
        next_states.append(torch.FloatTensor(env.get_features(env.get_next_state(shape, action_index))))
    next_states = torch.stack(next_states)

    predictions = model(next_states)[:, 0]
    chosen_index = torch.argmax(predictions).item()

    # print(shape, st)
    action = utils.ACTIONS[shape][chosen_index]
    return jsonify(action)


if __name__ == '__main__':
    app.run()
