from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import utils
import numpy as np

app = Flask(__name__)
CORS(app, resources=r'/*')

GRID_HEIGHT = 20
GRID_WIDTH = 10

# Q_values = dict()
# for shape in utils.SHAPES:
#     Q_values[shape] = np.load('./Q_mat_' + shape + '.npy')

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
    # reduced_state = utils.encode_state(board, 4)
    shape = params[1]["flag"]
    print(shape, st)
    # action_index = np.argmax(Q_values[shape][tuple(reduced_state)])
    action_index = utils.get_next_action_state_test(st, shape)
    action = utils.ACTIONS[shape][action_index]
    return jsonify(action)


if __name__ == '__main__':
    app.run()
