from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/tetris/test')
def hello():
    return 'Hello World!'


@app.route('/tetris/next/<board>', methods=['GET'])
def get_next_step(board):
    print(board)
    return jsonify(board)


if __name__ == '__main__':
    app.run()
