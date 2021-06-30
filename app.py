from flask import Flask, render_template, Response, request
from alpha_zero import load_model, get_move_probability, get_move_probability_parallel
from treesearch import MonteCarloTreeSearch
from game import Game, FIRST_PLAYER, SECOND_PLAYER, convert_move_action, convert_action_move, convert_action_move_exists
from hyper_params import MODEL
import chess
import chess.pgn
import multiprocessing
import numpy as np


app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')


@app.route('/test')
def test():
    player_turn = request.args.get('turn')
    return {'test': [1,2,3, player_turn]}


@app.route('/newgame')
def new_game():
    global player, pgn_game, node
    pgn_game = chess.pgn.Game()
    pgn_game.setup(env.board)
    node = pgn_game
    player_turn = request.args.get('turn') == 'true'
    state = env.reset()
    player = MonteCarloTreeSearch(state, FIRST_PLAYER, 'best_player')
    if player_turn == FIRST_PLAYER:
        print('first player')
        legal = []
        for move in env.board.legal_moves:
            legal.append(move.uci())
        return {'move': '', 'legal_uci': legal, 'winner': ''}
    else:
        env.render()
        return bot_move()



@app.route('/makemove')
def make_move():
    global node
    player_uci = request.args.get('uci')
    move = chess.Move.from_uci(player_uci)
    node = node.add_variation(move)
    if env.board.turn == SECOND_PLAYER:
        move = chess.Move(from_square=chess.square_mirror(move.from_square),
                          to_square=chess.square_mirror(move.to_square), promotion=move.promotion)
    action = convert_move_action(move)
    if not player.head.children:
        player.populate_children(player.head, env)
    print(player.head.next_actions)
    state, reward, done, _ = env.step(action, False)
    if reward:
        pi = get_move_probability(env, player, model)
        print(list(map(lambda
                           x: f"p {x[0].stats['p']}, q {-1 * x[0].stats['q']}, move {convert_action_move_exists(x[1], env.board, env.board.turn).uci()}" if abs(x[0].stats['q'])> 0.5 and x[0].stats['n'] else '' ,
                       zip(player.head.children, player.head.next_actions))))
    state, reward, done, _ = env.step(action)
    if done:
        if reward == 0:
            winner = 'tie'
        else:
            if env.board.turn == FIRST_PLAYER:
                winner = 'white' if reward == 1 else 'black'
            else:
                winner = 'black' if reward == 1 else 'white'
        return {'move': '', 'legal_uci': [], 'winner': winner}

    env.render()
    player.movehead(action)

    return bot_move()


def bot_move():
    global node
    pi = get_move_probability(env, player, model)
    # value, probs = model(np.expand_dims(env.state, axis=0).astype(dtype=np.float32))
    # probs = np.array(probs[0])
    # value = int(value[0])
    # pi = np.zeros(env.action_space)
    # pi[player.head.next_actions] = probs[player.head.next_actions]
    action = np.argmax(pi)
    print(convert_action_move(action, env.board, env.board.turn))
    move = convert_action_move_exists(action, env.board, env.board.turn)
    print('\n'.join(list(map(lambda x: f"p {x[0].stats['p']}, q {-1 * x[0].stats['q']}, n {x[0].stats['n']}, move {convert_action_move_exists(x[1], env.board, env.board.turn).uci()}", zip(player.head.children, player.head.next_actions)))))
    print(pi[env.legal_actions()], str(pi[action]*100) + '%', move.uci())
    node = node.add_variation(move)
    state, reward, done, _ = env.step(action)
    env.render()
    player.movehead(action)
    if done:
        if reward == 0:
            winner = 'tie'
        else:
            if env.board.turn == FIRST_PLAYER:
                winner = 'white' if reward == 1 else 'black'
            else:
                winner = 'black' if reward == 1 else 'white'
        return {'move': move.uci(), 'legal_uci': [], 'winner': winner}
    else:
        legal = []
        for legal_move in env.board.legal_moves:
            legal.append(legal_move.uci())
        return {'move': move.uci(), 'legal_uci': legal, 'winner': ''}


if __name__ == "__main__":
    env = Game()
    m = multiprocessing.Manager()
    lock = m.Lock()
    state_init = env.reset()
    pgn_game = chess.pgn.Game()
    pgn_game.setup(env.board)
    model, checkpoint = load_model(env.action_space, MODEL)
    player = MonteCarloTreeSearch(state_init, FIRST_PLAYER, 'best_player')
    node = pgn_game
    app.run(debug=True)

    # get_move_probability_parallel
