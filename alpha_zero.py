import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from time import time
import os
import concurrent.futures
import multiprocessing

from game import Game, FIRST_PLAYER, SECOND_PLAYER
from model import AlphaZeroNetwork
from treesearch import MonteCarloTreeSearch
from hyper_params import *

STORE_DATA_PATH = './datas/training.npz'
MODEL = 'model.h5'
OPTIMIZER = 'model_optimizer.pkl'
BEST_MODEL = 'best_model.h5'
BEST_OPTIMIZER = 'best_optimizer.pkl'
STORE_MODEL_BASE = 'models'


def run():
    self_play_parallel()

    self_play_parallel()
    train_network()


def play_bot(user_turn=FIRST_PLAYER):
    env = Game()
    best_model, _ = load_model(env.action_space, MODEL)
    state = env.reset()
    best_player = MonteCarloTreeSearch(state, FIRST_PLAYER, 'best_player')

    state = env.reset()
    env.render()
    best_player.reset(state)
    done = False
    reward = 0
    half_move_clock = 0
    while not done:
        if best_player.head.player_turn == user_turn:
            # !!! get input
            action = -1
            while action not in env.legal_actions():
                print(env.legal_actions())
                action = int(input())
                if not best_player.head.children:
                    best_player.populate_children(best_player.head, env)
        else:
            pi = get_move_probability(env, best_player, best_model)
            print(pi)
            action = np.flatnonzero(np.random.multinomial(1, pi))[0]
        state, reward, done, _ = env.step(action)
        env.render()
        half_move_clock += 1
        best_player.movehead(action)
        if done:
            break
    player_turn = best_player.head.player_turn
    other_player = FIRST_PLAYER if player_turn == SECOND_PLAYER else SECOND_PLAYER
    if reward == 1:
        print(player_turn, 'wins!')
    elif reward == -1:
        print(other_player, 'wins!')
    else:
        print('draw!')


def save_model(checkpoint):
    checkpoint.save()
    return


def load_model(num_actions, file_name):
    model = AlphaZeroNetwork(num_actions)
    model.compile(loss={'output_1': 'mean_squared_error', 'output_2': policy_loss},
                  optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM),
                  loss_weights={'output_1': 0.5, 'output_2': 0.5}
                  )
    path = os.path.join(os.getcwd(), STORE_MODEL_BASE)

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint is not None:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return model, manager


def test_network():
    env = Game()
    model, checkpoint = load_model(env.action_space, MODEL)

    count = 1
    for num, fn in enumerate(os.listdir(STORE_DATA_BASE)):
        path = os.path.join(STORE_DATA_BASE, fn)
        with np.load(path, allow_pickle=True) as data:
            input_list = data['state_list'][:64]
            output1 = data['reward_list'][:64]
            output2 = data['move_probability_list'][:64]

        dataset = tf.data.Dataset.from_tensor_slices((input_list, (output1, output2))).shuffle(64).batch(1)
        model.evaluate(dataset)
        # if count % EVALUATE_EVERY_N_LOOP == 0:
        #     evaluate_network(model, best_model)
        #     save_model(best_model, BEST_MODEL)
        # count += 1

        break


def traintest_network():
    env = Game()
    model = AlphaZeroNetwork(env.action_space)
    model.compile(loss={'output_1': 'mean_squared_error', 'output_2': policy_loss},
                  optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM),
                  loss_weights={'output_1': 0.5, 'output_2': 0.5}
                  )
    for num, fn in enumerate(os.listdir(STORE_DATA_BASE)):
        path = os.path.join(STORE_DATA_BASE, fn)
        with np.load(path, allow_pickle=True) as data:
            input_list = data['state_list'][:32]
            output1 = data['reward_list'][:32]
            output2 = data['move_probability_list'][:32]

        dataset = tf.data.Dataset.from_tensor_slices((input_list, (output1, output2))).shuffle(640).batch(640)
        model.fit(input_list, (output1,output2), batch_size=32, epochs=10)
        print(model.layers[0].layers[1].moving_mean)
        output_nt = model(input_list, training=False)
        print(model.layers[0].layers[1].moving_mean)
        output_t = model(input_list, training=True)
        print(model.layers[0].layers[1].moving_mean)
        for i in range(len(input_list)):
            print(output1[i], output_nt[0][i], output_t[0][i])
        # model.fit(input_list, (output1, output2), epochs=3, batch_size=1)
        results = model.evaluate(input_list, (output1, output2))
        print(results)
        # save_model(manager)
        break

def testtest():
    import chess
    import chess.pgn
    from game import convert_move_action, convert_action_move
    env = Game()
    model, manager = load_model(env.action_space, MODEL)
    results_value = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    for pgn_n, fn in enumerate(os.listdir(CHESS_GAMES_PATH)):
        pgn = open(os.path.join(CHESS_GAMES_PATH, fn))
        while 1:
            game = chess.pgn.read_game(pgn)
            if game is None:
                print('game')
                break
            result = game.headers['Result']
            if result not in results_value:
                print('result')
                continue
            value = results_value[result]
            loser = None if value == 0 else (chess.WHITE if value == -1 else chess.BLACK)
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                policy = np.zeros(4672)
                state = Game.get_state(board)
                # states[move_count] = state
                print('---')
                print(value * (1 if board.turn == chess.WHITE else -1), model(np.expand_dims(state, axis=0).astype(dtype=np.float32))[0])
                print(board)
                mirrored = move
                if board.turn == chess.BLACK:
                    mirrored = chess.Move(from_square=chess.square_mirror(move.from_square),
                                          to_square=chess.square_mirror(move.to_square), promotion=move.promotion)
                try:
                    print(convert_action_move(convert_move_action(mirrored), board, board.turn))
                    if board.turn != loser:
                        policy[convert_move_action(mirrored)] = 1
                except KeyError as err:
                    print(board, move, board.turn)
                    raise KeyError(err)

                board.push(move)
            return

def train_network():
    env = Game()
    model, checkpoint = load_model(env.action_space, MODEL)

    count = 1
    for num, fn in enumerate(os.listdir(STORE_DATA_BASE)):
        path = os.path.join(STORE_DATA_BASE, fn)
        with np.load(path, allow_pickle=True) as data:
            input_list = data['state_list']
            output1 = data['reward_list']
            output2 = data['move_probability_list']

        dataset = tf.data.Dataset.from_tensor_slices((input_list, (output1, output2))).batch(1024).repeat(2)
        model.fit(dataset, steps_per_epoch=244, epochs=2)
        # if count % EVALUATE_EVERY_N_LOOP == 0:
        #     evaluate_network(model, best_model)
        #     save_model(best_model, BEST_MODEL)
        # count += 1

        save_model(checkpoint)


def evaluate_network(model, best_model):
    env = Game()
    state = env.reset()

    m = multiprocessing.Manager()
    lock = m.Lock()

    eval_player = MonteCarloTreeSearch(state, FIRST_PLAYER, 'eval_player')
    best_player = MonteCarloTreeSearch(state, FIRST_PLAYER, 'best_player')
    score = {eval_player.name: 0, best_player.name: 0, 'ties': 0}
    for game_count in range(EVALUATE_GAMES):
        if game_count % 2:
            player1 = {'tree': eval_player, 'model': model}
            player2 = {'tree': best_player, 'model': best_model}
        else:
            player1 = {'tree': best_player, 'model': best_model}
            player2 = {'tree': eval_player, 'model': model}
        players = {FIRST_PLAYER: player1, SECOND_PLAYER: player2}
        state = env.reset()
        player1['tree'].reset(state)
        player2['tree'].reset(state)
        done = False
        reward = 0
        half_move_clock = 0
        while not done:
            if half_move_clock % 2 == 0:
                pi = get_move_probability_parallel(env, player1['tree'], player1['model'], lock)
                if not player2['tree'].head.children:
                    player2['tree'].populate_children(player2['tree'].head, env)
            else:
                pi = get_move_probability_parallel(env, player2['tree'], player2['model'], lock)
                if not player1['tree'].head.children:
                    player1['tree'].populate_children(player1['tree'].head, env)
            action = np.flatnonzero(np.random.multinomial(1, pi))[0]
            state, reward, done, _ = env.step(action)
            half_move_clock += 1
            player1['tree'].movehead(action)
            player2['tree'].movehead(action)
            if done:
                break
        player_turn = player1['tree'].head.player_turn
        other_player = FIRST_PLAYER if player1['tree'].head.player_turn == SECOND_PLAYER else SECOND_PLAYER
        if reward == 1:
            score[players[player_turn]['tree'].name] += 1
        elif reward == -1:
            score[players[other_player]['tree'].name] += 1
        else:
            score['ties'] += 1
    print(score)
    if score[best_player.name] > 0:
        if float(score[eval_player.name])/float(score[best_player.name]) > 1.22:
            best_model.set_weights(model.get_weights())
            return True
    elif score[eval_player.name] > score[best_player.name]:
        best_model.set_weights(model.get_weights())
        return True
    return False


def policy_loss(y_true, y_pred):

    p = y_pred
    pi = y_true

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)
    return loss


def generate_game(iteration):
    env = Game()
    model, _ = load_model(env.action_space, MODEL)
    count = 0
    state = env.reset()
    player = MonteCarloTreeSearch(state, env)

    m = multiprocessing.Manager()
    lock = m.Lock()

    state_list = []
    move_probability_list = []
    turn_list = []
    done = False
    reward = 0
    start = time()
    while not done:
        s = time()
        pi = get_move_probability_parallel(env, player, model, lock)
        action = np.flatnonzero(np.random.multinomial(1, pi))[0]
        state_list.append(state)
        move_probability_list.append(pi)
        turn_list.append(1 if player.head.player_turn == FIRST_PLAYER else -1)
        count += 1

        state, reward, done, _ = env.step(action)
        player.movehead(action)
        print(time()-s)
        if done:
            break
    current_player = player.head.player_turn
    direction = 1 if current_player == FIRST_PLAYER else -1
    # value for non_current_player is -1 * reward
    reward_list = reward * direction * np.array(turn_list)
    print(iteration, count, time()-start)
    return state_list, move_probability_list, reward_list


def self_play_parallel(num_games=SELF_PLAY_GAMES):
    state_list = []
    move_probability_list = []
    reward_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for states, move_probabilities, rewards in executor.map(generate_game, list(range(num_games))):
            state_list.extend(states)
            move_probability_list.extend(move_probabilities)
            reward_list.extend(rewards)
    print(len(rewards))

    # np.savez(STORE_DATA_PATH, state_list=state_list, move_probability_list=move_probability_list,
    #          reward_list=reward_list)

    # for states, move_probabilities, rewards in map(generate_game, list(range(SELF_PLAY_GAMES))):
    #     state_list.extend(states)
    #     move_probability_list.extend(move_probabilities)
    #     reward_list.extend(rewards)


def self_play(num_games=SELF_PLAY_GAMES):
    state_list = []
    move_probability_list = []
    reward_list = []

    for states, move_probabilities, rewards in map(generate_game, list(range(num_games))):
        state_list.extend(states)
        move_probability_list.extend(move_probabilities)
        reward_list.extend(rewards)
    print(len(rewards))


def get_move_probability_parallel(env, player, model, lock):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        s = time()
        future = []
        paths = []
        leaves = []
        leaf_states = np.zeros((NUM_SEARCHES, *env.observation_space))
        rewards = np.zeros(NUM_SEARCHES)
        dones = np.zeros(NUM_SEARCHES)
        for i in range(NUM_SEARCHES):
            env_copy = copy.copy(env)
            future.append(executor.submit(player.reach_leaf_node_parallel, env_copy, lock))
            # future.append(player.reach_leaf_node(env_copy))
        future = list(map(lambda x: x.result(), future))
        for i, (path, reward, done) in enumerate(list(map(lambda x: x, future))):
            leaves.append(path[-1])
            leaf_states[i] = path[-1].state
            paths.append(path)
            rewards[i] = reward
            dones[i] = done
    values = player.evaluate_leaves(leaves, leaf_states, model)
    values = np.where(dones, rewards, values)
    for path, value in zip(paths, values):
        player.backfill(path, value)
    pi = np.zeros(env.action_space)
    for action, child in zip(player.head.next_actions, player.head.children):
        pi[action] = np.power(child.stats['n'], 1 / TAO)
    return pi / np.sum(pi)


def get_move_probability(env, player, model):
    for i in range(NUM_SEARCHES):
        env_copy = copy.deepcopy(env)
        path, reward, done = player.reach_leaf_node(env_copy)
        if done:
            player.backfill(path, reward)
        else:
            value = player.evaluate_node(path[-1], model)
            player.backfill(path, value)
    pi = np.zeros(env.action_space)
    for action, child in zip(player.head.next_actions, player.head.children):
        pi[action] = np.power(child.stats['n'], 1/TAO)
    return pi/np.sum(pi)


if __name__ == '__main__':

    # start_time = time()
    # self_play(1)
    # print(time() - start_time)

    # train_network()
    # traintest_network()
    traintest_network()
    # testtest()
    # test_network()

    # play_bot()

    # with np.load(STORE_DATA_PATH, allow_pickle=True) as data:
    #     input_list = data['state_list']
    #     move_list = data['move_probability_list']
    #     rewards = data['reward_list']
    # print(input_list.shape, move_list.shape, rewards.shape)

    # train_network(model)
    # evaluate_network(model, best_model)
    # save_model(model)

# !!!efficiency
# !!!play wit bot
# !!!run
