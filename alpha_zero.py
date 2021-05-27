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
BEST_MODEL = 'best_model.h5'
STORE_MODEL_BASE = 'models'


def run():
    self_play_parallel()

    self_play_parallel()
    train_network()


def play_bot(user_turn=FIRST_PLAYER):
    env = Game()
    best_model = load_model(env.action_space, BEST_MODEL)
    state = env.reset()
    best_player = MonteCarloTreeSearch(state, FIRST_PLAYER, 'best_player')

    state = env.reset()
    best_player.reset(state)
    done = False
    reward = 0
    half_move_clock = 0
    while not done:
        if best_player.head.turn == user_turn:
            # !!! get input
            action = input()
            if not best_player.head.children:
                best_player.populate_children(best_player.head, env)
        else:
            pi = get_move_probability(env, best_player, best_model)
            action = np.flatnonzero(np.random.multinomial(1, pi))[0]
        state, reward, done, _ = env.step(action)
        half_move_clock += 1
        best_player.movehead(action)
        if done:
            break
    player_turn = best_player.head.turn
    other_player = FIRST_PLAYER if player_turn == SECOND_PLAYER else SECOND_PLAYER
    if reward == 1:
        print(player_turn, 'wins!')
    elif reward == -1:
        print(other_player, 'wins!')
    else:
        print('draw!')


def save_model(model, file_name):
    path = os.path.join(os.getcwd(), STORE_MODEL_BASE, file_name)
    model.save(path)
    return


def load_model(num_actions, file_name):
    path = os.path.join(os.getcwd(), STORE_MODEL_BASE, file_name)
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
    else:
        model = AlphaZeroNetwork(num_actions)
    return model


def train_network():
    env = Game()
    model = load_model(env.action_space, MODEL)
    best_model = load_model(env.action_space, BEST_MODEL)
    model.compile(loss={'output_1': 'mean_squared_error', 'output_2': policy_loss},
                  optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM),
                  loss_weights={'output_1': 0.5, 'output_2': 0.5}
                  )
    count = 1
    while True:
        with np.load(STORE_DATA_PATH) as data:
            input_list = data['state_list']
            target_dict = {'output_1': data['move_probability_list'], 'output_2': data['reward_list']}
        dataset = tf.data.Dataset.from_tensor_slices((input_list, target_dict)).shuffle(len(input_list)).batch(2048)
        model.fit(dataset)
        if count % EVALUATE_EVERY_N_LOOP == 0:
            evaluate_network(model, best_model)
            save_model(best_model, BEST_MODEL)
        count += 1
        save_model(model, MODEL)


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

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss


def generate_game(iteration):
    env = Game()
    model = load_model(env.action_space, MODEL)
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
    while not done:
        pi = get_move_probability_parallel(env, player, model, lock)
        action = np.flatnonzero(np.random.multinomial(1, pi))[0]
        state_list.append(state)
        move_probability_list.append(pi)
        turn_list.append(1 if player.head.player_turn == FIRST_PLAYER else -1)
        count += 1

        state, reward, done, _ = env.step(action)
        player.movehead(action)

        if done:
            break
    current_player = player.head.player_turn
    direction = 1 if current_player == FIRST_PLAYER else -1
    # value for non_current_player is -1 * reward
    reward_list = reward * direction * np.array(turn_list)
    print(iteration)
    return state_list, move_probability_list, reward_list


def self_play_parallel():
    state_list = []
    move_probability_list = []
    reward_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for states, move_probabilities, rewards in executor.map(generate_game, list(range(SELF_PLAY_GAMES))):
            state_list.extend(states)
            move_probability_list.extend(move_probabilities)
            reward_list.extend(rewards)

    np.savez(STORE_DATA_PATH, state_list=state_list, move_probability_list=move_probability_list,
             reward_list=reward_list)


def self_play_test():

    for i in range(SELF_PLAY_GAMES):
        env = Game()
        model = load_model(env.action_space, MODEL)
        count = 0
        state = env.reset()
        player = MonteCarloTreeSearch(state, env)

        state_list = []
        move_probability_list = []
        turn_list = []
        done = False
        while not done:
            s = time()
            pi = get_move_probability(env, player, model)
            action = np.flatnonzero(np.random.multinomial(1, pi))[0]
            print(time() - s, action)
            state_list.append(state)
            move_probability_list.append(pi)
            turn_list.append(1 if player.head.player_turn == FIRST_PLAYER else -1)
            count += 1

            state, reward, done, _ = env.step(action)
            player.movehead(action)

            if done:
                break
        # value for non_current_player is -1 * reward


def self_play(model):
    env = Game()
    state = env.reset()
    done = False
    count = 0
    state_list = np.zeros((NUM_SAVE_POSITIONS, 8, 8, 20))
    move_probability_list = np.zeros((NUM_SAVE_POSITIONS, env.action_space))
    reward_list = np.zeros(NUM_SAVE_POSITIONS)
    player = MonteCarloTreeSearch(state)

    while True:
        start_count = count
        turn_list = []
        state = env.reset()
        player.reset(state)
        reward = 0

        while not done:
            pi = get_move_probability(env, player, model)
            action = np.flatnonzero(np.random.multinomial(1, pi))[0]
            state_list[count] = state
            move_probability_list[count] = pi

            if count < NUM_SAVE_POSITIONS - 1:

                turn_list.append(1 if player.head.player_turn == FIRST_PLAYER else -1)
                count += 1
            else:
                turn_list[-1] = 1 if player.head.player_turn == FIRST_PLAYER else -1

            state, reward, done, _ = env.step(action)
            env.render()
            player.movehead(action)

            if done:
                break
        current_player = player.head.player_turn
        direction = 1 if current_player == FIRST_PLAYER else -1
        # value for non_current_player is -1 * reward
        reward_list[start_count: count] = reward * direction * np.array(turn_list)
        if count == NUM_SAVE_POSITIONS:
            break
    np.savez(STORE_DATA_PATH, state_list=state_list, move_probability_list=move_probability_list,
             reward_list=reward_list)


def get_move_probability_parallel(env, player, model, lock):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = []
        paths = []
        leaves = []
        leaf_states = []
        rewards = []
        dones = []
        for i in range(NUM_SEARCHES):
            env_copy = copy.deepcopy(env)
            future.append(executor.submit(player.reach_leaf_node_parallel, env_copy, lock))
            # future.append(player.reach_leaf_node(env_copy, lock))
        for path, reward, done in list(map(lambda x: x.result(), future)):
            leaves.append(path[-1])
            leaf_states.append(path[-1].state)
            paths.append(path)
            rewards.append(reward)
            dones.append(done)
    values = player.evaluate_leaves(leaves, np.array(leaf_states), model).flatten()
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

    start_time = time()
    self_play_test()
    print(time() - start_time)
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
