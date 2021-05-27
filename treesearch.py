import numpy as np
from collections import deque
import tensorflow as tf
from time import time

from game import Game, FIRST_PLAYER, SECOND_PLAYER
from model import AlphaZeroNetwork
from hyper_params import *


class Node:
    def __init__(self, state, player_turn):
        self.state = state
        # index of the children nodes matches the index of action it took to reach the corresponding state
        self.next_actions = []
        # we can keep the node object in since this MonteCarloTreeSearch.tree will contain reference to the node
        # so the memory is not duplicated
        self.children = []
        self.stats = {'n': 0, 'w': 0, 'q': 0, 'p': 0}
        self.player_turn = player_turn
        self.id = str(np.packbits(state.reshape(-1)).view(dtype=np.uint64))


class Edge:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0
        self.action = None


class MonteCarloTreeSearch:
    def __init__(self, state, player_turn=FIRST_PLAYER, name='eval_player'):
        # the environment and model are better passed in as arguments for needed methods since they
        # not be picklable
        self.name = name
        self.cpuct = CPUCT
        self.noise_x = NOISE_X
        self.dir_alpha = DIR_ALPHA
        self.head = Node(state, player_turn)

    def populate_children(self, node, env):
        nodes = []
        actions = []
        next_turn = SECOND_PLAYER if node.player_turn == FIRST_PLAYER else FIRST_PLAYER
        for action in env.legal_actions():
            state, reward, done, _ = env.step(action, update_state=False)
            nodes.append(Node(state, next_turn))
            actions.append(action)
        node.children = nodes
        node.next_actions = actions

    def reach_leaf_node(self, env):
        # this method takes in the deep copy of the env
        node = self.head
        path = deque([])
        reward = 0
        done = False
        while node.children:
            q = np.array(list(map(lambda next_node: next_node.stats['q'], node.children)), dtype=np.float32)
            p = np.array(list(map(lambda next_node: next_node.stats['p'], node.children)), dtype=np.float32)
            dir_a = np.random.dirichlet([self.dir_alpha]*len(node.children))
            child_ns = np.array(list(map(lambda next_node: next_node.stats['n'], node.children)), dtype=np.float32)
            u = self.cpuct * (self.noise_x * p + (1 - self.noise_x) * dir_a) * np.sqrt(np.sum(child_ns))/(1 + child_ns)
            puct = q + u
            state, reward, done, _ = env.step(node.next_actions[np.argmax(puct)])
            path.append(node)
            node = node.children[np.argmax(puct)]
        path.append(node)
        self.populate_children(node, env)
        return path, reward, done

    def reach_leaf_node_parallel(self, env, lock):
        # this method takes in the deep copy of the env
        node = self.head
        path = deque([])
        reward = 0
        done = False
        with lock:
            if not node.children:
                print('its doing it')
                path.append(node)
                self.populate_children(node, env)
                print('its done')
                return path, reward, done
        while node.children:
            q = np.array(list(map(lambda next_node: next_node.stats['q'], node.children)), dtype=np.float32)
            p = np.array(list(map(lambda next_node: next_node.stats['p'], node.children)), dtype=np.float32)
            dir_a = np.random.dirichlet([self.dir_alpha]*len(node.children))
            child_ns = np.array(list(map(lambda next_node: next_node.stats['n'], node.children)), dtype=np.float32)
            u = self.cpuct * (self.noise_x * p + (1 - self.noise_x) * dir_a) * np.sqrt(np.sum(child_ns))/(1 + child_ns)
            puct = q + u
            if np.sum(puct) == 0:
                action = np.random.randint(0, len(node.children))
            else:
                action = np.argmax(puct)
            state, reward, done, _ = env.step(node.next_actions[action])
            path.append(node)
            node = node.children[action]
            with lock:
                if not node.children:
                    path.append(node)
                    self.populate_children(node, env)
                    break
        return path, reward, done

    def evaluate_node(self, node, model):
        values, policy = model(np.expand_dims(node.state, axis=0).view(dtype=np.float32))
        legal_input = np.array(policy[0])[node.next_actions]
        e_x = np.exp(legal_input)
        softmax = e_x/np.sum(e_x)
        for i in range(len(node.children)):
            node.children[i].stats['p'] = softmax[i]
        return int(values[0, 0])

    def evaluate_leaves(self, nodes, states, model):
        values, policy = model(states.view(dtype=np.float32))
        for i, node in enumerate(nodes):
            legal_input = np.array(policy)[i][node.next_actions]
            e_x = np.exp(legal_input)
            softmax = e_x/np.sum(e_x)
            for i in range(len(node.children)):
                node.children[i].stats['p'] = softmax[i]
        return np.array(values)

    def backfill(self, path, value):
        current_player = path[-1].player_turn

        for node in path:
            if node.player_turn == current_player:
                direction = 1
            else:
                direction = -1
            node.stats['n'] += 1
            node.stats['w'] += value * direction
            node.stats['q'] = node.stats['w']/node.stats['n']

    def reset(self, state, player_turn=FIRST_PLAYER):
        self.head = Node(state, player_turn)

    def movehead(self, action):
        self.head = self.head.children[np.argwhere(np.array(self.head.next_actions) == action)[0, 0]]
