import chess
import chess.pgn
import numpy as np
import sys
import numpy
import time
numpy.set_printoptions(threshold=sys.maxsize)

num_letter = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'
}
letter_num = {
    'a': 0, 'b': 8, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8
}
map_action_uci = {}
map_uci_action = {}
FIRST_PLAYER = chess.WHITE
SECOND_PLAYER = chess.BLACK
# note: to remove duplicated states in MCTS, we flip the board when player turn is black. However, the chess board
# in the simulation is not flipped (for displaying purposes).
# Take this into account when converting actions and states from/to MCTS and Environment


def convert_action_move(action, piece_type_orig, player_turn):
    """
    converts action index to chess.Move object
    :param action:
    :param piece_type_orig:
    :param player_turn:
    :return:
    """
    # 0 <= from_square < 64      0 <= move info < 73
    # index = from_square + 8 * 8 move info
    move_info = action // 64
    from_square = action % 64
    to_square = action % 64
    promotion = None

    if move_info < 56:
        # queen move
        # move_info = direction_base + distance * 8
        distance = move_info // 8
        direction_base = move_info % 8
        # queen_direction maps direction_base to amount of squares to add from from_sqaure to reach the direction
        queen_direction = {
            0: 7, 1: 8, 2: 9, 3: 1, 4: -7, 5: -8, 6: -9, 7: -1
        }
        to_square += queen_direction[direction_base] * distance
        # pawn promote to queen
        row_target = to_square // 8
        if piece_type_orig == chess.PAWN and (row_target == 7 or row_target == 1):
            promotion = chess.QUEEN
    elif move_info < 64:
        knight_move_base = move_info - 56
        # knight move
        knight_move = {
            0: 15, 1: 17, 2: 10, 3: -6, 4: -17, 5: -19, 6: -10, 7: 6
        }
        to_square += knight_move[knight_move_base]
    else:
        # pawn under promotion
        pawn_move_base = (move_info - 64) // 3
        promote_int = (move_info - 64) % 3
        pawn_move = {
            0: 7, 1: 8, 2: 9
        }
        under_promotion = {
            0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK
        }
        to_square += pawn_move[pawn_move_base]
        promotion = under_promotion[promote_int]
    if player_turn == chess.BLACK:
        from_square, to_square = chess.square_mirror(from_square), chess.square_mirror(to_square)
    return chess.Move(from_square=from_square, to_square=to_square, promotion=promotion)


def convert_action_move_exists(action, board, player_turn):
    """
    Converts action index to chess.Move object.
    Assume the action key exists in map_action_uci
    :param action:
    :param board:
    :param player_turn:
    :return:
    """

    move = chess.Move.from_uci(map_action_uci[action])
    if player_turn == chess.BLACK:
        move = chess.Move(from_square=chess.square_mirror(move.from_square),
                          to_square=chess.square_mirror(move.to_square), promotion=move.promotion)
    if move.promotion == chess.QUEEN:
        move.promotion = None
    rank = move.to_square//8
    try:
        if move.promotion is None and board.piece_at(move.from_square).piece_type == chess.PAWN and \
                (rank == 7 or rank == 0):
            move.promotion = chess.QUEEN
    except AttributeError as err:
        print(board, move, action, player_turn)
        raise AttributeError(err)
    return move


def convert_move_action(move):
    """
    Converts chess.Move object to action index.
    Also populates the map_uci_action and map_action_uci with the corresponding action and uci key
    if the key did not exist before.
    Assumes the move is mirrored if the turn is black

    0 <= from_square < 64      0 <= move info < 73
    0 <= queen_move < 56
    56 <= knight_move < 64
    64 <= pawn_move < 73
    index = from_square + 8 * 8 move info
    :param move:
    :return:
    """
    # Note: we flip the board prior when we list the available moves to reduce computation so
    # the board does not need to be flipped here
    move_difference = move.to_square - move.from_square

    queen_direction = {
        7: 0, 8: 1, 9: 2, 1: 3, -7: 4, -8: 5, -9: 6, -1: 7
    }
    knight_move = {
        15: 0, 17: 1, 10: 2, -6: 3, -17: 4, -15: 5, -10: 6, 6: 7
    }
    pawn_move = {
        7: 0, 8: 1, 9: 2
    }
    under_promotion = {
        chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2
    }
    if move.uci() in map_uci_action:
        return map_uci_action[move.uci()]
    if move_difference in knight_move:
        move_info = 56 + knight_move[move_difference]
    else:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            move_info = 64 + 3 * pawn_move[move_difference] + under_promotion[move.promotion]
        else:
            dx = move.to_square % 8 - move.from_square % 8
            dy = move.to_square // 8 - move.from_square // 8
            distance = max(abs(dx), abs(dy))
            direction = queen_direction[move_difference // distance]
            move_info = direction + 8 * (distance - 1)
    action = move.from_square + 64 * move_info
    map_uci_action[move.uci()] = action
    map_action_uci[action] = move.uci()
    return action


class Game:
    def __init__(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self.black = None
        self.white = None
        self.observation_space = (8, 8, 20)
        self.state = self._update_state()
        self.action_space = 4672
        self.done = False

    def render(self):
        print(self.board)
        print()

    def reset(self):
        self.board.reset()
        self.state = self._update_state()
        self.done = False
        return self.state

    def _update_state(self):
        new_state = np.zeros((8, 8, 20), dtype=np.uint8)
        new_state[:, :, 0].fill(self.board.has_queenside_castling_rights(chess.WHITE))
        new_state[:, :, 1].fill(self.board.has_kingside_castling_rights(chess.WHITE))
        new_state[:, :, 2].fill(self.board.has_queenside_castling_rights(chess.BLACK))
        new_state[:, :, 3].fill(self.board.has_kingside_castling_rights(chess.BLACK))
        new_state[:, :, 4].fill(self.board.turn)
        new_state[7 - (self.board.halfmove_clock // 16), (self.board.halfmove_clock // 2) % 8, 5] = True
        new_state[:, :, 6].fill(True)
        new_state[:, :, 7:19] = self._get_board_array()
        new_state[:, :, 19].fill(self.board.is_repetition(2))
        return new_state

    def _get_board_array(self):
        # boards, identical when flipped, are valued equally according to MCTS
        board = self.board if self.board.turn == chess.WHITE else self.board.mirror()
        board_list = [chess.SquareSet(chess.BB_EMPTY) for _ in range(12)]
        pieces = [chess.PAWN, chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN, chess.KING]
        players = [chess.WHITE, chess.BLACK]
        i = 0
        for player in players:
            for piece in pieces:
                board_list[i] = board.pieces(piece, player)
                i += 1
        square = board.ep_square
        if square:
            ours = square < 32
            col = square % 8
            dest_square = 8 * 7 + col if not ours else col
            if ours:
                board_list[0].remove(square + 8)
                board_list[0].add(dest_square)
            else:
                board_list[6].remove(square - 8)
                board_list[6].add(dest_square)
        return np.unpackbits(np.array(board_list, dtype=np.uint64).view(dtype=np.uint8)).\
            reshape([8, 8, len(board_list)])

    def step(self, action, update_state=True):
        if self.done:
            return self.state, 0, self.done, None
        move = convert_action_move_exists(action, self.board, self.board.turn)

        self.board.push(move)
        new_state = self._update_state()
        reward = 0
        outcome = self.board.outcome(claim_draw=True)
        # change reward so the reward is in perspective to the last player
        done = False
        if outcome is not None:
            result = outcome.result().split('-')
            if result[0] == '1/2':
                reward = 0
            else:
                if self.board.turn == chess.WHITE:
                    reward = 1 if result[0] == '1' else -1
                else:
                    reward = 1 if result[1] == '1' else -1
            done = True
        if update_state:
            self.state = new_state
            self.done = done
        else:
            self.board.pop()
        return new_state, reward, done, move

    def legal_actions(self, board=None):
        # flip board if player turn is black to avoid duplicate states
        board = self.board if board is None else board
        board = board if board.turn == chess.WHITE else board.mirror()

        actions = []
        for move in board.legal_moves:
            actions.append(convert_move_action(move))
        return actions


def play_game(game_count, print_value):
    game = Game('1n6/P1P2k1P/2n5/8/2N5/4N3/pp1K2p1/5N2 w - - 0 1')
    state = game.reset()
    pgn_game = chess.pgn.Game()
    pgn_game.setup(game.board)
    node = pgn_game
    d = False
    r = 0
    step = 0
    s = time.time()
    while not d:
        if step == 1 and print_value:
            with open('./games/pgn.txt', 'w') as file:
                print(state.transpose([2, 0, 1]), file=file)
        step += 1
        # try:
        #     move = chess.Move.from_uci(input())
        # except ValueError:
        #     continue
        # if move in games.board.legal_moves:
        #     game.step(move)
        # print(game.board.legal_moves)
        try:
            legal_actions = game.legal_actions()
            assert len(list(game.board.legal_moves)) == len(legal_actions)
            action = np.random.choice(legal_actions)
            # if step == 1:
            #     # action = convert_move_action(chess.Move.from_uci('g2g1'))
            #     action = convert_move_action(chess.Move.from_uci('e2e3'))

            state, r, d, move = game.step(action)
            if print_value:
                print(move, action)
            node = node.add_variation(move)
        except ValueError as err:
            print('value error', err)
            continue
        if print_value:
            print(game.board, r, step)
    print(game_count)
    if print_value:
        print(time.time() - s)
        print(game.board.outcome(claim_draw=True).termination, r)
    if print_value:
        print('printing...')
        with open('./games/pgn.pgn', 'w') as file:
            print(pgn_game, file=file)
        with open('./games/pgn.txt', 'w') as f:
            print(state.transpose([2, 0, 1]), file=f)


if __name__ == '__main__':
    import concurrent.futures
    start = time.time()
    print_values = np.zeros(10)
    game_counts = np.arange(10)
    # print_values[-1] = 1
    with concurrent.futures.ProcessPoolExecutor() as execute:
        results = [list(execute.map(play_game, game_counts, print_values))]
    # play_game(0, 1)
    print(time.time() - start)
