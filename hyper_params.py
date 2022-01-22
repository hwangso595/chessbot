# Model
TAO = 1
LEARNING_RATE = 0.01
BATCH_MOMENTUM = 0.00000000
L2 = 0.01
MOMENTUM = 0.9

RESIDUAL_LAYERS = 20
NUM_FILTERS = 128

# Alpha Zero
MAX_MEM = 250000
NUM_SAVE_POSITIONS = 25000000
EVALUATE_GAMES = 1000
NUM_SEARCHES = 400
EVALUATE_EVERY_N_LOOP = 1000
SELF_PLAY_GAMES = 50

#DEBUG
# NUM_SAVE_POSITIONS = 500
# EVALUATE_GAMES = 4
# NUM_SEARCHES = 300
# EVALUATE_EVERY_N_LOOP = 10
# SELF_PLAY_GAMES = 1000

# MCTS
CPUCT = 2
NOISE_X = 0.75
DIR_ALPHA = 1

# SETTINGS
CHESS_GAMES_PATH = './games' # contains pgn files for the network to train on
STORE_DATA_BASE = './datas' # path to store the numpy datasets
MODEL = 'model.h5' # can be ignored since we are using checkpoints
BEST_MODEL = 'best_model.h5' # can be ignored since we are using checkpoints
STORE_MODEL_BASE = 'models'
