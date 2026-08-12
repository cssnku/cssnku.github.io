import torch

# Device used for training and inference
device = torch.device("cuda")

# Dataset configuration (used by match_finder.py)
base_dict = 'Tianya'
train_dict = 'Tianya'
test_set = 'CSDN'
strategy = 'original_dataset'  # 'original_dataset' or 'general_dataset'
threshold = 100

"""
RNN hyper-parameters.
These values are tuned for the current datasets; do not change them casually.
"""
# embedding_dim: dimension of the keypress embedding
embedding_dim = 256

# num_embeddings: number of keypress token types (~53 after char-level tokenization).
# Keep it as is: one slot is reserved for unseen characters and one extra for safety.
num_embeddings = 54

# hidden_size: hidden size of the GRU
hidden_size = 256

# fc_dim: number of neurons in the fully-connected layer
fc_dim = 512

# type_num: number of edit-operation classes. Must be slightly larger than the total
# number of edit operations extracted by the tokenizer (at least total + 1).
type_num = 1430

# dropout_rate: dropout probability applied in the classification head
# (dropout is applied twice: after fc1 and before fc2)
dropout_rate = 0.4

# num_layers: number of GRU layers
num_layers = 3

"""
Tokenizer-related parameters.
Same warning: do not modify without understanding the comments above.
"""

# pwd_oov_token: token id for out-of-vocabulary characters; should be at least
# (number of observed character types + 1)
pwd_oov_token = 54

# ops_oov_token: token id for out-of-vocabulary edit operations; should be at least
# (number of observed edit operations + 1)
ops_oov_token = 1422

# Maximum password length (in keypresses) used when padding sequences
MAX_LEN = 30