import torch
import torch.nn as nn

# Internal project modules
import hyper_param as param


class SimpleP2E(nn.Module):
    """A GRU-based model that maps a password pair to a sequence of atomic edit operations.

    Given a source password and a target password (both represented as keypress
    sequences), the model predicts the edit operation applied at each position.
    """

    def __init__(self,
                 embedding_dim=param.embedding_dim,
                 vocab_size=param.num_embeddings,
                 hidden_size=param.hidden_size,
                 num_layers=param.num_layers,
                 dropout_rate=param.dropout_rate,
                 fc_dim=param.fc_dim,
                 type_num=param.type_num
                 ):
        super(SimpleP2E, self).__init__()
        self.embedding_dim = embedding_dim
        # Shared embedding layer for both source and target keypress sequences
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,)
        # GRU input size is the concatenation of the two embedding vectors
        self.gru = nn.GRU(input_size=2 * embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=fc_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=fc_dim, out_features=type_num)
        self.softmax = nn.Softmax()

    def forward(self, x1, x2, L):
        # Inputs have shape (batch_size, sequence_length); transpose to (seq_len, batch)
        x1 = x1.t()
        x2 = x2.t()
        x1 = self.embedding(x1.long())
        x2 = self.embedding(x2.long())

        # Concatenate the two embeddings along the feature dimension
        x = torch.cat((x1, x2), dim=-1)  # (seq_len, batch, 2*embedding_dim)

        # Encode with the GRU
        x, _ = self.gru(x)  # (seq_len, batch, hidden)

        # Take the hidden state at the last valid time step of each sample
        x = x.transpose(0, 1)  # (batch, seq_len, hidden)
        # L stores the true length of each sample in the batch
        idx = (L - 1).view(-1, 1, 1).expand(-1, 1, x.size(2)).to(x.device)  # (batch, 1, hidden)
        x = x.gather(1, idx).squeeze(1)  # (batch, hidden)

        # Classification head. No softmax here: CrossEntropyLoss applies it internally.
        x = self.fc1(x)
        x = self.dropout(torch.relu(x))
        x = self.fc2(x)
        return x

    def freeze_layers(self):
        """Freeze the classifier (fc1/fc2) and the first GRU layer.

        Used when the model is fine-tuned on a new dataset so that only the
        remaining GRU layers are updated (partial transfer learning).
        """
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

        layer_idx = 0
        for name, param in self.gru.named_parameters():
            if name.endswith(f'_l{layer_idx}'):
                param.requires_grad = False

