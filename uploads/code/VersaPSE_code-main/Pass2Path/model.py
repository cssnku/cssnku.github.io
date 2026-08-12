"""
Seq2Seq model with stacked residual LSTMs.

Reference: https://github.com/matejklemen/stacked-residual-lstm/blob/master/seq2seq.py

The model translates a keyboard-sequence representation of a source password
into a sequence of edit operations (the "path") that transforms it into a
target password.
"""
from copy import deepcopy
from random import random

import numpy as np
import torch
from torch import nn, topk
import torch.nn.functional as F
from tqdm import tqdm

class ResidualLSTMEncoder(nn.Module):
    # Encoder: stacks multiple residual LSTMs over the embedded key sequence.
    def __init__(self, embedding_dim, vocab_size, num_layers, residual_layers,
                 inp_hid_size, device, dropout=0.0, bidirectional=False, residual_n=1):
        super().__init__()

        self.is_res = np.zeros(num_layers, dtype=bool)
        if residual_layers:
            for layer_id in residual_layers:
                self.is_res[layer_id] = True
        self.residual_n = residual_n

        # internally, each direction will produce inp_hid_size/2 features, which will get concatenated back together
        if bidirectional and inp_hid_size % 2 != 0:
            raise ValueError("Hidden state size must be even")
        self.bidirectional = bidirectional
        num_directions = 2 if self.bidirectional else 1

        self.input_size = inp_hid_size
        self.hidden_size = inp_hid_size // num_directions
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(num_embeddings=embedding_dim,
                                       embedding_dim=self.input_size)
        self.layers = nn.ModuleList([nn.LSTM(input_size=self.input_size,
                                             hidden_size=self.hidden_size,
                                             batch_first=True,
                                             bidirectional=self.bidirectional) for _ in range(num_layers)])
        self.device=device

    def forward(self, encoded_seq):
        """
        Arguments:
        ----------
        encoded_seq: torch.Tensor
            (batch_size, max_seq_len) tensor, containing integer-encoded
            sequences in current batch

        Returns:
        --------
        torch.Tensor, (list, list):
            [0] hidden states of all timesteps for the last layer (for attention)
                ((batch_size, max_seq_len, hidden_size) tensor)
            [1] list of last hidden states for each layer
                (num_layers * (1, batch_size, hidden_size) tensors)
            [2] list of last cell states for each layer
                (num_layers * (1, batch_size, hidden_size) tensors)
        """
        encoded_seq=encoded_seq.to(self.device)
        embedded = self.dropout(self.embeddings(encoded_seq))

        last_t_hids, last_t_cells = [], []
        all_layers_hids = [embedded]
        curr_inp = embedded
        for i, curr_lstm in enumerate(self.layers):
            curr_out, (curr_hid, curr_cell) = curr_lstm(curr_inp)

            all_layers_hids.append(curr_out)
            if self.is_res[i]:
                take_input_from = i - self.residual_n + 1
                if take_input_from >= 0:
                    identity = all_layers_hids[take_input_from]
                else:
                    # e.g. there is no input from 5 layers back when we are at point after layer 0
                    # (the only input that could be added in this case is the embeddings)
                    raise ValueError(f"Cannot add identity from {self.residual_n} layers back after layer {i}")

                curr_inp = curr_out + identity
            else:
                curr_inp = curr_out

            # after all but last layer
            if i < self.num_layers - 1:
                curr_inp = self.dropout(curr_inp)

            last_t_hids.append(curr_hid)
            last_t_cells.append(curr_cell)

        return all_layers_hids[-1], (last_t_hids, last_t_cells)


class ResidualLSTMDecoder(nn.Module):
    # Decoder: autoregressively generates the edit-operation path token by token.
    def __init__(self, vocab_size, num_layers, residual_layers,
                 inp_size, hid_size, device, dropout=0.0, num_attn_layers=1, residual_n=1):
        super().__init__()

        self.is_res = np.zeros(num_layers, dtype=bool)
        if residual_layers:
            for layer_id in residual_layers:
                self.is_res[layer_id] = True
        self.residual_n = residual_n

        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=inp_size)

        # LSTM input size: if attention is disabled, only the embedded token
        # is fed in; otherwise it is concatenated with the attended encoder states.
        self.layers = nn.ModuleList([nn.LSTM(input_size=(inp_size + hid_size) if num_attn_layers > 0 else inp_size,
                                             hidden_size=hid_size,
                                             batch_first=True) for _ in range(num_layers)])
        self.fc = nn.Linear(hid_size, vocab_size)
        self.device=device

    def forward(self,
                encoded_input,
                enc_hidden,
                dec_hiddens,
                dec_cells):
        # encoded_input: (batch_size, 1) tensor
        # enc_hidden: (batch_size, max_seq_len, hidden_size) tensor
        # dec_hiddens: list of num_layers * (1, batch_size, hidden_size) tensors
        # dec_cells: list of num_layers * (1, batch_size, hidden_size) tensors
        encoded_input=encoded_input.to(self.device)

        embedded_input = self.dropout(self.embeddings(encoded_input))
        curr_inp = embedded_input

        all_layers_hids = []
        hids, cells = [], []
        for i, curr_lstm in enumerate(self.layers):
            # Attention is disabled in this project; pass the raw input directly.
            decoder_inp = curr_inp

            all_layers_hids.append(decoder_inp)
            curr_out, (curr_hid, curr_cell) = curr_lstm(decoder_inp, (dec_hiddens[i], dec_cells[i]))
            if self.is_res[i]:
                take_input_from = i - self.residual_n + 1
                if take_input_from >= 0:
                    identity = all_layers_hids[take_input_from]
                else:
                    # e.g. there is no input from 5 layers back when we are at point after layer 0
                    # (the only input that could be added in this case is the input embeddings)
                    raise ValueError(f"Cannot add identity from {self.residual_n} layers back after layer {i}")
                curr_inp = curr_out + identity
            else:
                curr_inp = curr_out

            # after all but last layer
            if i < self.num_layers - 1:
                curr_inp = self.dropout(curr_inp)

            hids.append(curr_hid)
            cells.append(curr_cell)

        word_logits = self.fc(curr_inp)

        return word_logits, hids, cells

class Seq2Seq(nn.Module):
    # Full sequence-to-sequence model that couples the encoder and decoder above.
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.max_seq_len=61

    def forward(self, input, trg):
        last_lay_hids, (last_t_hids, last_t_cells) = self.encoder(input)
        curr_input=torch.tensor([[0] for _ in range(len(input))],
                                  dtype=torch.long, device=self.encoder.device)
        curr_hids, curr_cells = last_t_hids, last_t_cells
        teacher_forcing_proba = 0.5

        # Decoder pass
        this_batch_logits = []
        for dec_step in range(self.max_seq_len):
            logits, curr_hids, curr_cells = self.decoder(curr_input,
                                                           enc_hidden=last_lay_hids,
                                                           dec_hiddens=curr_hids,
                                                           dec_cells=curr_cells)
            this_batch_logits.append(logits[:, 0, :].unsqueeze(1))
            probas = F.softmax(logits, dim=2)
            curr_preds = torch.argmax(probas, dim=2)
            use_teacher_forcing = random() < teacher_forcing_proba
            if use_teacher_forcing:
                curr_input = trg[:, dec_step].unsqueeze(1)
            else:
                curr_input = curr_preds
        return torch.cat(this_batch_logits, dim=1)

    def prob_extract(self, trg, path):
        # Compute the joint probability of the model generating the given
        # target path, conditioned on the source sequence `trg`.
        last_lay_hids, (last_t_hids, last_t_cells) = self.encoder(trg)

        probs = torch.tensor([[1] for _ in range(len(trg))],
                                 dtype=torch.long, device=self.encoder.device)

        curr_hids, curr_cells = last_t_hids, last_t_cells

        for step in range(len(path[0])):
            curr_input = path[:, step].unsqueeze(1)
            logits, curr_hids, curr_cells = self.decoder(curr_input,
                                                         enc_hidden=last_lay_hids,
                                                         dec_hiddens=curr_hids,
                                                         dec_cells=curr_cells)

            output_prob = F.softmax(logits.squeeze(1))

            curr_prob=torch.gather(output_prob, 1, path[:, step].unsqueeze(1))

            probs=probs*curr_prob
        return probs.squeeze(1).tolist()

    def freeze_layers(self):
        # Freeze the decoder parameters (used in transfer learning).
        for param in self.decoder.parameters():
            param.requires_grad = False


