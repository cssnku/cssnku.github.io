import random
from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import queue
import os
import config


class Encoder(nn.Module):
    def __init__(self, vob_size, embed_dim, hidden_dim, layer_num=1, pad_idx=0, dropout=0.5):
        super(Encoder, self).__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(num_embeddings=vob_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=self.pad_idx)

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=layer_num,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(p=dropout)

    # x.shape: (batch, seq_len) character indices.
    # mask.shape: (batch, seq_len) real-length indicator per sample.
    def forward(self, x, mask):
        embedded = self.embedding(x)

        embedded = self.dropout(embedded)

        seq_lens = mask.sum(dim=-1)

        packed = pack_padded_sequence(input=embedded, lengths=seq_lens.to('cpu'), batch_first=True,
                                      enforce_sorted=False)
        output_packed, (h, c) = self.lstm(packed)

        output, _ = pad_packed_sequence(sequence=output_packed,
                                        batch_first=True,
                                        padding_value=self.pad_idx,
                                        total_length=seq_lens.max())
        # output.shape = [batch_size, max_src_len, num_directions(=2) * num_hiddens]
        return output, (h, c)


class Reduce(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super(Reduce, self).__init__()

        self.hidden_dim = hidden_dim

        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h, c):
        assert 2 == h.shape[0], "hidden shape's first dimension should be 2, but got %d" % h.shape[0]
        assert 2 == c.shape[0], "cell shape's first dimension should be 2, but got %d" % c.shape[0]

        assert self.hidden_dim == h.shape[2]
        assert self.hidden_dim == c.shape[2]

        # Merge the forward / backward hidden states of the bidirectional
        # encoder into a single initial decoder state, preserving batch order.
        h = h.permute(1, 0, 2).reshape(len(h[0, :, :]), -1)  # [batch, hidden*2]
        c = c.permute(1, 0, 2).reshape(len(c[0, :, :]), -1)

        h_output = self.dropout(self.reduce_h(h))
        c_output = self.dropout(self.reduce_c(c))

        h_output = F.relu(h_output)
        c_output = F.relu(c_output)

        # h_output.shape == c_output.shape = [1, batch_size, num_hiddens]
        return h_output.unsqueeze(0), c_output.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, hidden_dim, use_coverage=False):
        super(Attention, self).__init__()

        self.use_coverage = use_coverage

        self.w_h = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)  #
        self.w_s = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)

        if self.use_coverage:
            self.w_c = nn.Linear(1, hidden_dim * 2)

        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    # h: encoder hidden states h_i at each step t, shape (batch, seq_len, hidden*2).
    # mask: 0-1 encoder mask, shape (batch, seq_len).
    # s: decoder state s_t (one step), shape (batch, hidden*2).
    # coverage: sum of attention scores, shape (batch, seq_len).
    def forward(self, h, mask, s, coverage):

        encoder_feature = self.w_h(h)  # (batch, seq_len, hidden*2)
        decoder_feature = self.w_s(s).unsqueeze(1)  # (batch, 1, hidden*2)

        # Broadcast sum of encoder and decoder features.
        attention_feature = encoder_feature + decoder_feature  # (batch, seq_len, hidden*2)

        if self.use_coverage:
            coverage_feature = self.w_c(coverage.unsqueeze(2))  # (batch, seq_len, hidden*2)
            attention_feature += coverage_feature

        e_t = self.v(torch.tanh(attention_feature)).squeeze(dim=2)  # (batch, seq_len)

        mask_bool = (mask == 0)  # Positions to mask out are True.
        e_t.masked_fill_(mask=mask_bool, value=-float('inf'))

        a_t = torch.softmax(e_t, dim=-1)  # (batch, seq_len)

        if self.use_coverage:
            next_coverage = coverage + a_t
        else:
            next_coverage = None
        return a_t, next_coverage


class GeneraProb(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
        super(GeneraProb, self).__init__()

        self.w_h = nn.Linear(hidden_dim * 2, 1)
        self.w_s = nn.Linear(hidden_dim * 2, 1)
        self.w_x = nn.Linear(embed_dim, 1)

    # h : weight sum of encoder output ,(batch,hidden*2)
    # s : decoder state                 (batch,hidden*2)
    # x : decoder input                 (batch,embed)
    def forward(self, h, s, x):
        h_feature = self.w_h(h)  # (batch,1)
        s_feature = self.w_s(s)  # (batch,1)
        x_feature = self.w_x(x)  # (batch,1)

        gen_feature = h_feature + s_feature + x_feature  # (batch,1)

        gen_p = torch.sigmoid(gen_feature)

        return gen_p


class Decoder(nn.Module):
    def __init__(self, vob_size, embed_dim, hidden_dim, layer_num=1, dropout=0.5, pad_idx=0, pointer_gen=True,
                 use_coverage=False):
        super(Decoder, self).__init__()

        self.pointer_gen = pointer_gen
        self.use_coverage = use_coverage

        self.embedding = nn.Embedding(
            num_embeddings=vob_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.get_lstm_input = nn.Linear(in_features=hidden_dim * 2 + embed_dim,
                                        out_features=embed_dim)

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=layer_num,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=False)

        self.attention = Attention(hidden_dim=hidden_dim, use_coverage=use_coverage)

        if pointer_gen:
            self.genera_prob = GeneraProb(hidden_dim=hidden_dim,
                                          embed_dim=embed_dim)

        self.dropout = nn.Dropout(p=dropout)

        # Project the concatenated decoder output + context vector to the vocab.
        self.out = nn.Sequential(nn.Linear(in_features=hidden_dim * 3, out_features=hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hidden_dim, out_features=vob_size))

    # decoder_input_one_step: (batch, 1) the token fed at the current step.
    # decoder_status: (h_t, c_t) with h_t of shape (1, batch, hidden).
    # encoder_output: (batch, seq_len, hidden*2).
    # encoder_mask: (batch, seq_len).
    # context_vec: (batch, hidden*2) attention-weighted sum of encoder outputs.
    # oovs_zero: (batch, max_oov_size) all-zero tensor.
    # encoder_with_oov: (batch, seq_len) token indices (may exceed vocab size for OOV).
    # coverage: sum of attention at each step.

    def forward(self, decoder_input_one_step, decoder_status, encoder_output,
                encoder_mask, context_vec, oovs_zero, encoder_with_oov, coverage, step):

        embed = self.embedding(decoder_input_one_step)  # (batch, embed_dim)
        x = self.get_lstm_input(torch.cat([context_vec, embed], dim=-1)).unsqueeze(
            dim=1)  # (batch, 1, hidden*2 + embed_dim)

        decoder_status = tuple([_status.contiguous() for _status in decoder_status])

        decoder_output, next_decoder_status = self.lstm(x, decoder_status)

        h_t, c_t = next_decoder_status

        batch_size = c_t.shape[1]

        h_t_reshape = h_t.reshape(batch_size, -1)
        c_t_reshape = c_t.reshape(batch_size, -1)

        status = torch.cat([h_t_reshape, c_t_reshape], dim=-1)  # (batch, hidden*2)

        # attention_score: (batch, seq_len) weight of each encoder token.
        # next_coverage: (batch, seq_len) cumulative attention.
        attention_score, next_coverage = self.attention(h=encoder_output,
                                                        mask=encoder_mask,
                                                        s=status,
                                                        coverage=coverage)

        # Context vector: attention-weighted sum of encoder outputs, (batch, hidden*2).
        current_context_vec = torch.einsum("ab,abc->ac", attention_score, encoder_output)

        # Genera_p: (batch, 1) probability of copying from the source vs generating.
        genera_p = None
        if self.pointer_gen:
            genera_p = self.genera_prob(h=current_context_vec,
                                        s=status,
                                        x=x.squeeze())

        # (batch, hidden*3)
        out_feature = torch.cat([decoder_output.squeeze(dim=1), current_context_vec], dim=-1)

        # (batch, vob_size)
        output = self.out(out_feature)
        vocab_dist = torch.softmax(output, dim=-1)

        if self.pointer_gen:
            # Mix the vocabulary distribution with the copy (attention) distribution.
            vocab_dist_p = vocab_dist * genera_p
            context_dist_p = attention_score * (1 - genera_p)
            if oovs_zero is not None:
                vocab_dist_p = torch.cat([vocab_dist_p, oovs_zero], dim=-1)
            final_dist = vocab_dist_p.scatter_add(dim=-1, index=encoder_with_oov, src=context_dist_p)
        else:
            final_dist = vocab_dist

        return final_dist, next_decoder_status, current_context_vec, attention_score, genera_p, next_coverage


class PointerGeneratorNetworks(nn.Module):
    def __init__(
            self,
            vob_size=50000,  # vocabulary size
            embed_dim=64,  # embedding dimension
            hidden_dim=256,  # number of LSTM hidden units
            pad_idx=0,  # index of the padding token
            dropout=0.5,  # dropout ratio
            pointer_gen=True,  # whether to use the pointer-generator mechanism
            use_coverage=False,  # whether to use coverage loss
            eps=1e-12,  # small constant for numerical stability
            coverage_loss_weight=1.0,
            unk_token_idx=1,
            start_token_idx=2,
            stop_token_idx=3,
            max_decoder_len=32,  # maximum decoder output length
            min_decoder_len=1,  # minimum decoder output length
    ):
        super(PointerGeneratorNetworks, self).__init__()
        self.all_mode = ["train", "eval", "decode"]

        self.vob_size = vob_size
        self.use_coverage = use_coverage
        self.eps = eps
        self.coverage_loss_weight = coverage_loss_weight
        self.max_decoder_len = max_decoder_len
        self.min_decoder_len = min_decoder_len
        self.start_token_idx = start_token_idx
        self.stop_token_idx = stop_token_idx
        self.unk_token_idx = unk_token_idx

        self.encoder = Encoder(
            vob_size=vob_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            pad_idx=pad_idx,
            dropout=0.5
        )

        self.reduce = Reduce(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.decoder = Decoder(
            vob_size=vob_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pad_idx=pad_idx,
            pointer_gen=pointer_gen,
            use_coverage=use_coverage
        )

    # encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec, coverage, beam_size

    def forward(
            self,
            encoder_input,
            encoder_mask,
            encoder_with_oov,
            oovs_zero=None,
            context_vec=None,
            coverage=None,
            decoder_input=None,
            decoder_mask=None,
            decoder_target=None,
            mode="train",
            start_tensor=None,
            beam_size=4,
            test_batch_size=1,
            per_node_beam_size=40
    ):

        if mode in ["train", "eval"]:
            return self._forward(encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec, coverage,
                                 decoder_input, decoder_mask, decoder_target)
        elif mode in ["cal"]:
            return self.forward_eval(encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec, coverage,
                                     decoder_input, decoder_mask, decoder_target)
        elif mode in ["code_decode"]:
            if test_batch_size == 1:
                return self.code_decoder(
                    encoder_input=encoder_input,
                    encoder_mask=encoder_mask,
                    encoder_with_oov=encoder_with_oov,
                    oovs_zero=oovs_zero,
                    context_vec=context_vec,
                    coverage=coverage,
                    beam_size=beam_size
                )

    def _forward(
            self,
            encoder_input,
            encoder_mask,
            encoder_with_oov,
            oovs_zero=None,
            context_vec=None,
            coverage=None,
            decoder_input=None,
            decoder_mask=None,
            decoder_target=None
    ):
        # Supervised training forward pass: returns the masked average NLL loss.
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)
        decoder_status = self.reduce(*encoder_hidden)

        decoder_lens = decoder_mask.sum(dim=-1)
        batch_max_decoder_len = decoder_lens.max()
        assert batch_max_decoder_len <= self.max_decoder_len

        all_step_loss = []
        for step in range(batch_max_decoder_len):
            decoder_input_one_step = decoder_input[:, step]

            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_input_one_step=decoder_input_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs,
                    encoder_mask=encoder_mask,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero,
                    encoder_with_oov=encoder_with_oov,
                    coverage=coverage,
                    step=step)

            target = decoder_target[:, step].unsqueeze(1)
            probs = torch.gather(final_dist, dim=-1, index=target).squeeze()
            step_loss = -torch.log(probs + self.eps)

            if self.use_coverage:
                coverage_loss = self.coverage_loss_weight * torch.sum(torch.min(attention_score, coverage), dim=-1)
                step_loss += coverage_loss
                coverage = next_coverage

            all_step_loss.append(step_loss)

        # token_loss.shape = [batch_size, batch_max_decoder_len]
        if all_step_loss[0].numel() != 1:
            token_loss = torch.stack(all_step_loss, dim=1)
        else:
            token_loss = torch.stack(all_step_loss, dim=-1)

        decoder_mask_cut = decoder_mask[:, :batch_max_decoder_len].float()

        token_loss_with_mask = token_loss * decoder_mask_cut
        batch_loss_sum_token = token_loss_with_mask.sum(dim=-1)
        batch_loss_mean_token = batch_loss_sum_token / decoder_lens.float()
        result_loss = batch_loss_mean_token.mean()

        return result_loss

    # Compute the joint probability of a given password pair.
    def forward_eval(
            self,
            encoder_input,
            encoder_mask,
            encoder_with_oov,
            oovs_zero=None,
            context_vec=None,
            coverage=None,
            decoder_input=None,
            decoder_mask=None,
            decoder_target=None
    ):
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)
        decoder_status = self.reduce(*encoder_hidden)

        decoder_lens = decoder_mask.sum(dim=-1)
        batch_max_decoder_len = decoder_lens.max()
        assert batch_max_decoder_len <= self.max_decoder_len

        # Track which samples have already produced the stop token so their
        # probability is not multiplied any further.
        end_mask = torch.zeros(len(encoder_input), dtype=torch.bool, device=config.device)
        res_probs = None
        for step in range(batch_max_decoder_len):
            decoder_input_one_step = decoder_input[:, step]

            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_input_one_step=decoder_input_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs,
                    encoder_mask=encoder_mask,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero,
                    encoder_with_oov=encoder_with_oov,
                    coverage=coverage,
                    step=step)

            target = decoder_target[:, step].unsqueeze(1)
            batch_size = len(encoder_input)
            update_mask = ~end_mask
            if res_probs is None:
                res_probs = final_dist[torch.arange(batch_size), target.squeeze(1)]
            else:
                res_probs[update_mask] *= final_dist[update_mask, target.squeeze(1)[update_mask]]
            is_end = (target.squeeze(1) == 1)
            end_mask |= is_end
        return res_probs

    def freeze_layers(self):
        # Freeze the decoder and reduce layers for continual-learning reuse.
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.reduce.parameters():
            param.requires_grad = False

