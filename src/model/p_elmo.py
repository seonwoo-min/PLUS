# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" P-ELMo model classes and functions """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from src.model.plus_rnn import PLUS_RNN

class P_ELMo(PLUS_RNN):
    """ P-ELMo model """
    def __init__(self, cfg):
        super(PLUS_RNN, self).__init__()

        # LM hidden states -> EM input model (ELMo-style; use hidden states as input to EM model)
        self.fc_lm = nn.Linear(cfg.lm_dim, cfg.lm_proj_dim)
        self.x_embed = nn.Embedding(cfg.input_dim-1, cfg.lm_proj_dim, padding_idx=cfg.input_dim - 2)
        self.rnn = nn.LSTM(cfg.lm_proj_dim, cfg.hidden_dim, cfg.num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * cfg.hidden_dim, cfg.embedding_dim)

        # language modeling decoder
        self.decoder = nn.Linear(2 * cfg.hidden_dim, cfg.input_dim-1)

        # ordinal classification
        # initialize ordinal regression bias with 10 following ICLR19_Bepler
        bias_init = 10
        if cfg.num_classes is not None:
            self.ordinal_weight = nn.Parameter(torch.ones(1, cfg.num_classes))
            self.ordinal_bias = nn.Parameter(torch.zeros(cfg.num_classes) + bias_init)

    def forward(self, x_aa, x_lm, lengths):
        order = torch.argsort(lengths, descending=True)
        order_rev = torch.argsort(order)
        x_aa, x_lm, lengths = x_aa[order], x_lm[order], lengths[order]

        # language model + x embedding
        total_length = x_aa.shape[1] - 2
        for i in range(x_aa.size(0)): x_aa[i, 1:lengths[i] - 1] -= 1
        x_aa = pack_padded_sequence(x_aa[:, 1:-1], lengths - 2, batch_first=True)
        x_aa = PackedSequence(self.x_embed(x_aa.data), x_aa.batch_sizes)
        x_lm = pack_padded_sequence(x_lm, lengths - 2, batch_first=True)
        x_lm = PackedSequence(self.fc_lm(x_lm.data), x_lm.batch_sizes)
        h = PackedSequence(nn.ReLU()(x_aa.data + x_lm.data), x_aa.batch_sizes)

        # feed-forward bidirectional RNN
        self.rnn.flatten_parameters()
        h, _ = self.rnn(h)
        h = pad_packed_sequence(h, batch_first=True, total_length=total_length)[0][order_rev]

        # final projection
        z = self.fc(h)
        return z, h

    def em(self, h, lengths, cpu=False):
        # get representations with different lengths from the collated single matrix
        e = [None] * len(lengths)
        for i in range(len(lengths)):
            if cpu: e[i] = h[i, :lengths[i] - 2].cpu()
            else:   e[i] = h[i, :lengths[i] - 2]
        return e


class P_ELMo_lm(nn.Module):
    """ language model for P-ELMo """
    def __init__(self, cfg):
        super(P_ELMo_lm, self).__init__()
        self.embed = nn.Embedding(cfg.input_dim, cfg.input_dim-1, padding_idx=cfg.input_dim-1)

        input_dim = cfg.input_dim - 1
        layers = []
        for _ in range(cfg.num_layers):
            layers.append(nn.LSTM(input_dim, cfg.hidden_dim, 1, batch_first=True))
            input_dim = cfg.hidden_dim
        self.rnn = nn.ModuleList(layers)

        self.decoder = nn.Linear(cfg.hidden_dim, cfg.input_dim-1)

    def forward(self, x, lengths):
        x = self.embed(x)
        h_fwd, h_rvs = x[:, :-1], x[:, 1:]
        h_fwd, h_rvs = self.transform(h_fwd, self.reverse(h_rvs))

        b, l, d = h_fwd.size()
        h_fwd = self.decoder(h_fwd.contiguous().view(-1, d))
        h_rvs = self.decoder(h_rvs.contiguous().view(-1, d))

        zero = h_fwd.new_zeros((b, 1, h_fwd.size(1)))
        h_fwd = torch.cat((zero, h_fwd.view(b, l, -1)), 1)
        h_rvs = torch.cat((h_rvs.view(b, l, -1), zero), 1)
        logits = F.log_softmax(h_fwd + h_rvs, dim=2)

        return logits

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = self.state_dict()
        for key, value in torch.load(pretrained_model, map_location=torch.device('cpu')).items():
            if key.startswith("module"): key = key[7:]
            if key in state_dict and value.shape == state_dict[key].shape: state_dict[key] = value
        self.load_state_dict(state_dict)

    def encode(self, x, lengths):
        order = torch.argsort(lengths, descending=True)
        order_rev = torch.argsort(order)
        x, lengths = x[order], lengths[order]

        with torch.no_grad():
            x = self.embed(x)
            h_fwd, h_rvs = x[:, :-1], x[:, 1:]
            total_length = x.shape[1] - 1
            h_fwd = pack_padded_sequence(h_fwd, lengths - 1, batch_first=True)
            h_rvs = pack_padded_sequence(h_rvs, lengths - 1, batch_first=True)
            h = self.transform(h_fwd, self.reverse(h_rvs, total_length), total_length, last_only=False)[order_rev]

        return h

    def reverse(self, h, total_length=None):
        # reverse given input tensor for reverse direction RNN
        if isinstance(h, PackedSequence):
            h, lengths = pad_packed_sequence(h, batch_first=True, total_length=total_length)
            h_rvs = h.clone().zero_()
            for i in range(h.size(0)):
                n = lengths[i]
                idx = [j for j in range(n - 1, -1, -1)]
                idx = torch.LongTensor(idx).to(h.device)
                h_rvs[i, :n] = h[i].index_select(0, idx)
            # repack h_rvs
            h_rvs = pack_padded_sequence(h_rvs, lengths, batch_first=True)
        else:
            idx = [i for i in range(h.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx).to(h.device)
            h_rvs = h.index_select(1, idx)
        return h_rvs

    def transform(self, h_fwd, h_rvs, total_length=None, last_only=True):
        # transform into forward and reverse rnn hidden states
        h = []
        for l in range(len(self.rnn)):
            self.rnn[l].flatten_parameters()
            h_fwd, _ = self.rnn[l](h_fwd)
            h_rvs, _ = self.rnn[l](h_rvs)
            if not last_only:
                h.append(pad_packed_sequence(h_fwd, batch_first=True, total_length=total_length)[0][:, :-1])
                h.append(pad_packed_sequence(self.reverse(h_rvs), batch_first=True, total_length=total_length)[0][:, 1:])

        if not last_only:
            return torch.cat(h, 2)
        else:
            return h_fwd, self.reverse(h_rvs)
