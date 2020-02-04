# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" MLP model classes and functions """

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cfg, per_seq=False):
        """ MLP model for fine-tuning prediction tasks """
        super(MLP, self).__init__()
        self.drop = nn.Dropout(cfg.dropout)
        self.relu = nn.ReLU()
        self.per_seq = per_seq
        if self.per_seq:
            self.attention = nn.Linear(cfg.input_dim, 1)
        self.hidden = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.output = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, X):
        logits = []
        for x in X:
            if self.per_seq:
                att = self.attention(x)
                x = torch.sum(x * F.softmax(att, 1).expand_as(x), 0)
            x = self.drop(self.relu(self.hidden(x)))
            x = self.output(x)
            if self.per_seq: logits.append(x)
            else:            logits.append(x.unsqueeze(0))
        return logits

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        for key, value in torch.load(pretrained_model, map_location=torch.device('cpu')).items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)
