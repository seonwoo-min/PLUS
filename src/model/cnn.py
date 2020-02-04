# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" CNN model classes and functions """

import torch
import torch.nn as nn


class ConvNet2D(nn.Module):
    def __init__(self, embed_dim, num=50, width=7):
        """ CNN model for protein contact map prediction """
        super(ConvNet2D, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(2*embed_dim, num, 1)
        self.conv2 = nn.Conv2d(num, 1, width, padding=width//2)
        self.clip()

    def forward(self, z):
        z = z.transpose(1, 2) #(b,L,d) -> (b,d,L)
        z_dif = torch.abs(z.unsqueeze(2) - z.unsqueeze(3))
        z_mul = z.unsqueeze(2) * z.unsqueeze(3)
        z = torch.cat([z_dif, z_mul], 1) # (b,2d,L,L)
        h = self.relu(self.conv1(z))
        logits = self.conv2(h).squeeze(1)
        return logits

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = self.state_dict()
        for key, value in torch.load(pretrained_model, map_location=torch.device('cpu')).items():
            if key.startswith("module"): key = key[7:]
            if key in state_dict and value.shape == state_dict[key].shape: state_dict[key] = value
        self.load_state_dict(state_dict)

    def clip(self):
        # force the conv layer to be transpose invariant
        w = self.conv2.weight
        self.conv2.weight.data[:] = 0.5*(w + w.transpose(2,3))


