# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" Localization FASTA file loading functions """

import numpy as np

import torch

from plus.data.fasta import parse_stream


locs = {"Cell.membrane":0, "Cytoplasm":1, "Endoplasmic.reticulum":2, "Extracellular":3, "Golgi.apparatus":4,
        "Lysosome/Vacuole":5, "Mitochondrion":6, "Nucleus":7, "Peroxisome":8, "Plastid":9}


def load_localization(cfg, idx, encoder, sanity_check=False):
    """ load Localization sequence data from FASTA file """
    with open(cfg.path[idx], 'rb') as f:
        sequences, labels = [], []
        for name, sequence in parse_stream(f):
            # protein sequence length configurations
            if cfg.min_len != -1 and len(sequence) <  cfg.min_len: continue
            if cfg.max_len != -1 and len(sequence) >  cfg.max_len: continue
            if cfg.truncate != -1 and len(sequence) > cfg.truncate:
                sequence = sequence[:cfg.truncate // 2] + sequence[- cfg.truncate // 2:]
            if sanity_check and len(sequences) == 100: break

            sequence = encoder.encode(sequence.upper())
            label = name.decode("utf-8").strip().split()[1].split("-")[0]
            sequences.append(sequence), labels.append(np.array([locs[label]]))

    sequences = [torch.from_numpy(sequence).long() for sequence in sequences]
    labels = torch.from_numpy(np.stack(labels))
    return sequences, labels
