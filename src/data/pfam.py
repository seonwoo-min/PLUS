# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" Pfam FASTA file loading functions """

import numpy as np
import pandas as pd

import torch

from src.data.fasta import parse_stream


def load_pfam(cfg, idx, encoder, sanity_check=False):
    """ load Pfam sequence data from FASTA file """
    with open(cfg.path[idx], 'rb') as f:
        struct_dict = {}
        for name, sequence in parse_stream(f):
            # input sequence length configurations
            if sanity_check: break
            if cfg.min_len != -1 and len(sequence) < cfg.min_len: continue
            if cfg.max_len != -1 and len(sequence) > cfg.max_len: continue

            struct = name.decode("utf-8").split()[2].split(";")[1]
            if struct not in struct_dict: struct_dict[struct] = 0
            struct_dict[struct] += 1

    with open(cfg.path[idx], 'rb') as f:
        sequences, structs, structs_idx, struct_dict_filtered, n = [], [], [], {}, 0
        for name, sequence in parse_stream(f):
            # input sequence length configurations
            sequence = encoder.encode(sequence.upper())
            struct = name.decode("utf-8").split()[2].split(";")[1]

            if cfg.min_len != -1 and len(sequence) < cfg.min_len: continue
            if cfg.max_len != -1 and len(sequence) > cfg.max_len: continue
            if cfg.min_seq != -1 and not sanity_check and (struct not in struct_dict or struct_dict[struct] < cfg.min_seq): continue
            if cfg.truncate != -1 and len(sequence) > cfg.truncate: sequence = sequence[:cfg.truncate]
            if sanity_check and len(sequences) == 100: break

            if struct not in struct_dict_filtered:
                structs_idx.append([])
                struct_dict_filtered[struct] = len(struct_dict_filtered)
            idx = struct_dict_filtered[struct]
            sequences.append(sequence); structs.append(idx); structs_idx[idx].append(n); n+= 1

    sequences = [torch.from_numpy(sequence).long() for sequence in sequences]
    structs = torch.from_numpy(np.stack(structs, 0))

    return sequences, structs, structs_idx


def load_pfam_pairs(cfg, idx, encoder, sanity_check=False):
    """ load Pfam pairs data from csv file """
    table = pd.read_csv(cfg.path[idx], sep='\t')
    sequences0_all = [encoder.encode(sequence.encode('utf-8').upper()) for sequence in table['sequence_A']]
    sequences1_all = [encoder.encode(sequence.encode('utf-8').upper()) for sequence in table['sequence_B']]
    labels_all = table['is_samefamily'].values

    # input sequence length configurations
    sequences0, sequences1, labels = [], [], []
    for i, (sequence0, sequence1) in enumerate(zip(sequences0_all, sequences1_all)):
        if cfg.min_len != -1 and len(sequence0) <  cfg.min_len: continue
        if cfg.max_len != -1 and len(sequence0) >  cfg.max_len: continue
        if cfg.min_len != -1 and len(sequence1) <  cfg.min_len: continue
        if cfg.max_len != -1 and len(sequence1) >  cfg.max_len: continue
        if sanity_check and len(sequences0) == 100: break

        sequences0.append(sequence0.astype(np.long))
        sequences1.append(sequence1.astype(np.long))
        labels.append(labels_all[i].astype(np.float32))

    sequences0 = [torch.from_numpy(sequence0) for sequence0 in sequences0]
    sequences1 = [torch.from_numpy(sequence1) for sequence1 in sequences1]
    labels = torch.from_numpy(np.stack(labels)).unsqueeze(1).byte()

    return sequences0, sequences1, labels

