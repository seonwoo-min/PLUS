# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" SecStr FASTA file loading functions """

import numpy as np

import torch
import torch.nn.functional as F

from src.data.fasta import parse_ss_stream


def load_secstr(cfg, idx, encoder, sanity_check=False):
    """ load sequence and SecStr data from FASTA file """
    with open(cfg.path[idx], 'rb') as f:
        sequences, secstrs = [], []
        for name, sequence, secstr in parse_ss_stream(f):
            # input sequence length configurations
            sequence = encoder.encode(sequence.upper())
            secstr = secstr.decode("utf-8").strip()
            secstr_np = np.zeros(len(secstr), dtype=float)
            for i in range(len(secstr)):
                secstr_np[i] = float(secstr[i])
            if cfg.min_len != -1 and len(sequence) < cfg.min_len: continue
            if cfg.max_len != -1 and len(sequence) > cfg.max_len: continue
            if cfg.truncate != -1 and len(sequence) > cfg.truncate: sequence = sequence[:cfg.truncate]
            if sanity_check and len(sequences) == 100: break

            sequences.append(sequence); secstrs.append(secstr_np)

    sequences = [torch.from_numpy(sequence).long() for sequence in sequences]
    secstrs = [torch.from_numpy(secstr).long() for secstr in secstrs]

    return sequences, secstrs


def evaluate_secstr(result):
    """ evaluate secstr classification task (acc8 / acc3) """
    if "logits" not in result: return
    logits, labels = result["logits"], result["labels"]

    # matrix for 3-class classification
    A, I = np.zeros((8, 3), dtype=np.float32), np.zeros(8, dtype=int)
    A[0, 0] = A[1, 0] = A[2, 0] = 1.0; I[0] = I[1] = I[2] = 0
    A[3, 1] = A[4, 1] = 1.0;           I[3] = I[4] = 1
    A[5, 2] = A[6, 2] = A[7, 2] = 1.0; I[5] = I[6] = I[7] = 2
    A, I = torch.from_numpy(A), torch.from_numpy(I)

    if "label_weights" not in result:
        if not isinstance(logits, list): logits = [logits]
        if not isinstance(labels, list): labels = [labels]
        n, correct8, correct3 = 0, 0, 0
        b, j = 0, 0

        for i in range(len(logits)):
            l = len(logits[i][0])
            n += l
            p = torch.mm(F.softmax(logits[i][0], 1), A)
            correct8 += torch.sum((torch.max(logits[i][0], 1)[1] == labels[b][j][:l])).item()
            correct3 += torch.sum((torch.max(p, 1)[1] == I[labels[b][j]][:l])).item()

            if j == len(labels[b]) - 1: b, j = b+1, 0
            else: j += 1
    else:
        n = torch.sum(result["label_weights"]).item()
        p = torch.matmul(F.softmax(logits, 2), A)
        correct8 = torch.sum((torch.max(logits, 2)[1] == labels).masked_select(result["label_weights"])).item()
        correct3 = torch.sum((torch.max(p, 2)[1] == I[labels]).masked_select(result["label_weights"])).item()

    result["acc8"] = correct8 / (n + np.finfo(float).eps)
    result["acc3"] = correct3 / (n + np.finfo(float).eps)