# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)

""" FASTA file loading functions """

import numpy as np

import torch
import torch.nn.functional as F

def load_fasta(cfg, idx, encoder, onehot=True, label=True, cls=True, sanity_check=False):
    """ load sequence data from FASTA file """
    with open(cfg.path[idx], 'rb') as f:
        sequences, labels = [], []
        if onehot: eye = torch.eye(len(encoder) - 1).numpy() # excluding 0 reserved for padding tokens
        for name, sequence in parse_stream(f):
            # input sequence length configurations
            sequence = encoder.encode(sequence.upper())
            if cfg.min_len != -1 and len(sequence) < cfg.min_len: continue
            if cfg.max_len != -1 and len(sequence) > cfg.max_len: continue
            if cfg.truncate != -1 and len(sequence) > cfg.truncate: sequence = sequence[:cfg.truncate]
            if sanity_check and len(sequences) == 100: break

            if onehot: sequences.append(np.transpose(eye[sequence - 1]).astype(np.float32))
            else: sequences.append(sequence.astype(np.long))
            if label: labels.append(float(name.decode("utf-8").split()[-1]))

    sequences = [torch.from_numpy(sequence) for sequence in sequences]
    labels = torch.from_numpy(np.stack(labels))
    if cls: labels = F.one_hot(labels.long()).float()
    return sequences, labels


def parse_stream(f, comment=b'#'):
    """ parse fasta stream for (name, sequence) data """
    name, sequence = None, []
    for line in f:
        if line.startswith(comment): continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                yield name, b''.join(sequence)
            name, sequence = line[1:], []
        else:
            sequence.append(line.upper())
    if name is not None:
        yield name, b''.join(sequence)


# PLUS

def parse_ss_stream(f, comment=b'#'):
    """ parse fasta stream for (name, sequence, secstr) data """
    flag = -1
    name, sequence, secstr = None, [], []
    for line in f:
        if line.startswith(comment): continue
        line = line.rstrip(b'\r\n')
        if line.startswith(b'>'):
            if name is not None and flag == 1:
                yield name, b''.join(sequence), b''.join(secstr)
            elif flag == 0:
                assert line[1:].startswith(name)

            # each sequence has an amino acid sequence and secstr sequence associated with it
            name = line[1:]
            tokens = name.split(b':')
            name = b':'.join(tokens[:-1])
            flag = tokens[-1]

            if flag == b'sequence':
                flag = 0
                sequence = []
                secstr = []
            elif flag == b'secstr':
                flag = 1
            else:
                raise Exception("Unrecognized flag: " + flag.decode())

        elif flag == 0:
            sequence.append(line)
        elif flag == 1:
            secstr.append(line)
        else:
            raise Exception("Flag not set properly")

    if name is not None:
        yield name, b''.join(sequence), b''.join(secstr)

