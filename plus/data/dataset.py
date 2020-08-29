# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" PyTorch dataset classes and functions """

import sys
import random
import numpy as np

import torch.utils.data

from plus.preprocess import preprocess_seq_for_rnn, preprocess_seq_for_tfm, preprocess_label_for_tfm


class Pfam_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS Pfam training
        if random_pairing is True make random pairs from the same or different families with 0.5 probability """
    def __init__(self, sequences, structs, structs_idx, encoder, cfg, rnn=True, max_len=None, random_pairing=True, augment=True, sanity_check=False):
        self.sequences = sequences
        self.structs = structs
        self.structs_idx = structs_idx
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.augment = augment
        self.max_len = max_len
        self.random_pairing = random_pairing
        self.sanity_check = sanity_check

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        if not self.random_pairing:
            if self.rnn:
                instance = preprocess_seq_for_rnn(self.sequences[i], self.num_alphabets, self.cfg, self.augment)
                return instance
            else:
                sys.exit("PLUS-TFM Pfam pre-training without random pairing is not supported")
        else:
            trial = 0
            while(1):
                sequence0 = self.sequences[i + trial // 10]
                is_samefamily = random.random() > 0.5
                if is_samefamily: family = self.structs[i + trial // 10]
                else:
                    while(1):
                        family = random.randint(0, len(self.structs_idx) - 1)
                        if family != self.structs[i + trial // 10] or self.sanity_check: break
                sequence1 = self.sequences[random.choice(self.structs_idx[family])]
                is_samefamily = torch.tensor([is_samefamily])
                if self.max_len is None or len(sequence0) + len(sequence1) + 3 <= self.max_len: break
                else: trial += 1

            if self.rnn:
                instance0 = preprocess_seq_for_rnn(sequence0, self.num_alphabets, self.cfg)
                instance1 = preprocess_seq_for_rnn(sequence1, self.num_alphabets, self.cfg)
                return instance0, instance1, is_samefamily
            else:
                instance = preprocess_seq_for_tfm(sequence0, sequence1, self.num_alphabets, self.cfg, self.max_len)
                return (*instance, is_samefamily)


class PairedPfam_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS Pfam evaluation """
    def __init__(self, sequences0, sequences1, labels, encoder, cfg, rnn=False, max_len=None):
        self.sequences0 = sequences0
        self.sequences1 = sequences1
        self.labels = labels
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.augment = True
        if not self.rnn: self.set_max_len(max_len)

    def __len__(self):
        return len(self.sequences0)

    def __getitem__(self, i):
        sequence0, sequence1 = self.sequences0[i], self.sequences1[i]
        is_samefamily = self.labels[i]
        if self.rnn:
            instance0 = preprocess_seq_for_rnn(sequence0, self.num_alphabets, self.cfg, self.augment)
            instance1 = preprocess_seq_for_rnn(sequence1, self.num_alphabets, self.cfg, self.augment)
            return instance0, instance1, is_samefamily
        else:
            instance = preprocess_seq_for_tfm(sequence0, sequence1, self.num_alphabets, self.cfg, self.max_len, self.augment)
            return (*instance, is_samefamily)

    def set_max_len(self, max_len):
        """ set max_len """
        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = 128
            for sequence0, sequence1 in zip(self.sequences0, self.sequences1):
                if len(sequence0) + len(sequence1) + 3 > self.max_len:
                    self.max_len = len(sequence0) + len(sequence1) + 3

    def set_augment(self, augment):
        """ set augmentation flag """
        self.augment = augment


class Homology_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS Homology training - make all pairs """
    def __init__(self, sequences, labels, cmaps, encoder, cfg, rnn=True, max_len=None):
        self.sequences = sequences
        self.labels = labels
        self.cmaps = cmaps
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)**2

    def __getitem__(self, k):
        n = len(self.sequences)
        i, j = k // n, k % n
        sequence0, sequence1 = self.sequences[i], self.sequences[j]
        similarity_level = self.labels[i, j]

        if self.rnn:
            instance0 = preprocess_seq_for_rnn(sequence0, self.num_alphabets, self.cfg)
            instance1 = preprocess_seq_for_rnn(sequence1, self.num_alphabets, self.cfg)
            if self.cmaps is not None: return instance0, instance1, similarity_level, self.cmaps[i], self.cmaps[j]
            else:                      return instance0, instance1, similarity_level
        else:
            instance = preprocess_seq_for_tfm(sequence0, sequence1, self.num_alphabets, self.cfg, self.max_len)
            if self.cmaps is not None: return (*instance, similarity_level, self.cmaps[i], self.cmaps[j])
            else:                      return (*instance, similarity_level)


class PairedHomology_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS Homology evaluation """
    def __init__(self, sequences0, sequences1, labels, cmaps0, cmaps1, encoder, cfg, rnn=False, max_len=None):
        self.sequences0 = sequences0
        self.sequences1 = sequences1
        self.labels = labels
        self.cmaps0 = cmaps0
        self.cmaps1 = cmaps1
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.augment = True
        if not self.rnn: self.set_max_len(max_len)

    def __len__(self):
        return len(self.sequences0)

    def __getitem__(self, i):
        sequence0, sequence1 = self.sequences0[i], self.sequences1[i]
        similarity_level = self.labels[i]
        if self.rnn:
            instance0 = preprocess_seq_for_rnn(sequence0, self.num_alphabets, self.cfg, self.augment)
            instance1 = preprocess_seq_for_rnn(sequence1, self.num_alphabets, self.cfg, self.augment)
            if self.cmaps0 is not None: return instance0, instance1, similarity_level, self.cmaps0[i], self.cmaps1[i]
            else:                       return instance0, instance1, similarity_level
        else:
            instance = preprocess_seq_for_tfm(sequence0, sequence1, self.num_alphabets, self.cfg, self.max_len, self.augment)
            if self.cmaps0 is None: return (*instance, similarity_level)
            else:                   return (*instance, similarity_level, self.cmaps0[i], self.cmaps1[i])

    def set_max_len(self, max_len):
        """ set max_len """
        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = 128
            for sequence0, sequence1 in zip(self.sequences0, self.sequences1):
                if len(sequence0) + len(sequence1) + 3 > self.max_len:
                    self.max_len = len(sequence0) + len(sequence1) + 3

    def set_augment(self, augment):
        """ set augmentation flag """
        self.augment = augment


class Seq_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS single sequence task training and evaluation """
    def __init__(self, sequences, labels, encoder, cfg, rnn=False, max_len=None, truncate=True):
        self.sequences = sequences
        self.labels = labels
        self.valids = None
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.truncate = truncate
        self.augment = True
        if not self.rnn: self.set_max_len(max_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        if self.rnn:
            instance = preprocess_seq_for_rnn(self.sequences[i], self.num_alphabets, self.cfg, self.augment)
            return instance, self.labels[i]
        else:
            instance_seq = preprocess_seq_for_tfm(self.sequences[i], None, self.num_alphabets, self.cfg, self.max_len, self.augment)
            if self.valids is None:
                return (*instance_seq, self.labels[i])
            else:
                instance_label = preprocess_label_for_tfm(self.labels[i], self.valids[i], self.max_len)
                return (*instance_seq, *instance_label)

    def set_max_len(self, max_len):
        """ set max_len """
        if max_len is not None:
            self.max_len = max_len
            if not self.truncate:
                # split sequences/labels longer than max_len
                sequences, labels, valids, l = [], [], [], self.max_len - 2
                for i in range(len(self.sequences)):
                    seq, label = self.sequences[i], self.labels[i]
                    while len(seq) > self.max_len - 2:
                        sequences.append(seq[:l]);  seq = seq[l:]
                        labels.append(label[:l]);   label = label[l:]
                        valids.append(False)
                    sequences.append(seq); labels.append(label); valids.append(True)

                self.sequences = sequences
                self.labels = labels
                self.valids = valids
        else:
            self.max_len = 128
            for sequence in self.sequences:
                if len(sequence) > self.max_len:
                    self.max_len = len(sequence) + 2

    def set_augment(self, augment):
        """ set augmentation flag """
        self.augment = augment


class Embedding_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for protein sequence embedding """
    def __init__(self, sequences, encoder, cfg, rnn=False):
        self.sequences = sequences
        self.valids = None
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        if not self.rnn: self.set_max_len()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        if self.rnn: instance = preprocess_seq_for_rnn(self.sequences[i], self.num_alphabets, self.cfg, augment=False)
        else:        instance = preprocess_seq_for_tfm(self.sequences[i], None, self.num_alphabets, self.cfg, self.max_len, augment=False)

        return instance

    def set_max_len(self):
        """ set max_len """
        self.max_len = 128
        for sequence in self.sequences:
            if len(sequence) > self.max_len:
                self.max_len = len(sequence) + 2


def collate_sequences_pelmo(x):
    """ collate sequences with different lengths into a single matrix
        to match the P-ELMo(ICLR19_Bepler) implementation, use 0 for [START / STOP] tokens and mask_idx for [PADDING] """
    lengths = np.array([len(seq) for seq in x])
    b, l = len(x), max(lengths)

    x_block = x[0].new_zeros((b, l)) + x[0][0]
    for i in range(b):
        seq = x[i]
        x_block[i, 1:len(seq)-1] = seq[1:-1]
        x_block[i, 0] = 0
        x_block[i, len(seq)-1] = 0
    lengths = torch.from_numpy(lengths)
    return x_block, lengths


def collate_sequences(args):
    """ collate sequences with different lengths into a single matrix; use 0 for [START / STOP/ PADDING] tokens """
    label = (len(args[0]) == 2)
    if label:
        mlm = (len(args[0][0]) == 4)
        if mlm:
            x = [a[0][0] for a in args]
            masked_pos = torch.stack([a[0][1] for a in args], 0)
            masked_tokens = torch.stack([a[0][2] for a in args], 0)
            masked_weights = torch.stack([a[0][3] for a in args], 0)
        else: x = [a[0] for a in args]
        y = [a[1] for a in args]
        amino_level = (len(y[0]) != 1)
    else:
        mlm = (len(args[0]) == 4)
        if mlm:
            x = [a[0] for a in args]
            masked_pos = torch.stack([a[1] for a in args], 0)
            masked_tokens = torch.stack([a[2] for a in args], 0)
            masked_weights = torch.stack([a[3] for a in args], 0)
        else: x = [a for a in args]

    lengths = np.array([len(seq) for seq in x])
    b, l = len(x), max(lengths)
    x_block = x[0].new_zeros((b, l))
    for i in range(b):
        x_block[i, 1:len(x[i])-1] = x[i][1:-1]
    lengths = torch.from_numpy(lengths)

    if label:
        if amino_level:
            y_block = y[0].new_zeros((b, l - 2))  # amino-acid level task
            for i in range(b): y_block[i, :len(y[i])] = y[i]
        else:
            y_block = torch.stack(y, 0)  # protein-level task

    if mlm and label: return x_block, lengths, masked_pos, masked_tokens, masked_weights, y_block
    elif label:       return x_block, lengths, y_block
    elif mlm:         return x_block, lengths, masked_pos, masked_tokens, masked_weights
    else:             return x_block, lengths


def collate_paired_sequences(args):
    """ collate paired sequences with different lengths into a single matrix; use 0 for [START / STOP/ PADDING] tokens """
    mlm = (len(args[0][0]) == 4)
    if mlm:
        x0 = [a[0][0] for a in args]
        masked_pos0     = torch.stack([a[0][1] for a in args], 0)
        masked_tokens0  = torch.stack([a[0][2] for a in args], 0)
        masked_weights0 = torch.stack([a[0][3] for a in args], 0)
        x1 = [a[1][0] for a in args]
        masked_pos1     = torch.stack([a[1][1] for a in args], 0)
        masked_tokens1  = torch.stack([a[1][2] for a in args], 0)
        masked_weights1 = torch.stack([a[1][3] for a in args], 0)
    else:
        x0 = [a[0] for a in args]
        x1 = [a[1] for a in args]
    y = torch.stack([a[2] for a in args], 0)

    cmap = (len(args[0]) != 3)
    if cmap:
        cmaps0 = [a[3] for a in args]
        cmaps1 = [a[4] for a in args]

    lengths0 = np.array([len(seq) for seq in x0])
    lengths1 = np.array([len(seq) for seq in x1])
    x0_block = x0[0].new_zeros((len(x0), max(lengths0)))
    x1_block = x1[0].new_zeros((len(x1), max(lengths1)))
    for i in range(len(x0)):
        x0_block[i, 1:len(x0[i])-1] = x0[i][1:-1]
        x1_block[i, 1:len(x1[i])-1] = x1[i][1:-1]
    lengths0 = torch.from_numpy(lengths0)
    lengths1 = torch.from_numpy(lengths1)

    if mlm and cmap:       return x0_block, lengths0, masked_pos0, masked_tokens0, masked_weights0, x1_block, lengths1, masked_pos1, masked_tokens1, masked_weights1, y, cmaps0, cmaps1
    elif mlm and not cmap: return x0_block, lengths0, masked_pos0, masked_tokens0, masked_weights0, x1_block, lengths1, masked_pos1, masked_tokens1, masked_weights1, y
    elif not mlm and cmap: return x0_block, lengths0, x1_block, lengths1, y, cmaps0, cmaps1
    else:                  return x0_block, lengths0, x1_block, lengths1, y


def collate_sequences_for_embedding(args):
    x = [a for a in args]
    lengths = np.array([len(seq) for seq in x])
    b, l = len(x), max(lengths)
    x_block = x[0].new_zeros((b, l))
    for i in range(b):
        x_block[i, 1:len(x[i])-1] = x[i][1:-1]
    lengths = torch.from_numpy(lengths)
    return x_block, lengths


class HomolgySampler(torch.utils.data.sampler.Sampler):
    """ Weighted sampling of considering the similarity levels and their number of seq pairs """
    def __init__(self, labels, cfg):
        similarity = labels.numpy().sum(2)
        levels, counts = np.unique(similarity, return_counts=True)
        order = np.argsort(levels)
        levels, counts = levels[order], counts[order]
        weights = counts ** cfg.tau / counts
        weights = torch.as_tensor(weights, dtype=torch.double)

        similarity = similarity.ravel()
        levels, counts = np.unique(similarity, return_counts=True)
        order = np.argsort(levels)
        levels, counts = levels[order], counts[order]
        similarity_counts = np.zeros((len(levels) + 1), dtype=np.int32)
        for i in range(len(levels)):
            similarity_counts[i+1] = similarity_counts[i] + counts[i]
        similarity_order = np.argsort(similarity)

        self.weights = weights
        self.similarity_counts = similarity_counts
        self.similarity_order = similarity_order
        self.num_samples = cfg.epoch_size
        self.replacement = False

    def __iter__(self):
        level_sampling = torch.multinomial(self.weights, self.num_samples, replacement=True)
        sampled_levels, sampled_counts = np.unique(level_sampling, return_counts=True)

        sampled_pairs = []
        for l, c in zip(sampled_levels, sampled_counts):
            idxs = np.random.randint(0, self.similarity_counts[l]+1, c)
            idxs = self.similarity_order[idxs]
            sampled_pairs += idxs.tolist()

        return iter(sampled_pairs)

    def __len__(self):
        return self.num_samples

