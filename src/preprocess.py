# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Dong-Hyun Lee's code (https://github.com/dhlee347/pytorchic-bert)
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" pre-processing functions """

import random

import torch


def preprocess_seq_for_rnn(x, num_alphabets, cfg, augment=True):
    """ pre-processing steps for PLUS-RNN pre-training """
    tokens = torch.zeros(len(x) + 2, dtype=torch.long) + (num_alphabets - 1)
    tokens[1:len(x) + 1] = x

    if not augment:
        return tokens

    elif cfg.mask_ratio > 0:
        masked_pos = torch.zeros(cfg.max_pred, dtype=torch.long)
        masked_tokens = torch.zeros(cfg.max_pred, dtype=torch.long)
        masked_weights = torch.zeros(cfg.max_pred, dtype=torch.bool)
        # the number of prediction is usually less than max_pred when sequence is short
        n_pred = min(cfg.max_pred, int(len(x) * cfg.mask_ratio))
        # candidate positions of masked tokens
        cand_pos = [i for i in range(len(tokens)) if tokens[i] != (num_alphabets - 1)]
        random.shuffle(cand_pos)
        for i, pos in enumerate(cand_pos[:n_pred]):
            masked_pos[i] = pos-1
            masked_tokens[i] = tokens[pos]
            if random.random() < 0.8:    tokens[pos] = num_alphabets - 1
            elif random.random() < 0.5:  tokens[pos] = random.randint(1, num_alphabets - 1)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights[:n_pred] = True

        return tokens, masked_pos, masked_tokens, masked_weights

    elif cfg.augment > 0:
        for pos in range(1, len(x) + 1):
            if random.random() < cfg.augment: tokens[pos] = random.randint(1, num_alphabets - 1)

    return tokens


def preprocess_seq_for_tfm(x0, x1=None, num_alphabets=21, cfg=None, max_len=512, augment=True):
    """ pre-processing steps for PLUS-TFM pre-training """
    special_tokens = {"MASK": torch.tensor([num_alphabets], dtype=torch.long),
                      "CLS":  torch.tensor([num_alphabets + 1], dtype=torch.long),
                      "SEP":  torch.tensor([num_alphabets + 2], dtype=torch.long)}
    tokens = torch.zeros(max_len, dtype=torch.long)
    segments = torch.zeros(max_len, dtype=torch.long)
    input_mask = torch.zeros(max_len, dtype=torch.bool)

    # -3  for special tokens [CLS], [SEP], [SEP]
    x0, x1 = truncate_seq_pair(x0, x1, max_len)

    # set tokens and segments
    if x1 is not None:
        pair_len = len(x0) + len(x1) + 3
        tokens[:pair_len] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"], x1, special_tokens["SEP"]])
        segments[len(x0) + 2:pair_len] = 1
        input_mask[:pair_len] = True
    else:
        single_len = len(x0) + 2
        tokens[:len(x0) + 2] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"]])
        input_mask[:len(x0) + 2] = True

    if not augment:
        return tokens, segments, input_mask

    elif cfg.mask_ratio > 0:
        max_pred = int(max_len * cfg.mask_ratio)
        masked_pos = torch.zeros(max_pred, dtype=torch.long)
        masked_tokens = torch.zeros(max_pred, dtype=torch.long)
        masked_weights = torch.zeros(max_pred, dtype=torch.bool)

        if x1 is not None:
            # the number of prediction is sometimes less than max_pred when sequence is short
            n_pred = min(max_pred, int(pair_len * cfg.mask_ratio))
            n_pred0 = int(n_pred * (len(x0) / (len(x0) + len(x1))))
            n_pred1 = n_pred - n_pred0
            # candidate positions of masked tokens
            cand_pos0 = [i + 1 for i in range(len(x0)) if tokens[i + 1] != (num_alphabets - 1)]
            cand_pos1 = [i + len(x0) + 2 for i in range(len(x1)) if tokens[i + len(x0) + 2] != (num_alphabets - 1)]
            random.shuffle(cand_pos0); random.shuffle(cand_pos1)
            for i, pos in enumerate(cand_pos0[:n_pred0]):
                masked_pos[i] = pos
                masked_tokens[i] = tokens[pos]
                if random.random() < 0.8:   tokens[pos] = special_tokens["MASK"]
                elif random.random() < 0.5: tokens[pos] = random.randint(1, num_alphabets - 1)
            for i, pos in enumerate(cand_pos1[:n_pred1]):
                masked_pos[i + n_pred0] = pos
                masked_tokens[i + n_pred0] = tokens[pos]
                if random.random() < 0.8:   tokens[pos] = special_tokens["MASK"]
                elif random.random() < 0.5: tokens[pos] = random.randint(1, num_alphabets - 1)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights[:n_pred] = True

        else:
            # the number of prediction is sometimes less than max_pred when sequence is short
            n_pred = min(max_pred, int(single_len * cfg.mask_ratio))
            # candidate positions of masked tokens
            cand_pos = [i + 1 for i in range(len(x0)) if tokens[i + 1] != (num_alphabets - 1)]
            random.shuffle(cand_pos)
            for i, pos in enumerate(cand_pos[:n_pred]):
                masked_pos[i] = pos
                masked_tokens[i] = tokens[pos]
                if random.random() < 0.8:   tokens[pos] = special_tokens["MASK"]
                elif random.random() < 0.5: tokens[pos] = random.randint(1, num_alphabets - 1)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights[:n_pred] = 1

        return tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights

    elif cfg.augment > 0:
        for pos in range(1, len(x0) + 1):
            if random.random() < cfg.augment: tokens[pos] = random.randint(1, num_alphabets - 1)

    return tokens, segments, input_mask


def preprocess_label_for_tfm(y, v, max_len):
    """ pre-processing steps for PLUS-TFM fine-tuning """
    labels = torch.zeros(max_len, dtype=torch.long)
    valids = torch.zeros(1, dtype=torch.uint8)
    weights = torch.zeros(max_len, dtype=torch.bool)

    labels[1:len(y) + 1] = y
    valids[0] = 1 if v else 0
    weights[1:len(y) + 1] = True

    return labels, valids, weights


def truncate_seq_pair(x0, x1, max_len):
    """ clip sequences for the maximum length limitation """
    if x1 is not None:
        max_len -= 3
        while True:
            if len(x0) + len(x1) <= max_len: break
            elif len(x0) > len(x1): x0 = x0[:-1]
            else: x1 = x1[:-1]
    else:
        max_len -= 2
        x0 = x0[:max_len]
    return x0, x1

