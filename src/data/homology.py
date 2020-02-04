# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)
# PLUS

""" Homology FASTA file loading functions """

import os
import glob
import numpy as np
import pandas as pd
import PIL.Image as Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

import torch

from src.data.fasta import parse_stream


def load_homology(cfg, idx, encoder, contact_map=False, sanity_check=False):
    """ load Homology sequence data from FASTA file """
    if contact_map: cmap_dict = {os.path.basename(path)[:7]: path for path in glob.glob(cfg.path["2.06cmap"])}
    with open(cfg.path[idx], 'rb') as f:
        sequences, structs, cmaps = [], [], []
        for name, sequence in parse_stream(f):
            # input sequence length configurations
            sequence = encoder.encode(sequence.upper())
            struct = encode_homology_struct(name)

            if cfg.min_len != -1 and len(sequence) < cfg.min_len: continue
            if cfg.max_len != -1 and len(sequence) > cfg.max_len: continue
            if cfg.truncate != -1 and len(sequence) > cfg.truncate: sequence = sequence[:cfg.truncate]
            if sanity_check and len(sequences) == 500: break

            if contact_map:
                cmap_name = name.split()[0].decode('utf-8')
                if cmap_name not in cmap_dict: cmap_name = 'd' + cmap_name[1:]
                im = np.array(Image.open(cmap_dict[cmap_name]), copy=False)
                cmap = np.zeros(im.shape, dtype=np.float32)
                cmap[im == 1] = -1; cmap[im == 255] = 1; cmap[np.tril_indices(cmap.shape[0], k=1)] = -1
                cmaps.append(cmap)

            sequences.append(sequence); structs.append(struct)

    sequences = [torch.from_numpy(sequence).long() for sequence in sequences]
    structs = torch.from_numpy(np.stack(structs, 0))
    similarity_levels = torch.cumprod((structs.unsqueeze(1) == structs.unsqueeze(0)).long(), 2)
    cmaps = [torch.from_numpy(cmap) for cmap in cmaps] if contact_map else None

    return sequences, similarity_levels, cmaps


def load_homology_pairs(cfg, idx, encoder, contact_map=False, sanity_check=False):
    """ load Homology pairs data from csv file """
    if contact_map:
        cmap_dict = {}
        for path in glob.glob(cfg.path["2.06cmap"]): cmap_dict[os.path.basename(path)[:7]] = path
        for path in glob.glob(cfg.path["2.07cmap"]): cmap_dict[os.path.basename(path)[:7]] = path
    table = pd.read_csv(cfg.path[idx], sep='\t')
    sequences0_all = [encoder.encode(sequence.encode('utf-8').upper()) for sequence in table['sequence_A']]
    sequences1_all = [encoder.encode(sequence.encode('utf-8').upper()) for sequence in table['sequence_B']]
    names0_all = table['pdb_id_A'].values
    names1_all = table['pdb_id_B'].values
    labels_all = table['similarity'].values

    # input sequence length configurations
    sequences0, sequences1, names0, names1, labels = [], [], [], [], []
    for i, (sequence0, sequence1, name0, name1) in enumerate(zip(sequences0_all, sequences1_all, names0_all, names1_all)):
        if cfg.min_len != -1 and len(sequence0) <  cfg.min_len: continue
        if cfg.max_len != -1 and len(sequence0) >  cfg.max_len: continue
        if cfg.min_len != -1 and len(sequence1) <  cfg.min_len: continue
        if cfg.max_len != -1 and len(sequence1) >  cfg.max_len: continue
        if cfg.pair_min_len != -1 and len(sequence0) + len(sequence1) < cfg.pair_min_len: continue
        if cfg.pair_max_len != -1 and len(sequence0) + len(sequence1) > cfg.pair_max_len: continue
        if sanity_check and len(sequences0) == 100: break

        sequences0.append(sequence0.astype(np.long))
        sequences1.append(sequence1.astype(np.long))
        names0.append(name0)
        names1.append(name1)
        labels.append(labels_all[i].astype(np.long))

    sequences0 = [torch.from_numpy(sequence0) for sequence0 in sequences0]
    sequences1 = [torch.from_numpy(sequence1) for sequence1 in sequences1]
    similarity = torch.from_numpy(np.stack(labels)).long()
    similarity_levels = torch.zeros(similarity.size(0), 4, dtype=torch.long)
    for i in range(similarity_levels.size(0)): similarity_levels[i, :similarity[i]] = 1

    if contact_map:
        cmaps0, cmaps1 = [], []
        for name0, name1 in zip(names0, names1):
            if name0 not in cmap_dict: name0 = 'd' + name0[1:]
            if name1 not in cmap_dict: name1 = 'd' + name1[1:]
            im0, im1 = np.array(Image.open(cmap_dict[name0]), copy=False), np.array(Image.open(cmap_dict[name1]), copy=False)
            cmap0, cmap1 = np.zeros(im0.shape, dtype=np.float32),  np.zeros(im1.shape, dtype=np.float32)
            cmap0[im0 == 1] = -1; cmap0[im0 == 255] = 1; cmap0[np.tril_indices(cmap0.shape[0], k=1)] = -1
            cmap1[im1 == 1] = -1; cmap1[im1 == 255] = 1; cmap1[np.tril_indices(cmap1.shape[0], k=1)] = -1
            cmaps0.append(torch.from_numpy(cmap0))
            cmaps1.append(torch.from_numpy(cmap1))
    else: cmaps0, cmaps1 = None, None

    return sequences0, sequences1, similarity_levels, cmaps0, cmaps1


def encode_homology_struct(name):
    """ encode structure levels as integer by right-padding with zero byte to 4 bytes """
    tokens = name.split()
    struct = b''
    for s in tokens[1].split(b'.'):
        n = len(s)
        s = s + b'\x00' * (4 - n)
        struct += s
    struct = np.frombuffer(struct, dtype=np.int32)
    return struct


def evaluate_homology(result):
    """ evaluate homology based on ordinal classification results (corr / aupr) """
    if "logits" not in result: return
    logits, labels = result["logits"], result["labels"]

    similarity_levels_hat = logits
    similarity_scores = torch.sum(similarity_levels_hat * torch.arange(5).float(), 1).numpy()
    similarity = labels.numpy()

    pred_level = np.digitize(similarity_scores, find_best_thresholds(similarity_scores, similarity)[1:], right=True)
    result["correct"] = torch.tensor(np.sum(pred_level == similarity)).item()
    isinf, isnan = np.isinf(similarity_scores), np.isnan(similarity_scores)
    if np.any(isinf) or np.any(isnan):
        select = (~ isinf) * (~ isnan)
        similarity_scores = similarity_scores[select]
        similarity = similarity[select]

    if len(similarity_scores) > 1:
        result["r"]   = pearsonr( similarity_scores, similarity)[0]
        result["rho"] = spearmanr(similarity_scores, similarity)[0]
        aupr = []
        for i in range(4):
            target = (similarity > i).astype(np.float32)
            aupr.append(average_precision_score(target, similarity_scores.astype(np.float32)))
        result["aupr_cl"] = aupr[0]
        result["aupr_fo"] = aupr[1]
        result["aupr_sf"] = aupr[2]
        result["aupr_fa"] = aupr[3]
    else:
        result["r"]   = 0
        result["rho"] = 0
        result["aupr_cl"] = 0
        result["aupr_fo"] = 0
        result["aupr_sf"] = 0
        result["aupr_fa"] = 0

def find_best_threshold(x, y, tr0=-np.inf):
    """ FInd threshold of given label for the highest accuracy """
    tp, tn = np.zeros(len(x) + 1), np.zeros(len(x) + 1)
    tp[0], tn[0] = y.sum(), 0
    order = np.argsort(x)

    for i in range(len(x)):
        j = order[i]
        tp[i + 1] = tp[i] - y[j]
        tn[i + 1] = tn[i] + 1 - y[j]

    acc = (tp + tn) / len(y)
    i = np.argmax(acc) - 1
    if i < 0: tr = tr0
    else: tr = x[order[i]]

    return tr


def find_best_thresholds(x, y):
    """ Find thresholds for the highest accuracy """
    thresholds = np.zeros(5)
    thresholds[0] = -np.inf
    for i in range(4):
        mask = (x > thresholds[i])
        xi = x[mask]
        labels = (y[mask] > i).astype(np.int)
        tr = find_best_threshold(xi, labels, tr0=thresholds[i])
        thresholds[i + 1] = tr
    return thresholds
