# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

""" Utility functions """

import os
import sys
import math
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import auc, roc_auc_score
from scipy.stats import pearsonr, spearmanr, t

import torch
import torch.nn as nn


def Print(string, output, newline=False):
    """ print to stdout and a file (if given) """
    time = datetime.now()
    print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_output(args, string, test=False, embedding=False):
    """ set output configurations """
    output, save_prefix, index = sys.stdout, None, ""
    if args["output_path"] is not None:
        if not test and not embedding:
            if not os.path.exists(args["output_path"] + "/weights/"):
                os.makedirs(args["output_path"] + "/weights/", exist_ok=True)
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/weights/" + index

        elif not embedding:
            if not os.path.exists(args["output_path"]):
                os.makedirs(args["output_path"], exist_ok=True)
            if args["pretrained_model"] is not None:
                index += os.path.splitext(args["pretrained_model"])[0].split("/")[-1] + "_"
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")

        else:
            if not os.path.exists(args["output_path"] + "/embeddings/"):
                os.makedirs(args["output_path"] + "/embeddings/", exist_ok=True)
            if args["pretrained_model"] is not None:
                index += os.path.splitext(args["pretrained_model"])[0].split("/")[-1] + "_"
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/embeddings/"

    return output, save_prefix


def load_models(args, models, device, data_parallel, output, tfm_cls=True):
    """ load models if pretrained_models are available """
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_model" if idx == "" else "pretrained_%s_model" % idx
        if idx in args and args[idx] is not None:
            Print('loading %s weights from %s' % (idx, args[idx]), output)
            if not tfm_cls and idx == "pretrained_model": models[m][0].load_weights(args[idx], tfm_cls)
            else:                                         models[m][0].load_weights(args[idx])

        models[m][0] = models[m][0].to(device)
        if data_parallel: models[m][0] = nn.DataParallel(models[m][0])


def evaluate_result(dict, metric):
    """ calculate the evaluation metric from the given results """
    result = None
    if metric   == "acc"  : result = (dict["correct"]) / (dict["n"]+ np.finfo(float).eps)
    elif metric == "pr"   : result = dict["tp"] / (dict["tp"] + dict["fp"]+ np.finfo(float).eps)
    elif metric == "re"   : result = dict["tp"] / (dict["tp"] + dict["fn"]+ np.finfo(float).eps)
    elif metric == "sp"   : result = dict["tn"] / (dict["tn"] + dict["fp"]+ np.finfo(float).eps)
    elif metric == "f1"   : result = 2 * dict["tp"] / (2 * dict["tp"] + dict["fp"] + dict["fn"]+ np.finfo(float).eps)
    elif metric == "mcc"  : result = ((dict["tp"] * dict["tn"] - dict["fp"] * dict["fn"]) /
                                    (math.sqrt((dict["tp"] + dict["fp"]) * (dict["tp"] + dict["fn"]) *
                                               (dict["tn"] + dict["fp"]) * (dict["tn"] + dict["fn"])) + np.finfo(float).eps))

    if result is None:
        if "labels" in dict and len(dict["labels"].shape) == 2: dict["labels"] = dict["labels"][:, 0]
        if "logits" in dict and len(dict["logits"].shape) == 2: dict["logits"] = dict["logits"][:, 0]
    if metric   == "auc"  : result = auc(dict["labels"], dict["logits"])
    elif metric == "aupr" : result = roc_auc_score(dict["labels"], dict["logits"])
    elif metric == "r"    : result = pearsonr(dict["labels"], dict["logits"])[0]
    elif metric == "rho"  : result = spearmanr(dict["labels"], dict["logits"])[0]
    return result


def steiger_test(xy, xz, yz, n):
    """ One-tailed Steiger's test for the statistic significance between two dependent correlation coefficients """
    # ab: correlation coefficient between a and b (xy, xz, yz)
    # n: number of elements in x, y and z
    d = xy - xz
    determin = 1 - xy ** 2 - xz ** 2 - yz ** 2 + 2 * xy * xz * yz
    av = (xy + xz)/2
    cube = (1 - yz) * (1 - yz) * (1 - yz)
    e = (n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + (av ** 2) * cube))
    if e < 0:
        return np.nan, np.nan
    t2 = d * np.sqrt(e)
    p = 1 - t.cdf(abs(t2), n - 2)

    return t2, p