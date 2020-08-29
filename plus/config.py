# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Dong-Hyun Lee's code (https://github.com/dhlee347/pytorchic-bert)
# PLUS functions

""" PLUS config classes and functions """

import os
import sys
import json
import math
from collections import OrderedDict

import torch

from plus.utils import Print


class DataConfig():
    def __init__(self, file=None, idx="data_config"):
        """ data configurations """
        self.idx = idx                  # config index
        self.min_len = -1               # minimum length of protein sequence to use
        self.max_len = -1               # maximum length of protein sequence to use
        self.pair_min_len = -1          # minimum length sum of protein sequence pair to use
        self.pair_max_len = -1          # maximum length sum of protein sequence pair to use
        self.min_seq = -1               # minimum number of protein sequence for a family to use
        self.truncate = -1              # truncate sequences longer than the pre-defined length
        self.path = OrderedDict()       # data file index:key dictionary

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("data-config [%s] does not exists" % file)
            cfg = json.load(open(file, "r"))
            for key, value in cfg.items():
                if   key == "min_len":          self.min_len = value
                elif key == "max_len":          self.max_len = value
                elif key == "min_seq":          self.min_seq = value
                elif key == "pair_min_len":     self.pair_min_len = value
                elif key == "pair_max_len":     self.pair_max_len = value
                elif key == "truncate":         self.truncate = value
                elif "path" in key:             self.path[key.split("_")[0]] = value
                else: sys.exit("# ERROR: invalid key [%s] in data-config file" % key)

    def get_config(self):
        configs = []
        if self.min_len != -1: configs.append("-- min_len: %s" % self.min_len)
        if self.max_len != -1: configs.append("-- max_len: %s" % self.max_len)
        if self.pair_min_len != -1: configs.append("-- pair_min_len: %s" % self.pair_min_len)
        if self.pair_max_len != -1: configs.append("-- pair_max_len: %s" % self.pair_max_len)
        if self.min_seq != -1: configs.append("-- min_seq: %s" % self.min_seq)
        if self.truncate != -1: configs.append("-- truncate: %s" % self.truncate)
        configs.append("-- path: %s" % list(self.path.items()))
        return configs


class ModelConfig():
    def __init__(self, file=None, idx="model_config", model_type="RNN", input_dim=None, lm_dim=None, num_classes=None):
        """ model configurations """
        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("model-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))
            if "model_type" in cfg: self.model_type = cfg["model_type"]; del cfg["model_type"]
            else:                   self.model_type = model_type
        else:                       self.model_type = model_type
        self.idx = idx  # config index
        self.max_len = None  # maximum length of input (for PLUS-TFM)
        self.num_classes = None

        if self.model_type == "RNN":
            self.rnn_type = "B"             # type of rnn
            self.input_dim = input_dim      # size of input tokens
            self.num_layers = 3             # number of RNN layers
            self.hidden_dim = 512           # hidden dimension of RNN layer

            # For embedding
            self.lm_dim = lm_dim            # language modeling dimension
            self.lm_proj_dim = -1           # language modeling projection dimension
            self.embedding_dim = -1         # embedding dimension
            if num_classes is not None: self.num_classes = num_classes - 1 # number of classes
            self.dropout = 0                # dropout

            if file is not None:
                for key, value in cfg.items():
                    if   key == "rnn_type":                     self.rnn_type = value
                    elif key == "num_layers":                   self.num_layers = value
                    elif key == "hidden_dim":                   self.hidden_dim = value
                    elif key == "lm_proj_dim":                  self.lm_proj_dim = value
                    elif key == "embedding_dim":                self.embedding_dim = value
                    else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

        elif self.model_type == "TFM":
            self.input_dim = input_dim + 3  # size of tokens
            self.num_layers = 12            # number of Transformer blocks
            self.hidden_dim = 768           # hidden dimension of Transformer
            self.num_heads = 12             # number of heads in multi-headed attention layer
            self.feedforward_dim = 3072     # dimension of position-wise feedforward layer
            self.num_classes = num_classes  # number of classes
            self.pos_encode = True          # use sinusoidal positional encoding otherwise use embedding
            self.dropout = 0.1              # dropout

            if file is not None:
                for key, value in cfg.items():
                    if   key == "num_layers":       self.num_layers = value
                    elif key == "hidden_dim":       self.hidden_dim = value
                    elif key == "num_heads":        self.num_heads = value
                    elif key == "feedforward_dim":  self.feedforward_dim = value
                    elif key == "pos_encode":       self.pos_encode = value
                    elif key == "max_len":          self.max_len = value
                    elif key == "dropout":          self.dropout = value
                    else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

        else:
            self.input_dim = input_dim      # input dimension
            self.projection = False         # flag for using projected representations
            self.hidden_dim = 100           # hidden dimension

            self.num_classes = num_classes  # number of classes
            self.dropout = 0                # dropout

            if file is not None:
                for key, value in cfg.items():
                    if   key == "projection":                   self.projection = value
                    elif key == "hidden_dim":                   self.hidden_dim = value
                    elif key == "dropout":                      self.dropout = value
                    else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

    def set_input_dim(self, input_dim):
        """ set input dimension of a model """
        self.input_dim = input_dim

    def get_config(self):
        configs = []
        if self.model_type == "RNN":
            configs.append("-- rnn_type: %s" % self.rnn_type)
            configs.append("-- num_layers: %s" % self.num_layers)
            configs.append("-- hidden_dim: %s" % self.hidden_dim)
            if self.lm_proj_dim   != -1: configs.append("-- lm_proj_dim: %s" % self.lm_proj_dim)
            if self.embedding_dim != -1: configs.append("-- embedding_dim: %s" % self.embedding_dim)
        elif self.model_type == "TFM":
            configs.append("-- num_layers: %s" % self.num_layers)
            configs.append("-- hidden_dim: %s" % self.hidden_dim)
            configs.append("-- num_heads: %s" % self.num_heads)
            configs.append("-- feedforward_dim: %s" % self.feedforward_dim)
            configs.append("-- pos_encode: %s" % self.pos_encode)
            configs.append("-- max_len: %s" % self.max_len)
            if self.dropout != 0: configs.append("-- dropout: %s" % self.dropout)
        else:
            configs = []
            configs.append("-- projection: %s" % self.projection)
            configs.append("-- hidden_dim: %s" % self.hidden_dim)
            if self.dropout != 0: configs.append("-- dropout: %s" % self.dropout)
        return configs


class RunConfig():
    def __init__(self, file=None, idx="run_config", eval=False, sanity_check=False):
        """ run (train/eval) configurations """
        self.idx = idx                  # config index
        self.eval = eval                # flag for training / evaluation
        self.batch_size_train = -1      # batch size for training
        self.batch_size_eval = -1       # batch size for evaluation
        self.cm_batch_size = -1         # contact map batch size
        self.num_epochs = -1            # number of epochs
        self.patience = -1              # patience during training
        self.learning_rate = -1         # learning rate
        self.pr_learning_rate = -1      # learning rate for prediction layers
        self.mask_ratio = -1            # mask the ratio of amino acids in each protein sequence at random
        self.max_pred = -1              # maximum masked language modeling prediction for a sequence
        self.lm_loss_lambda = -1        # hyperparameter for modulating language modeling loss
        self.cm_loss_lambda = -1        # hyperparameter for modulating contact map loss
        self.cls_loss_lambda = -1       # hyperparamerte for modulating similarity classification prediction loss
        self.tau = -1                   # sampling proportion exponent for different similarity levels
        self.epoch_size = -1            # number of pair examples per epoch
        self.augment = -1               # probability of resampling amino acid for data augmentation
        self.warm_up = -1               # linearly increasing learning rate from zero to the specified value for warm-up * total_steps
        self.total_steps = None         # total number of steps to train (must be initialized with set_total_step

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("train-config [%s] does not exists" % file)
            cfg = json.load(open(file, "r"))
            for key, value in cfg.items():
                if   key == "batch_size_train": self.batch_size_train = value
                elif key == "batch_size_eval":  self.batch_size_eval = value
                elif key == "cm_batch_size":    self.cm_batch_size = value
                elif key == "num_epochs":       self.num_epochs = value
                elif key == "patience":         self.patience = value
                elif key == "learning_rate":    self.learning_rate = value
                elif key == "pr_learning_rate": self.pr_learning_rate = value
                elif key == "mask_ratio":       self.mask_ratio = value
                elif key == "max_pred":         self.max_pred = value
                elif key == "lm_loss_lambda":   self.lm_loss_lambda = value
                elif key == "cm_loss_lambda":   self.cm_loss_lambda = value
                elif key == "cls_loss_lambda":  self.cls_loss_lambda = value
                elif key == "tau":              self.tau = value
                elif key == "epoch_size":       self.epoch_size = value
                elif key == "augment":          self.augment = value
                elif key == "warm_up":          self.warm_up = value
                else: sys.exit("# ERROR: invalid key [%s] in train-config file" % key)

        if sanity_check:
            self.batch_size_train = 32
            self.num_epochs = 5
            self.epoch_size = 100
            self.cm_batch_size = 2

    def set_total_steps(self, num_data):
        """ total gradient update steps for learning rate scheduling """
        self.total_steps = math.ceil(num_data / self.batch_size_train) * self.num_epochs

    def get_config(self):
        configs = []
        if not self.eval:
            if self.batch_size_train != -1: configs.append("-- batch_size_train: %s" % self.batch_size_train)
            if self.num_epochs != -1: configs.append("-- num_epochs: %s" % self.num_epochs)
            if self.patience != -1: configs.append("-- patience: %s" % self.patience)
            if self.learning_rate != -1: configs.append("-- learning_rate: %s" % self.learning_rate)
            if self.pr_learning_rate != -1: configs.append("-- pr_learning_rate: %s" % self.pr_learning_rate)
            if self.tau != -1: configs.append("-- tau: %s" % self.tau)
            if self.epoch_size != -1: configs.append("-- epoch_size: %s" % self.epoch_size)
            if self.augment != -1: configs.append("-- augment: %s" % self.epoch_size)
            if self.warm_up != -1: configs.append("-- warm_up: %s" % self.warm_up)

        if self.batch_size_eval != -1: configs.append("-- batch_size_eval: %s" % self.batch_size_eval)
        if self.cm_batch_size != -1: configs.append("-- cm_batch_size: %s" % self.cm_batch_size)
        if self.mask_ratio != -1: configs.append("-- mask_ratio: %s" % self.mask_ratio)
        if self.max_pred != -1: configs.append("-- max_pred: %s" % self.max_pred)
        if self.lm_loss_lambda != -1: configs.append("-- lm_loss_lambda: %s" % self.lm_loss_lambda)
        if self.cm_loss_lambda != -1: configs.append("-- cm_loss_lambda: %s" % self.cm_loss_lambda)
        if self.cls_loss_lambda!= -1: configs.append("-- cls_loss_lambda: %s"% self.cls_loss_lambda)

        return configs


def print_configs(args, cfgs, device, output):
    if args["sanity_check"]: Print(" ".join(['##### SANITY_CHECK #####']), output)
    Print(" ".join(['##### arguments #####']), output)

    for cfg in cfgs:
        Print(" ".join(['%s:' % cfg.idx, str(args[cfg.idx])]), output)
        for string in cfg.get_config(): Print(string, output)

    for model in ["pretrained_model", "pretrained_lm_model", "pretrained_cm_model", "pretrained_pr_model"]:
        if not model in args: continue
        elif args[model] is not None and not os.path.exists(args[model]):
            sys.exit("%s [%s] does not exists" % (model, args[model]))
        elif args[model] is not None: Print(" ".join(['%s: %s' % (model, args[model])]), output)

    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['log_file:', str(output.name)]), output, newline=True)
