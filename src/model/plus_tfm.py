# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Dong-Hyun Lee's code (https://github.com/dhlee347/pytorchic-bert)
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)

""" PLUS-TFM model classes and functions """

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.model.transformer as tfm
from src.data.transmembrane import is_prediction_correct, Grammar


class PLUS_TFM(nn.Module):
    """ PLUS-TFM model """
    def __init__(self, cfg):
        super(PLUS_TFM, self).__init__()
        self.transformer = tfm.Transformer(cfg)

        # masked language modeling (decoder is shared with embedding layer)
        self.fc_lm = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm_lm = tfm.LayerNorm(cfg)
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        # classification
        if cfg.num_classes is not None:
            self.drop_cls = nn.Dropout(cfg.dropout)
            self.cls = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, tokens, segments, input_mask, masked_pos=None, per_seq=True, embedding=False):
        h = self.transformer(tokens, segments, input_mask)

        if embedding:
            return h
        else:
            logits_lm = None
            if masked_pos is not None:
                masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
                h_masked = torch.gather(h, 1, masked_pos)
                h_masked = self.norm_lm(tfm.gelu(self.fc_lm(h_masked)))
                logits_lm = F.log_softmax(self.decoder(h_masked) + self.decoder_bias, dim=2)

            if per_seq: logits_cls = self.cls(self.drop_cls(h[:, 0]))
            else:       logits_cls = self.cls(self.drop_cls(h))

            return logits_lm, logits_cls

    def load_weights(self, pretrained_model, cls=True):
        # load pre-trained model weights
        state_dict = self.state_dict()
        for key, value in torch.load(pretrained_model, map_location=torch.device('cpu')).items():
            if key.startswith("module"): key = key[7:]
            if (cls and state_dict[key].shape == value.shape) or (not cls and "cls" not in key and state_dict[key].shape == value.shape): state_dict[key] = value
        self.load_state_dict(state_dict)

    def em(self, h, input_mask, cpu=False):
        # get representations with different lengths from the collated single matrix
        e = [None] * len(input_mask)
        for i in range(len(input_mask)):
            if cpu: e[i] = h[i, 1:torch.sum(input_mask[i]) - 1].cpu()
            else:   e[i] = h[i, 1:torch.sum(input_mask[i]) - 1]
        return e


def get_loss(batch, models_dict, cfg, tasks_dict, args, test=False):
    """ feed-forward and evaluate PLUS_TFM model """
    models, models_idx, tasks_idx = models_dict["model"], models_dict["idx"], tasks_dict["idx"]
    if not test: tasks_flag = tasks_dict["flags_train"]
    else:        tasks_flag = tasks_dict["flags_eval"]
    if args["paired"]:
        per_seq = True
        if   len(batch) == 4:
            tokens, segments, input_mask, labels = batch
            masked_pos = None
        elif len(batch) == 6:
            tokens, segments, input_mask, labels, cmaps0, cmaps1 = batch
            masked_pos = None
        elif len(batch) == 7:
            tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights, labels = batch
            per_seq = True
        elif len(batch) == 9:
            tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights, labels, cmaps0, cmaps1 = batch
    else:
        if   len(batch) == 4:
            tokens, segments, input_mask, labels = batch
            masked_pos,  per_seq = None, True
        elif len(batch) == 6:
            tokens, segments, input_mask, labels, valids, label_weights = batch
            masked_pos, per_seq = None, False
        elif len(batch) == 7:
            tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights, labels = batch
            per_seq = True
        elif len(batch) == 9:
            tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights, labels, valids, label_weights = batch
            per_seq = False
    logits_lm, logits_cls = models[models_idx.index("")](tokens, segments, input_mask, masked_pos, per_seq)

    results = []
    for task_idx, flag in zip(tasks_idx, tasks_flag):
        if task_idx not in ["lm", "cls"]: sys.exit("# ERROR: invalid task index [%s]; Supported indices are [lm, cls]" % task_idx)

        result = {"n": 0, "avg_loss": 0}
        if task_idx == "lm" and flag["exec"]:
            result = evaluate_lm(logits_lm, masked_tokens, masked_weights, flag)
            if cfg.lm_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.lm_loss_lambda

        elif task_idx == "cls" and flag["exec"]:
            if per_seq: result = args["evaluate_cls"](logits_cls, labels, flag, args)
            else:       result = args["evaluate_cls"](logits_cls, labels, label_weights, flag)
            if cfg.cls_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.cls_loss_lambda
            if "aggregate" in args:
                result["valid"] = [valids.cpu()]
                result["label_weights"] = [label_weights.cpu()]

        results.append(result)

    return results


def get_embedding(batch, models_dict, args):
    """ feed-forward and evaluate PLUS_TFM model """
    models, models_idx = models_dict["model"], models_dict["idx"]
    tokens, segments, input_mask = batch

    model = models[models_idx.index("")]
    h = model(tokens, segments, input_mask, embedding=True)

    h_list = model.module.em(h, input_mask, cpu=True) if args["data_parallel"] else model.em(h, input_mask, cpu=True)
    embeddings = [[], h_list]

    return embeddings


def evaluate_lm(logits_lm, masked_tokens, masked_weights, flag):
    """ evaluate (masked) language modeling """
    result = {}

    result["n"] = torch.sum(masked_weights).item()
    loss_lm = -logits_lm.gather(2, masked_tokens.unsqueeze(2)).squeeze()
    result["avg_loss"] = torch.mean(loss_lm.masked_select(masked_weights))

    if flag["acc"]:
        _, masked_tokens_hat = torch.max(logits_lm, 2)
        result["correct"] = torch.sum((masked_tokens_hat == masked_tokens).masked_select(masked_weights)).item()

    return result


def evaluate_sfp(logits_cls, labels, flag):
    """ evaluate same family prediction """
    result = {}

    result["n"] = len(logits_cls)
    is_samefamily = labels.squeeze(1).long()
    result["avg_loss"] = nn.CrossEntropyLoss()(logits_cls, is_samefamily)

    if flag["acc"]:
        _, samefamily_hat = torch.max(logits_cls, 1)
        result["correct"] = torch.sum((samefamily_hat == is_samefamily)).item()

    return result


def evaluate_homology(logits_cls, labels, flag, args):
    """ evaluate protein pair structural similarity classification """
    result = {}

    result["n"] = len(logits_cls)
    similarity = torch.sum(labels, 1)
    result["avg_loss"] = nn.CrossEntropyLoss()(logits_cls, similarity)

    if flag["acc"] or flag["pred"]:
        similarity_levels_hat = torch.nn.Softmax(dim=1)(logits_cls)

        if flag["acc"]:
            _, similarity_hat = torch.max(similarity_levels_hat, 1)
            result["correct"] = torch.sum((similarity_hat == similarity)).item()
        if flag["pred"]:
            result["logits"] = [similarity_levels_hat.cpu()]
            result["labels"] = [similarity.cpu()]

    return result


def evaluate_cls_protein(logits_cls, labels, flag, args):
    """ evaluate protein-level classification task """
    result = {}
    labels = labels[:, 0]

    result["n"] = len(logits_cls)
    if "regression" in args and args["regression"]:
        result["avg_loss"] = F.mse_loss(logits_cls, labels)
    else:
        result["avg_loss"] = F.cross_entropy(logits_cls, labels)

    if flag["acc"]:
        _, labels_hat = torch.max(logits_cls, 1)
        result["correct"] = torch.sum((labels_hat == labels)).item()
    if flag["pred"]:
        result["logits"] = [logits_cls.cpu()]
        result["labels"] = [labels.cpu()]

    return result


def evaluate_cls_amino(logits_cls, labels, label_weights, flag):
    """ evaluate amino-acid-level classification task """
    result = {}

    result["n"] = torch.sum(label_weights).item()
    loss_cls = F.cross_entropy(logits_cls.transpose(1, 2), labels, reduction="none")
    result["avg_loss"] = torch.mean(loss_cls.masked_select(label_weights))

    if flag["acc"]:
        _, labels_hat = torch.max(logits_cls, 2)
        result["correct"] = torch.sum((labels_hat == labels).masked_select(label_weights)).item()
    if flag["pred"]:
        result["logits"] = [logits_cls.cpu()]
        result["labels"] = [labels.cpu()]
        result["label_weights"] = [label_weights.cpu()]

    return result


def evaluate_transmembrane(result):
    """ evaluate transmembrane classification task """
    if "logits" not in result: return
    logits_cls, labels, label_weights = result["logits"], result["labels"], result["label_weights"]
    grammar = Grammar()

    n = len(logits_cls)
    log_p_hat = F.log_softmax(logits_cls, 2).detach().cpu().numpy()
    correct = 0
    for i in range(len(log_p_hat)):
        num = torch.sum(label_weights[i])
        label_hat, _ = grammar.decode(log_p_hat[i][:num])
        correct += is_prediction_correct(label_hat, labels[i][:num])

    result["acc_p"] = float(correct) / float(n)


def aggregate_transmembrane(result):
    """ aggreate amino-acid-level transmembrane predictions for protein-level evaluation """
    logits, labels, label_weights = [], [], []
    logit, label, label_weight = [], [], []
    for logits_mb, valids_mb, labels_mb, label_weights_mb in zip(result["logits"], result["valids"], result["labels"], result["label_weights"]):
        for i in range(len(logits_mb)):
            logit.append(logits_mb[i:i+1, 1:-1]); label.append(labels_mb[i:i+1, 1:-1]); label_weight.append(label_weights_mb[i:i+1, 1:-1])

            if valids_mb[i] == 1:
                if len(logits) == 1: logit = logit[0]; label = label[0]; label_weight = label_weight[0]
                else: logit = torch.cat(logit, 1); label = torch.cat(label, 1); label_weight = torch.cat(label_weight, 1)
                logits.append(logit); labels.append(label); label_weights.append(label_weight)
                logit, label, label_weight = [], [], []

    result["logits"] = torch.cat(logits, 0)
    result["labels"] = torch.cat(labels, 0)
    result["label_weights"] = torch.cat(label_weights, 0)

