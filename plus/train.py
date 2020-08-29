# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

""" Training classes and functions """

import sys
import numpy as np

import torch

from plus.utils import evaluate_result


class Trainer():
    """ training helper class """
    def __init__(self, models_list, get_loss, cfg, tasks_list, optim=None, dev_available=True):
        # initialize model information
        self.models_dict = {"model":[], "idx":[], "frz":[], "clip_grad":[], "clip_weight":[]}
        for model, idx, frz, clip_grad, clip_weight in models_list:
            self.models_dict["model"].append(model)
            self.models_dict["idx"].append(idx)
            self.models_dict["frz"].append(frz)
            self.models_dict["clip_grad"].append(clip_grad)
            self.models_dict["clip_weight"].append(clip_weight)
        self.get_loss = get_loss
        self.cfg = cfg
        self.optim = optim
        self.dev_available = dev_available

        # initialize task information
        self.tasks_dict = {"idx":[], "metrics_train":[], "metrics_eval":[],
                           "flags_train":[], "flags_eval":[], "results_train":[], "results_eval":[]}
        for idx, metrics_train, metrics_eval in tasks_list:
            self.tasks_dict["idx"].append(idx)
            self.tasks_dict["metrics_train"].append(metrics_train)
            self.tasks_dict["metrics_eval"].append(metrics_eval)
            self.tasks_dict["flags_train"].append({})
            self.tasks_dict["flags_eval"].append({})
            self.tasks_dict["results_train"].append({})
            self.tasks_dict["results_eval"].append({})
        for t in range(len(self.tasks_dict["idx"])): self.init_flag_result(t)

        # initialize logging parameters
        self.global_step = 0
        self.patience = self.cfg.patience
        self.loss_train = 0
        self.loss_eval = 0
        self.loss_best = 1000

    def train(self, batch, args={}):
        # train models
        self.global_step += 1
        for model in self.models_dict["model"]: model.train()

        results_train = self.get_loss(batch, self.models_dict, self.cfg, self.tasks_dict, args, test=False)

        loss_train, loss = 0, 0
        for t, result_train in enumerate(results_train):
            if self.tasks_dict["flags_train"][t]["exec"]:
                loss += result_train["avg_loss"]
                self.tasks_dict["results_train"][t]["loss"] += result_train["avg_loss"].item() * result_train["n"]
                for k in result_train.keys():
                    if k in ["n", "correct", "tn", "fp", "fn", "tp", "logits", "labels", "label_weights", "valids"]:
                        if k not in self.tasks_dict["results_train"][t]:
                            self.tasks_dict["results_train"][t][k]  = result_train[k]
                        else:
                            self.tasks_dict["results_train"][t][k] += result_train[k]
                    elif k != "avg_loss":
                        self.tasks_dict["results_train"][t][k] = result_train[k]
            loss_train += self.tasks_dict["results_train"][t]["loss"] / self.tasks_dict["results_train"][t]["n"]
        self.loss_train = loss_train

        loss.backward()
        for model, clip_grad in zip(self.models_dict["model"], self.models_dict["clip_grad"]):
            if clip_grad: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        self.optim.step()
        self.optim.zero_grad()
        for model, clip_weight in zip(self.models_dict["model"], self.models_dict["clip_weight"]):
            if clip_weight and args["data_parallel"]: model.module.clip()
            elif clip_weight: model.clip()

    def evaluate(self, batch, args={}):
        # evaluate models
        for model in self.models_dict["model"]: model.eval()

        with torch.no_grad():
            results_eval = self.get_loss(batch, self.models_dict, self.cfg, self.tasks_dict, args, test=True)

        loss_eval = 0
        for t, result_eval in enumerate(results_eval):
            if self.tasks_dict["flags_eval"][t]["exec"]:
                self.tasks_dict["results_eval"][t]["loss"] += result_eval["avg_loss"].item() * result_eval["n"]
                for k in result_eval.keys():
                    if k in ["n", "correct", "tn", "fp", "fn", "tp", "logits", "labels", "label_weights", "valids"]:
                        if k not in self.tasks_dict["results_eval"][t]:
                            self.tasks_dict["results_eval"][t][k]  = result_eval[k]
                        else:
                            self.tasks_dict["results_eval"][t][k] += result_eval[k]
                    elif k != "avg_loss":
                        self.tasks_dict["results_eval"][t][k] = result_eval[k]
            loss_eval += self.tasks_dict["results_eval"][t]["loss"] / self.tasks_dict["results_eval"][t]["n"]
        self.loss_eval = loss_eval

    def embed(self, batch, args={}):
        # embed protein sequences
        for model in self.models_dict["model"]: model.eval()

        with torch.no_grad():
            embeddings = self.get_loss(batch, self.models_dict, args)
            self.tasks_dict["results_eval"][0]["embeddings"][0] += embeddings[0]
            self.tasks_dict["results_eval"][0]["embeddings"][1] += embeddings[1]

    def save(self, save_prefix, args={}):
        # save current and best models
        if self.dev_available and self.loss_best > self.loss_eval:
            best = True
            self.loss_best = self.loss_eval
        elif not self.dev_available and self.loss_best > self.loss_train:
            best = True
            self.loss_best = self.loss_eval
        else:
            best = False

        if best: self.patience = self.cfg.patience
        else:    self.patience -= 1

        if save_prefix is None: return
        for model, idx, frz in zip(self.models_dict["model"], self.models_dict["idx"], self.models_dict["frz"]):
            if frz: continue
            if idx != "": idx += "_"
            torch.save(model.state_dict(), save_prefix + idx + 'current.pt')
            if best: torch.save(model.state_dict(), save_prefix + idx + 'best.pt')

    def load(self, save_prefix, args={}):
        # load best models
        for model, idx, frz in zip(self.models_dict["model"], self.models_dict["idx"], self.models_dict["frz"]):
            if frz: continue
            if idx != "": idx += "_"
            if args["data_parallel"]: model.module.load_weights(save_prefix + idx + 'best.pt')
            else: model.load_weights(save_prefix + idx + 'best.pt')

    def save_embeddings(self, save_prefix, args={}):
        # save embeddings
        if save_prefix is None: return
        else:
            embeddings = self.tasks_dict["results_eval"][0]["embeddings"]
            for i in range(len(embeddings[0])): np.save(save_prefix + "z%d.npy" % i, embeddings[0][i].numpy())
            for i in range(len(embeddings[1])): np.save(save_prefix + "h%d.npy" % i, embeddings[1][i].numpy())

    def init_flag_result(self, t):
        # initialize training/development dictionaries and flags of a task [t]
        for i in range(2):
            if i == 0: metrics = self.tasks_dict["metrics_train"][t]
            else:      metrics = self.tasks_dict["metrics_eval"][t]

            flag = {"exec":True, "acc":False, "conf":False, "pred":False}
            result = {"loss": 0, "n": np.finfo(float).eps, "embeddings":[[], []]}
            for metric in metrics:
                if metric in ["acc"]:
                    flag["acc"] = True
                    result["correct"] = 0
                if metric in ["pr", "re", "sp", "f1", "mcc"]:
                    flag["conf"] = True
                    result["tn"], result["fp"], result["fn"], result["tp"] = 0, 0, 0, 0
                else:
                    flag["pred"] = True
                    result["logits"], result["labels"] = [], []

            if i == 0: self.tasks_dict["flags_train"][t] = flag; self.tasks_dict["results_train"][t] = result
            else:      self.tasks_dict["flags_eval" ][t] = flag; self.tasks_dict["results_eval" ][t] = result

    def set_exec_flags(self, idxs, execs):
        # set training/development execution flags of a task [idx]
        for idx, exec in zip(idxs, execs):
            if idx not in self.tasks_dict["idx"]: continue
            t = self.tasks_dict["idx"].index(idx)
            self.tasks_dict["flags_train"][t]["exec"] = exec
            self.tasks_dict["flags_eval" ][t]["exec"] = exec

    def reset(self):
        # reset training/development dictionaries and flags
        for t in range(len(self.tasks_dict["idx"])): self.init_flag_result(t)

    def get_headline(self, test=False):
        # get a headline for logging
        if not test:
            headline_train = ["ep", "step(k)", "split", "loss"]
            headline_eval  = ['|', 'split', 'loss']
            if len(self.tasks_dict["idx"]) > 1:
                for idx, metrics_train, metrics_eval in zip(self.tasks_dict["idx"], self.tasks_dict["metrics_train"], self.tasks_dict["metrics_eval"]):
                    headline_train.append(idx + "_loss")
                    headline_eval.append(idx + "_loss")
                    for metric in metrics_train: headline_train.append(idx + "_" + metric)
                    for metric in metrics_eval:  headline_eval.append(idx + "_" + metric)
            else:
                headline_train += self.tasks_dict["metrics_train"][0]
                headline_eval  += self.tasks_dict["metrics_eval"][0]
            headline = headline_train + headline_eval
        else:
            headline_eval  = ['split', 'loss']
            if len(self.tasks_dict["idx"]) > 1:
                for idx, metrics_eval in zip(self.tasks_dict["idx"], self.tasks_dict["metrics_eval"]):
                    headline_eval.append(idx + "_loss")
                    for metric in metrics_eval:  headline_eval.append(idx + "_" + metric)
            else:
                headline_eval  += self.tasks_dict["metrics_eval"][0]
            headline = headline_eval

        return '\t'.join(headline)

    def get_log(self, ep=0, test_idx=None, args={}):
        # get current results for logging

        # aggregate results if necessary
        if "aggregate" not in args: self.aggregate_results()
        elif isinstance(args["aggregate"], list):
            t = self.tasks_dict["idx"].index(args["aggregate"][0])
            args["aggregate"][1](self.tasks_dict["results_train"][t])
            args["aggregate"][1](self.tasks_dict["results_eval" ][t])

        # evaluate results if necessary
        if "evaluate" in args:
            t = self.tasks_dict["idx"].index(args["evaluate"][0])
            args["evaluate"][1](self.tasks_dict["results_train"][t])
            args["evaluate"][1](self.tasks_dict["results_eval" ][t])

        if test_idx is None:
            log_train = [str(ep).zfill(int(np.floor(np.log10(self.cfg.num_epochs))) + 1), str(self.global_step // 1000),
                         "train", "{:.4f}".format(self.loss_train)]
            log_eval  = ["|", "dev", "{:.4f}".format(self.loss_eval)]
            if len(self.tasks_dict["idx"]) > 1:
                for t in range(len(self.tasks_dict["idx"])):
                    log_train.append("{:.4f}".format(self.tasks_dict["results_train"][t]["loss"] / self.tasks_dict["results_train"][t]["n"]))
                    log_eval.append( "{:.4f}".format(self.tasks_dict["results_eval" ][t]["loss"] / self.tasks_dict["results_eval" ][t]["n"]))
                    log_train += self.get_results(t, test=False)
                    log_eval  += self.get_results(t, test=True)
            else:
                log_train += self.get_results(0, test=False)
                log_eval  += self.get_results(0, test=True)
            log = log_train + log_eval

        else:
            log_eval  = [test_idx, "{:.4f}".format(self.loss_eval)]
            if len(self.tasks_dict["idx"]) > 1:
                for t in range(len(self.tasks_dict["idx"])):
                    log_eval.append( "{:.4f}".format(self.tasks_dict["results_eval" ][t]["loss"] / self.tasks_dict["results_eval" ][t]["n"]))
                    log_eval  += self.get_results(t, test=True)
            else:
                log_eval  += self.get_results(0, test=True)
            log = log_eval

        self.loss_train = 0
        self.loss_eval = 0

        return '\t'.join(log)

    def aggregate_results(self):
        for result_train, result_eval in zip(self.tasks_dict["results_train"], self.tasks_dict["results_eval"]):
            for idx in ["logits", "labels", "label_weights"]:
                if idx in result_train and len(result_train[idx]) > 0:
                    if len(result_train[idx]) == 1: result_train[idx] = result_train[idx][0]
                    #elif len(result_train[idx][0].shape) != 2: result_train[idx]  = torch.cat(result_train[idx])
                    else:
                        concat = True
                        shape = result_train[idx][0].shape[1:]
                        for i in range(1, len(result_train[idx])):
                            if result_train[idx][i].shape[1:] != shape: concat = False; break
                        if concat: result_train[idx]  = torch.cat(result_train[idx])
                if idx in result_eval and len(result_eval[idx]) > 0:
                    if len(result_eval[idx]) == 1: result_eval[idx] = result_eval[idx][0]
                    #elif len(result_eval[idx][0].shape) != 2: result_eval[idx] = torch.cat(result_eval[idx])
                    else:
                        concat = True
                        shape = result_eval[idx][0].shape[1:]
                        for i in range(1, len(result_eval[idx])):
                            if result_eval[idx][i].shape[1:] != shape: concat = False; break
                        if concat: result_eval[idx] = torch.cat(result_eval[idx])

    def get_results(self, t, test=False):
        # get results according to the given metrics
        log = []
        if not test: results, metrics = self.tasks_dict["results_train"][t], self.tasks_dict["metrics_train"][t]
        else:        results, metrics = self.tasks_dict["results_eval" ][t], self.tasks_dict["metrics_eval" ][t]

        for metric in metrics:
            # if metric was pre-computed
            if metric in results:
                log.append("{:.4f}".format(results[metric]))
            # otherwise compute the give metric
            elif metric in ["acc", "pr", "re", "sp", "f1", "mcc", "auc", "aupr", "r", "rho"]:
                log.append("{:.4f}".format(evaluate_result(results, metric)))
            else:
                sys.exit("# ERROR: metric [%s] is not supported]" % metric)

        return log

