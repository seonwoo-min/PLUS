# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# PLUS

import os
import sys
import argparse
import numpy as np

import torch

import plus.config as config
from plus.data.alphabets import Protein
import plus.data.homology as homology
import plus.data.dataset as dataset
import plus.model.plus_rnn as plus_rnn
import plus.model.plus_tfm as plus_tfm
import plus.model.p_elmo as p_elmo
import plus.model.cnn as cnn
from plus.train import Trainer
from plus.utils import Print, set_seeds, set_output, load_models


parser = argparse.ArgumentParser('Train a Model on Homology Datasets')
parser.add_argument('--data-config',  help='path for data configuration file')
parser.add_argument('--model-config', help='path for model configuration file')
parser.add_argument('--lm-model-config', help='path for lm-model configuration file (for P-ELMo)')
parser.add_argument('--run-config', help='path for run configuration file')
parser.add_argument('--pretrained-model', help='path for pretrained model file')
parser.add_argument('--pretrained-lm-model', help='path for pretrained lm-model file (for P-ELMo)')
parser.add_argument('--pretrained-cm-model', help='path for pretrained cm-model file (for P-ELMo and PLUS)')
parser.add_argument('--device', help='device to use; multi-GPU if given multiple GPUs sperated by comma (default: cpu)')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')
parser.add_argument('--output-index', help='index for outputs')
parser.add_argument('--sanity-check', default=False, action='store_true', help='sanity check flag')


def main():
    set_seeds(2020)
    args = vars(parser.parse_args())

    alphabet = Protein()
    cfgs = []
    data_cfg = config.DataConfig(args["data_config"]);   cfgs.append(data_cfg)
    if args["lm_model_config"] is None:
        model_cfg = config.ModelConfig(args["model_config"], input_dim=len(alphabet), num_classes=5)
        cfgs += [model_cfg]
    else:
        lm_model_cfg = config.ModelConfig(args["lm_model_config"], idx="lm_model_config", input_dim=len(alphabet))
        model_cfg = config.ModelConfig(args["model_config"], input_dim=len(alphabet),
                                       lm_dim=lm_model_cfg.num_layers * lm_model_cfg.hidden_dim * 2, num_classes=5)
        cfgs += [model_cfg, lm_model_cfg]
    run_cfg = config.RunConfig(args["run_config"], sanity_check=args["sanity_check"]);  cfgs.append(run_cfg)
    output, save_prefix = set_output(args, "train_homology_log")
    os.environ['CUDA_VISIBLE_DEVICES'] = args["device"] if args["device"] is not None else ""
    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1
    config.print_configs(args, cfgs, device, output)
    flag_rnn = (model_cfg.model_type == "RNN")
    flag_lm_model = (args["lm_model_config"] is not None)
    flag_lm_loss = (run_cfg.lm_loss_lambda != -1)
    flag_cm_loss = (run_cfg.cm_loss_lambda != -1)

    ## load a train dataset
    start = Print(" ".join(['start loading a train dataset:', data_cfg.path["train"]]), output)
    dataset_train = homology.load_homology(data_cfg, "train", alphabet, flag_cm_loss, args["sanity_check"])
    dataset_train = dataset.Homology_dataset(*dataset_train, alphabet, run_cfg, flag_rnn, model_cfg.max_len)
    sampler = dataset.HomolgySampler(dataset_train.labels, run_cfg)
    collate_fn = dataset.collate_paired_sequences if flag_rnn else None
    iterator_train = torch.utils.data.DataLoader(dataset_train, run_cfg.batch_size_train, collate_fn=collate_fn, sampler=sampler)
    end = Print(" ".join(['loaded', str(int(np.sqrt(len(dataset_train)))), 'sequences']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## load a dev dataset
    start = Print(" ".join(['start loading a dev dataset:', data_cfg.path["devpairs"]]), output)
    dataset_test = homology.load_homology_pairs(data_cfg, "devpairs", alphabet, flag_cm_loss, args["sanity_check"])
    dataset_test = dataset.PairedHomology_dataset(*dataset_test, alphabet, run_cfg, flag_rnn, model_cfg.max_len)
    iterator_test = torch.utils.data.DataLoader(dataset_test, run_cfg.batch_size_eval, collate_fn=collate_fn)
    end = Print(" ".join(['loaded', str(len(dataset_test)), 'sequence pairs']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## initialize a model
    start = Print('start initializing a model', output)
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    ### model
    if not flag_rnn:                model = plus_tfm.PLUS_TFM(model_cfg)
    elif not flag_lm_model:         model = plus_rnn.PLUS_RNN(model_cfg)
    else:                           model = p_elmo.P_ELMo(model_cfg)
    models_list.append([model, "", False, flag_rnn, flag_rnn])
    ### lm_model
    if flag_lm_model:
        lm_model = p_elmo.P_ELMo_lm(lm_model_cfg)
        models_list.append([lm_model, "lm", True, False, False])
    ### cm_model
    if flag_cm_loss:
        cm_model = cnn.ConvNet2D(model_cfg.embedding_dim)
        models_list.append([cm_model, "cm", False, False, True])
    params = []
    for model, _, frz, _, _ in models_list:
        if not frz: params += [p for p in model.parameters() if p.requires_grad]
    load_models(args, models_list, device, data_parallel, output, tfm_cls=flag_rnn)
    get_loss = plus_rnn.get_loss if flag_rnn else plus_tfm.get_loss
    end = Print('end initializing a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    optim = torch.optim.Adam(params, lr=run_cfg.learning_rate)
    tasks_list = [] # list of lists [idx, metrics_train, metrics_eval]
    tasks_list.append(["cls", [], ["acc", "r", "rho"]])
    if flag_lm_loss: tasks_list.append(["lm", [], ["acc"]])
    if flag_cm_loss: tasks_list.append(["cm", [], ["pr", "re", "f1"]])
    trainer = Trainer(models_list, get_loss, run_cfg, tasks_list, optim)
    trainer_args = {}
    trainer_args["data_parallel"] = data_parallel
    trainer_args["paired"] = True
    if   flag_rnn: trainer_args["evaluate_cls"] = plus_rnn.evaluate_homology
    else:          trainer_args["evaluate_cls"] = plus_tfm.evaluate_homology
    trainer_args["evaluate"] = ["cls", homology.evaluate_homology]
    end = Print('end setting trainer configurations', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## train a model
    start = Print('start training a model', output)
    Print(trainer.get_headline(), output)
    for epoch in range(run_cfg.num_epochs):
        ### train
        for B, batch in enumerate(iterator_train):
            batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
            trainer.train(batch, trainer_args)
            if B % 10 == 0: print('# epoch [{}/{}] train {:.1%} loss={:.4f}'.format(
                epoch + 1, run_cfg.num_epochs, B / len(iterator_train), trainer.loss_train), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

        ### evaluate cls and cm
        dataset_test.set_augment(False)
        trainer.set_exec_flags(["cls", 'lm', "cm"], [True, False, flag_cm_loss])
        for b, batch in enumerate(iterator_test):
            batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
            trainer.evaluate(batch, trainer_args)
            if b % 10 == 0: print('# cls {:.1%} loss={:.4f}'.format(
                b / len(iterator_test), trainer.loss_eval), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

        ### evaluate lm
        if flag_lm_loss:
            dataset_test.set_augment(True)
            trainer.set_exec_flags(["cls", 'lm', "cm"], [False, True, False])
            for b, batch in enumerate(iterator_test):
                batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
                trainer.evaluate(batch, trainer_args)
                if b % 10 == 0: print('# lm {:.1%} loss={:.4f}'.format(
                    b / len(iterator_test), trainer.loss_eval), end='\r', file=sys.stderr)
            print(' ' * 150, end='\r', file=sys.stderr)

        ### print log and save models
        trainer.save(save_prefix)
        Print(trainer.get_log(epoch + 1, args=trainer_args), output)
        trainer.set_exec_flags(["cls", "lm", "cm"], [True, True, True])
        trainer.reset()
        if trainer.patience == 0: break

    end = Print('end training a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)
    output.close()


if __name__ == '__main__':
    main()