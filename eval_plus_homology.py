# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# PLUS

import os
import sys
import argparse

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


parser = argparse.ArgumentParser('Evaluate a Model on Homology Dataset')
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
    run_cfg = config.RunConfig(args["run_config"], eval=True, sanity_check=args["sanity_check"]);  cfgs.append(run_cfg)
    output, save_prefix = set_output(args, "eval_homology_log", test=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args["device"] if args["device"] is not None else ""
    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1
    config.print_configs(args, cfgs, device, output)
    flag_rnn = (model_cfg.model_type == "RNN")
    flag_lm_model = (args["pretrained_lm_model"] is not None)
    flag_cm_model = (args["pretrained_cm_model"] is not None)

    ## load test datasets
    idxs_test, datasets_test, iterators_test = [key for key in data_cfg.path.keys() if "pairs" in key], [], []
    start = Print(" ".join(['start loading test datasets'] + idxs_test), output)
    collate_fn = dataset.collate_paired_sequences if flag_rnn else None
    for idx_test in idxs_test:
        dataset_test = homology.load_homology_pairs(data_cfg, idx_test, alphabet, flag_cm_model, args["sanity_check"])
        dataset_test = dataset.PairedHomology_dataset(*dataset_test, alphabet, run_cfg, flag_rnn, model_cfg.max_len)
        iterator_test = torch.utils.data.DataLoader(dataset_test, run_cfg.batch_size_eval, collate_fn=collate_fn)
        datasets_test.append(dataset_test); iterators_test.append(iterator_test)
        end = Print(" ".join(['loaded', str(len(dataset_test)), 'sequence pairs']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## initialize a model
    start = Print('start initializing a model', output)
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    ### model
    if not flag_rnn:                model = plus_tfm.PLUS_TFM(model_cfg)
    elif not flag_lm_model:         model = plus_rnn.PLUS_RNN(model_cfg)
    else:                           model = p_elmo.P_ELMo(model_cfg)
    models_list.append([model, "", True, False, False])
    ### lm_model
    if flag_lm_model:
        lm_model = p_elmo.P_ELMo_lm(lm_model_cfg)
        models_list.append([lm_model, "lm", True, False, False])
    ### cm_model
    if flag_cm_model:
        cm_model = cnn.ConvNet2D(model_cfg.embedding_dim)
        models_list.append([cm_model, "cm", True, False, False])
    load_models(args, models_list, device, data_parallel, output)
    get_loss = plus_rnn.get_loss if flag_rnn else plus_tfm.get_loss
    end = Print('end initializing a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    tasks_list = [] # list of lists [idx, metrics_train, metrics_eval]
    tasks_list.append(["cls", [], ["acc", "r", "rho", "aupr_cl", "aupr_fo", "aupr_sf", "aupr_fa"]])
    if not flag_lm_model: tasks_list.append(["lm", [], ["acc"]])
    if flag_cm_model: tasks_list.append(["cm", [], ["pr", "re", "f1"]])
    trainer = Trainer(models_list, get_loss, run_cfg, tasks_list)
    trainer_args = {}
    trainer_args["data_parallel"] = data_parallel
    trainer_args["paired"] = True
    if   flag_rnn: trainer_args["evaluate_cls"] = plus_rnn.evaluate_homology
    else:          trainer_args["evaluate_cls"] = plus_tfm.evaluate_homology
    trainer_args["evaluate"] = ["cls", homology.evaluate_homology]
    end = Print('end setting trainer configurations', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## evaluate a model
    start = Print('start evaluating a model', output)
    Print(trainer.get_headline(test=True), output)
    for idx_test, dataset_test, iterator_test in zip(idxs_test, datasets_test, iterators_test):

        ### evaluate cls and cm
        dataset_test.set_augment(False)
        trainer.set_exec_flags(["cls", 'lm', "cm"], [True, False, True])
        for b, batch in enumerate(iterator_test):
            batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
            trainer.evaluate(batch, trainer_args)
            if b % 10 == 0: print('# cls {:.1%} loss={:.4f}'.format(
                b / len(iterator_test), trainer.loss_eval), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

        ### evaluate lm
        if not flag_lm_model:
            dataset_test.set_augment(True)
            trainer.set_exec_flags(["cls", 'lm', "cm"], [False, True, False])
            for b, batch in enumerate(iterator_test):
                batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
                trainer.evaluate(batch, trainer_args)
                if b % 10 == 0: print('# lm {:.1%} loss={:.4f}'.format(
                    b / len(iterator_test), trainer.loss_eval), end='\r', file=sys.stderr)
            print(' ' * 150, end='\r', file=sys.stderr)

        Print(trainer.get_log(test_idx=idx_test, args=trainer_args), output)
        trainer.reset()

    end = Print('end evaluating a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)
    output.close()


if __name__ == '__main__':
    main()