# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# PLUS

import os
import sys
import argparse

import torch

import plus.config as config
from plus.data.alphabets import Protein
import plus.data.pfam as pfam
import plus.data.dataset as dataset
import plus.model.plus_rnn as plus_rnn
import plus.model.plus_tfm as plus_tfm
import plus.model.p_elmo as p_elmo
from plus.train import Trainer
from plus.utils import Print, set_seeds, set_output, load_models


parser = argparse.ArgumentParser('Evaluate a Model on Pfam Datasets')
parser.add_argument('--data-config',  help='path for data configuration file')
parser.add_argument('--model-config', help='path for model configuration file')
parser.add_argument('--run-config', help='path for run configuration file')
parser.add_argument('--pretrained-model', help='path for pretrained model file')
parser.add_argument('--device', help='device to use; multi-GPU if given multiple GPUs sperated by comma (default: cpu)')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')
parser.add_argument('--output-index', help='index for outputs')
parser.add_argument('--sanity-check', default=False, action='store_true', help='sanity check flag')


def main():
    set_seeds(2020)
    args = vars(parser.parse_args())

    alphabet = Protein()
    data_cfg  = config.DataConfig(args["data_config"])
    model_cfg = config.ModelConfig(args["model_config"], input_dim=len(alphabet), num_classes=2)
    run_cfg   = config.RunConfig(args["run_config"], eval=True, sanity_check=args["sanity_check"])
    output, save_prefix = set_output(args, "eval_pfam_log", test=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args["device"] if args["device"] is not None else ""
    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1
    config.print_configs(args, [data_cfg, model_cfg, run_cfg], device, output)
    flag_rnn = (model_cfg.model_type == "RNN")
    flag_plus = (model_cfg.rnn_type == "B") if flag_rnn else False
    flag_paired = ("testpairs" in data_cfg.path)

    ## load a test dataset
    start = Print(" ".join(['start loading a test dataset:', data_cfg.path["testpairs" if flag_paired else "test"]]), output)
    if flag_paired:
        dataset_test = pfam.load_pfam_pairs(data_cfg, "testpairs", alphabet, args["sanity_check"])
        dataset_test = dataset.PairedPfam_dataset(*dataset_test, alphabet, run_cfg, flag_rnn, model_cfg.max_len)
    else:
        dataset_test = pfam.load_pfam(data_cfg, "test", alphabet, args["sanity_check"])
        dataset_test = dataset.Pfam_dataset(*dataset_test, alphabet, run_cfg, flag_rnn, model_cfg.max_len,
                                            random_pairing=flag_paired, augment=flag_plus, sanity_check=args["sanity_check"])
    if flag_rnn and flag_paired: collate_fn = dataset.collate_paired_sequences
    elif flag_rnn and flag_plus: collate_fn = dataset.collate_sequences
    elif flag_rnn:               collate_fn = dataset.collate_sequences_pelmo
    else:                        collate_fn = None
    iterator_test = torch.utils.data.DataLoader(dataset_test, run_cfg.batch_size_eval, collate_fn=collate_fn)
    end = Print(" ".join(['loaded', str(len(dataset_test)), 'sequence(pair)s']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## initialize a model
    start = Print('start initializing a model', output)
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    if not flag_rnn:                model = plus_tfm.PLUS_TFM(model_cfg)
    elif flag_plus:                 model = plus_rnn.PLUS_RNN(model_cfg)
    else:                           model = p_elmo.P_ELMo_lm(model_cfg)
    models_list.append([model, "", True, False, False])
    load_models(args, models_list, device, data_parallel, output)
    get_loss = plus_rnn.get_loss if flag_rnn else plus_tfm.get_loss
    end = Print('end initializing a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    tasks_list = [] # list of lists [idx, metrics_train, metrics_eval]
    tasks_list.append(["lm",   [], ["acc"]])
    if flag_paired: tasks_list.append(["cls",  [], ["acc"]])
    trainer = Trainer(models_list, get_loss, run_cfg, tasks_list)
    trainer_args = {}
    trainer_args["data_parallel"] = data_parallel
    trainer_args["paired"] = flag_paired
    if   flag_paired and flag_rnn: trainer_args["evaluate_cls"] = plus_rnn.evaluate_sfp
    elif flag_paired:              trainer_args["evaluate_cls"] = plus_tfm.evaluate_sfp
    else:                          trainer_args["num_alphabets"] = len(alphabet)
    end = Print('end setting trainer configurations', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## evaluate a model
    start = Print('start evaluating a model', output)
    Print(trainer.get_headline(test=True), output)

    ### evaluate lm
    if flag_paired: dataset_test.set_augment(True)
    trainer.set_exec_flags(["lm", "cls"], [True, False])
    for b, batch in enumerate(iterator_test):
        batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
        trainer.evaluate(batch, trainer_args)
        if b % 10 == 0: print('# lm {:.1%} loss={:.4f}'.format(
            b / len(iterator_test), trainer.loss_eval), end='\r', file=sys.stderr)
    print(' ' * 150, end='\r', file=sys.stderr)

    ### evaluate cls
    if flag_paired:
        dataset_test.set_augment(False)
        trainer.set_exec_flags(["lm", "cls"], [False, True])
        for b, batch in enumerate(iterator_test):
            batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
            trainer.evaluate(batch, trainer_args)
            if b % 10 == 0: print('# cls {:.1%} loss={:.4f}'.format(
                b / len(iterator_test), trainer.loss_eval), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

    Print(trainer.get_log(test_idx="Pfam", args=trainer_args), output)
    end = Print('end evaluating a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)
    output.close()


if __name__ == '__main__':
    main()