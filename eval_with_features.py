from __future__ import print_function
from __future__ import absolute_import

import os
from pydoc import classname
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import build_dataset
from timm.models import create_model
import ood_utils
import utils

import modeling_finetune
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_scores(ftrain, ftest, food, args):
    return ood_utils.get_scores_one_cluster(ftrain, ftest, food, args)


def get_eval_results(ftrain, ftest, food, args):
    if ftrain is not None:
        if args.cc or args.avgcc:
            for i in range(args.nb_classes):
                ftrain[i] /= np.linalg.norm(ftrain[i], axis=-1, keepdims=True) + 1e-10
        else:
            ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
            
    if ftest is not None:
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10

    if food is not None:
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    dtest, dood, labels = get_scores(ftrain, ftest, food, args)

    fpr95 = ood_utils.get_fpr(dtest, dood, labels)
    auroc, aupr = ood_utils.get_roc_sklearn(dtest, dood, labels), ood_utils.get_pr_sklearn(dtest, dood, labels)
    return fpr95, auroc, aupr


def compute(model, ood_set, features_train, features_test, args):
    ood_sampler = torch.utils.data.SequentialSampler(ood_set)

    ood_loader = torch.utils.data.DataLoader(ood_set, sampler=ood_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False)

    features_ood = ood_utils.get_features(model, ood_loader, args.ood_dataset, args)
    fpr95, auroc, aupr = get_eval_results(features_train, features_test, features_ood, args,)
    return fpr95, auroc, aupr


def get_args():
    parser = argparse.ArgumentParser(description="SSD evaluation")
    parser.add_argument('--ood_dataset', help='name of ood dataset', default=None, type=str)
    parser.add_argument('--ood_data_path', help='path of ood dataset', default=None, type=str)
    
    parser.add_argument('--cc', help='class-conditioned distance', default=False, action='store_true')
    parser.add_argument('--avgcc', help='average of class-conditioned distance', default=False, action='store_true')
    parser.add_argument('--max_num', default=1e10, help='maximum number of test samples', type=int)

    parser.add_argument("--results-dir", type=str, default="./eval_results")
    parser.add_argument('--class_idx', help='One-class OOD: the idx of the ID class number. Multi-class OOD: None', default=None, type=int)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path", required=True)
    parser.add_argument("--metric", type=str, default='mahalanobis', help='OOD detection metric of the features',
                        choices=['mahalanobis', 'cos', 'projection', 'gauss', 'kmeans', 'euclidean', 'minkowski', 'chebyshev'])

    parser.set_defaults(eval_ood=True)
    # Model parameters
    parser.add_argument('--model', default='beit_large_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')

    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--data_set', default='IMNET', choices=['cifar10', 'cifar100', 'imagenet1k', 'imagenet30'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    args = parser.parse_args()
    return args


def main():
    # check ckpt
    assert os.path.exists(args.ckpt), 'Not find {}'.format(args.ckpt)
    print('loading from {}'.format(args.ckpt))

    # load checkpoint
    global model
    if args.ckpt.startswith('https'):
        state_dict = torch.hub.load_state_dict_from_url(
            args.ckpt, map_location='cpu', check_hash=True)
    else:
        state_dict = torch.load(args.ckpt, map_location='cpu')

    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    elif 'module' in state_dict.keys():
        state_dict = state_dict['module']

    for k in list(state_dict.keys()):
        state_dict[k.replace('module.', '')] = state_dict.pop(k)

    if 'pt22k' in args.ckpt:
        model = utils.set_checkpoint_for_finetune(model, args, args.ckpt)
    else:
        utils.load_state_dict(model, state_dict, prefix=args.model_prefix)

    # dataloaders
    train_set, args.nb_classes = build_dataset(is_train=True, data_set=args.data_set, args=args)
    test_set, _ = build_dataset(is_train=False, data_set=args.data_set, args=args)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, 
        batch_size=args.batch_size, num_workers=args.num_workers, 
        pin_memory=args.pin_mem, drop_last=True,)

    test_loader = torch.utils.data.DataLoader(test_set, sampler=test_sampler,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False)
    
    features_train = ood_utils.get_features(model, train_loader, args.data_set, args, is_train=True)
    features_test = ood_utils.get_features(model, test_loader, args.data_set, args)

    if args.class_idx is None:
        ood_set, _ = build_dataset(is_train=False, data_set=args.ood_dataset, args=args, ood=True, ood_data_path=args.ood_data_path)
        fpr95, auroc, aupr = compute(model, ood_set, features_train, features_test, args)

        ood = "%-15s\t" % (args.ood_dataset)
        logger.info("{}\tIn-data: {}\tOOD: {}\tFPR95: {:.2f}\tAUROC: {:.2f}\tAUPR: {:.2f}".format(
                    args.metric, args.data_set, ood, fpr95*100, auroc*100, aupr*100))
        aurocs.append(auroc*100)
        print('\t'.join('{:.1f}'.format(auroc) for auroc in aurocs))
    
    
    else:
        ood_multi_class_set, _ = build_dataset(is_train=False, data_set=args.data_set, args=args, ood=True, ood_data_path=args.ood_data_path)

        fpr95s, aurocs, auprs = [], [], []
        for d in range(len(cls_list)):
            if d == args.class_idx:
                continue
            ood_set = ood_utils.get_subclass_dataset(ood_multi_class_set, classes=cls_list[d])

            args.ood_dataset = str(d)
            fpr95, auroc, aupr = compute(model, ood_set, features_train, features_test, args)
            logger.info("{}\tDataset: {}\tID: {}\tOOD: {}\tFPR95: {:.2f}\tAUROC: {:.2f}\tAUPR: {:.2f}".format(
                args.metric, args.data_set, args.class_idx, d, fpr95*100, auroc*100, aupr*100))

            fpr95s.append(fpr95)
            aurocs.append(auroc)
            auprs.append(aupr)

        fpr95 = np.mean(fpr95s)*100
        auroc = np.mean(aurocs)*100
        aupr = np.mean(auprs)*100

        results = "{}\tDataset: {}\tID: {}\tOOD: {}\tFPR95: {:.2f}\tAUROC: {:.2f}\tAUPR: {:.2f}".format(
                args.metric, args.data_set, args.class_idx, 'all', fpr95, auroc, aupr)
        logger.info(results)
        return fpr95, auroc, aupr


def create_logger(results_dir):
    # create logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    # create results dir
    if results_dir is not None:
        ood_utils.mkdir(results_dir)
        results_file = os.path.join(results_dir, 'eval_results.txt')
        logger.addHandler(logging.FileHandler(results_file, "a"))
        print('=> Saving to {}'.format(results_file))
    return logger


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    _, args.nb_classes = build_dataset(is_train=False, data_set=args.data_set, args=args)
    model = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
        drop_rate=args.drop, drop_path_rate=args.drop_path, attn_drop_rate=args.attn_drop_rate, 
        drop_block_rate=None, use_mean_pooling=args.use_mean_pooling, init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias, use_abs_pos_emb=args.abs_pos_emb, 
        init_values=args.layer_scale_init_value,).cuda()
    
    # ood
    logger = create_logger(args.results_dir)
    if args.class_idx is not None:
        cls_list = ood_utils.get_superclass_list(args.data_set)
    main()
                
            

        

