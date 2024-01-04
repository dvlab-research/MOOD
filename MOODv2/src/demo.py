#!/usr/bin/env python
import argparse
import json
from os.path import basename, splitext
import os
import mmengine
import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp, softmax
from sklearn import metrics
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
import pickle
from os.path import dirname
import torchvision as tv
from PIL import Image
from mmpretrain.apis import init_model

def parse_args():
    parser = argparse.ArgumentParser(description='Detect an image')
    parser.add_argument(
        '--cfg', help='Path to config',
        default='/dataset/jingyaoli/AD/MOOD_/MOODv2/configs/beit-base-p16_224px.py')
    parser.add_argument('--ood_feature', 
        default=None, help='Path to ood feature file')
    parser.add_argument(
        '--checkpoint', help='Path to checkpoint',
        default='/dataset/jingyaoli/AD/MOODv2/pretrain/beit-base_3rdparty_in1k_20221114-c0a4df23.pth',)
    parser.add_argument('--img_path', help='Path to image', 
        default='/dataset/jingyaoli/AD/MOOD_/MOODv2/imgs/DTD_cracked_0004.jpg')
    parser.add_argument('--fc', 
        default='/dataset/jingyaoli/AD/MOODv2/outputs/beit-224px/fc.pkl', help='Path to fc path')
    parser.add_argument('--id_data', default='imagenet', help='id data name')
    parser.add_argument('--id_train_feature', 
        default='/dataset/jingyaoli/AD/MOODv2/outputs/beit-224px/imagenet_train.pkl', help='Path to data')
    parser.add_argument('--id_val_feature', 
        default='/dataset/jingyaoli/AD/MOODv2/outputs/beit-224px/imagenet_test.pkl', help='Path to output file')
    parser.add_argument('--ood_features', 
        default=None, nargs='+', help='Path to ood features')
    parser.add_argument(
        '--methods', nargs='+', 
        default=['MSP', 'MaxLogit', 'Energy', 'Energy+React', 'ViM', 'Residual', 'GradNorm', 'Mahalanobis', ],  # 'KL-Matching'
        help='methods')
    parser.add_argument(
        '--train_label',
        default='datalists/imagenet2012_train_random_200k.txt',
        help='Path to train labels')
    parser.add_argument(
        '--clip_quantile', default=0.99, help='Clip quantile to react')
    parser.add_argument(
        '--fpr', default=95, help='False Positive Rate')
    return parser.parse_args()

def evaluate(method, score_id, score_ood, target_fpr):
    threhold = np.percentile(score_id, 100 - target_fpr)
    if score_ood >= threhold:
        print('\033[94m', method, '\033[0m', 'evaluation:', '\033[92m', 'in-distribution', '\033[0m')
    else:
        print('\033[94m', method, '\033[0m', 'evaluation:', '\033[91m', 'out-of-distribution', '\033[0m')

def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def gradnorm(x, w, b, num_cls):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x, desc='Computing Gradnorm ID/OOD score'):
        targets = torch.ones((1, num_cls)).cuda()
        fc.zero_grad()
        loss = torch.mean(
            torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(
            fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

def extract_image_feature(args):
    torch.backends.cudnn.benchmark = True

    print('=> Loading model')
    cfg = mmengine.Config.fromfile(args.cfg)
    model = init_model(cfg, args.checkpoint, 0).cuda().eval()

    print('=> Loading image')
    if hasattr(cfg.model.backbone, 'img_size'):
        img_size = cfg.model.backbone.img_size
    else:
        img_size = 224

    transform = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    x = transform(Image.open(args.img_path).convert('RGB')).unsqueeze(0)

    print('=> Extracting feature')
    with torch.no_grad():
        x = x.cuda()
        if cfg.model.backbone.type == 'BEiTPretrainViT':
            # (B, L, C) -> (B, C)
            feat_batch = model.backbone(
                x, mask=None)[0].mean(1)
        elif cfg.model.backbone.type == 'SwinTransformer':
            # (B, C, H, W) -> (B, C)
            feat_batch = model.backbone(x)[0]
            B, C, H, W = feat_batch.shape
            feat_batch = feat_batch.reshape(B, C, -1).mean(-1)
        else:
            # (B, C)
            feat_batch = model.backbone(x)[0]
            assert len(feat_batch.shape) == 2
        feature = feat_batch.cpu().numpy()

    print(f'Extracted Feature: {feature.shape}')
    return feature

def main():
    args = parse_args()
    if args.ood_feature and os.path.exists(args.ood_feature):
        feature_ood = mmengine.load(args.ood_feature)
    else:
        feature_ood = extract_image_feature(args)

    if os.path.exists(args.fc):
        w, b = mmengine.load(args.fc)
        print(f'{w.shape=}, {b.shape=}')
    num_cls = len(b)

    train_labels = np.array([
        int(line.rsplit(' ', 1)[-1])
        for line in mmengine.list_from_file(args.train_label)
    ], dtype=int)

    print(f'image path: {args.img_path}')

    print('=> Loading features')
    feature_id_train = mmengine.load(args.id_train_feature).squeeze()
    feature_id_val = mmengine.load(args.id_val_feature).squeeze()

    print(f'{feature_id_train.shape=}, {feature_id_val.shape=}')

    if os.path.exists(args.fc):
        print('=> Computing logits...')
        logit_id_train = feature_id_train @ w.T + b
        logit_id_val = feature_id_val @ w.T + b
        logit_ood = feature_ood @ w.T + b

        print('=> Computing softmax...')
        softmax_id_train = softmax(logit_id_train, axis=-1)
        softmax_id_val = softmax(logit_id_val, axis=-1)
        softmax_ood = softmax(logit_ood, axis=-1)

        u = -np.matmul(pinv(w), b)

    # ---------------------------------------
    method = 'MSP'
    if method in args.methods:
        score_id = softmax_id_val.max(axis=-1)
        score_ood = softmax_ood.max(axis=-1)
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'MaxLogit'
    if method in args.methods:
        score_id = logit_id_val.max(axis=-1)
        score_ood = logit_ood.max(axis=-1)
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'Energy'
    if method in args.methods:
        score_id = logsumexp(logit_id_val, axis=-1)
        score_ood = logsumexp(logit_ood, axis=-1)
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'Energy+React'
    if method in args.methods:
        clip = np.quantile(feature_id_train, args.clip_quantile)
        logit_id_val_clip = np.clip(
            feature_id_val, a_min=None, a_max=clip) @ w.T + b
        score_id = logsumexp(logit_id_val_clip, axis=-1)

        logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
        score_ood = logsumexp(logit_ood_clip, axis=-1)
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'ViM'
    if method in args.methods:
        if feature_id_val.shape[-1] >= 2048:
            DIM = num_cls
        elif feature_id_val.shape[-1] >= 768:
            DIM = 512
        else:
            DIM = feature_id_val.shape[-1] // 2

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        score_id = -vlogit_id_val + energy_id_val

        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'Residual'
    if method in args.methods:
        if feature_id_val.shape[-1] >= 2048:
            DIM = 1000
        elif feature_id_val.shape[-1] >= 768:
            DIM = 512
        else:
            DIM = feature_id_val.shape[-1] // 2
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)

        score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'GradNorm'
    if method in args.methods:
        score_ood = gradnorm(feature_ood, w, b, num_cls)
        score_id = gradnorm(feature_id_val, w, b, num_cls)
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'Mahalanobis'
    if method in args.methods:
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(train_labels.max() + 1), desc='Computing classwise mean feature'):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = torch.from_numpy(np.array(train_means)).cuda().float()
        prec = torch.from_numpy(ec.precision_).cuda().float()

        score_id = -np.array(
            [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
            for f in tqdm(torch.from_numpy(feature_id_val).cuda().float(),  desc='Computing Mahalanobis ID score')])

        score_ood = -np.array([
            (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
            for f in tqdm(torch.from_numpy(feature_ood).cuda().float(), desc='Computing Mahalanobis OOD score')
        ])
        result = evaluate(method, score_id, score_ood, args.fpr)

    # ---------------------------------------
    method = 'KL-Matching'
    if method in args.methods:

        pred_labels_train = np.argmax(softmax_id_train, axis=-1)
        mean_softmax_train = []
        for i in tqdm(range(num_cls), desc='Computing classwise mean softmax'):
            mean_softmax = softmax_id_train[pred_labels_train == i]
            if mean_softmax.shape[0] == 0:
                mean_softmax_train.append(np.zeros((num_cls)))
            else:
                mean_softmax_train.append(np.mean(mean_softmax, axis=0))
            
        score_id = -pairwise_distances_argmin_min(
            softmax_id_val, np.array(mean_softmax_train), metric=kl)[1]

        score_ood = -pairwise_distances_argmin_min(
            softmax_ood, np.array(mean_softmax_train), metric=kl)[1]
        result = evaluate(method, score_id, score_ood, args.fpr)

if __name__ == '__main__':
    main()
