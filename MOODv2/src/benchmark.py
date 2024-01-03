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


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('fc', help='Path to fc path')
    parser.add_argument('id_data', help='id data name')
    parser.add_argument('id_train_feature', help='Path to data')
    parser.add_argument('id_val_feature', help='Path to output file')
    parser.add_argument('ood_features', nargs='+', help='Path to ood features')
    parser.add_argument(
        '--class_id', type=int, help='class id for 1 class ood detection')
    parser.add_argument(
        '--methods', nargs='+', 
        default=['MSP', 'MaxLogit', 'Energy', 'Energy+React', 'ViM', 'Residual', 'GradNorm', 'Mahalanobis', 'KL-Matching'],  # 
        help='methods')
    parser.add_argument(
        '--score_path', default=None, help='path to socres')
    parser.add_argument(
        '--train_label',
        default='datalists/imagenet2012_train_random_200k.txt',
        help='Path to train labels')
    parser.add_argument(
        '--test_label',
        default=None, help='Path to test labels. Only to divide 1 class ood datasets, instead of ood detection')
    parser.add_argument(
        '--clip_quantile', default=0.99, help='Clip quantile to react')
    parser.add_argument(
        '--save_path', default='results.json', help='Path to results')
    return parser.parse_args()


# region Helper
def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate(
        (np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# endregion

# region OOD


def gradnorm(x, w, b, num_cls):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = torch.ones((1, num_cls)).cuda()
        fc.zero_grad()
        loss = torch.mean(
            torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(
            fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


# endregion


def main():
    args = parse_args()

    if args.save_path == 'None':
        args.save_path = None 
    if args.score_path == 'None':
        args.score_path == None
    if args.class_id == 'None':
        args.class_id == None

    score_path = args.score_path
    if score_path:
        os.makedirs(score_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if os.path.exists(args.fc):
        w, b = mmengine.load(args.fc)
        print(f'{w.shape=}, {b.shape=}')
    num_cls = len(b)

    train_labels = np.array([
        int(line.rsplit(' ', 1)[-1])
        for line in mmengine.list_from_file(args.train_label)
    ], dtype=int)
    recall = 0.95

    if args.class_id is not None: 
        ood_names = list(range(max(train_labels) + 1))
        ood_names.remove(args.class_id)
    else:
        ood_names = [splitext(basename(ood))[0] for ood in args.ood_features]
    print(f'ood datasets: {ood_names}')

    print('load features')
    feature_id_train = mmengine.load(args.id_train_feature).squeeze()
    feature_id_val = mmengine.load(args.id_val_feature).squeeze()

    if args.class_id is not None: 
        # divide 1 class detection id and ood datasets
        feature_val = feature_id_val.copy()

        test_labels = np.array([
            int(line.rsplit(' ', 1)[-1])
            for line in mmengine.list_from_file(args.test_label)
        ], dtype=int)

        feature_id_train = feature_id_train[train_labels==args.class_id]
        feature_id_val = feature_val[test_labels==args.class_id]

        feature_oods = {
            name: feature_val[test_labels==name] for name in ood_names}

        train_labels = np.array([0] * len(feature_id_train))

    else:
        feature_oods = {
            name: mmengine.load(feat).squeeze()
            for name, feat in zip(ood_names, args.ood_features)
        }

    print(f'{feature_id_train.shape=}, {feature_id_val.shape=}')
    for name, ood in feature_oods.items():
        print(f'{name} {ood.shape}')

    if os.path.exists(args.fc):
        print('computing logits...')
        logit_id_train = feature_id_train @ w.T + b
        logit_id_val = feature_id_val @ w.T + b
        logit_oods = {name: feat @ w.T + b for name, feat in feature_oods.items()}

        print('computing softmax...')
        softmax_id_train = softmax(logit_id_train, axis=-1)
        softmax_id_val = softmax(logit_id_val, axis=-1)
        softmax_oods = {
            name: softmax(logit, axis=-1)
            for name, logit in logit_oods.items()
        }

        u = -np.matmul(pinv(w), b)

    df = pd.DataFrame(columns=['method', 'oodset', 'auroc', 'fpr'])
    dfs = []
    
    # ---------------------------------------
    method = 'MSP'
    if method in args.methods:
        print(f'\n{method}')
        result = []
        score_id = softmax_id_val.max(axis=-1)

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, softmax_ood in softmax_oods.items():
            score_ood = softmax_ood.max(axis=-1)

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)

            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'MaxLogit'
    if method in args.methods:
        print(f'\n{method}')
        result = []
        score_id = logit_id_val.max(axis=-1)

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, logit_ood in logit_oods.items():
            score_ood = logit_ood.max(axis=-1)

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)

            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Energy'
    if method in args.methods:
        print(f'\n{method}')
        result = []
        score_id = logsumexp(logit_id_val, axis=-1)

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, logit_ood in logit_oods.items():
            score_ood = logsumexp(logit_ood, axis=-1)

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)

            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Energy+React'
    if method in args.methods:
        print(f'\n{method}')
        result = []

        clip = np.quantile(feature_id_train, args.clip_quantile)
        print(f'clip quantile {args.clip_quantile}, clip {clip:.4f}')

        logit_id_val_clip = np.clip(
            feature_id_val, a_min=None, a_max=clip) @ w.T + b
        score_id = logsumexp(logit_id_val_clip, axis=-1)

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, feature_ood in feature_oods.items():
            logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
            score_ood = logsumexp(logit_ood_clip, axis=-1)

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)

            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'ViM'
    if method in args.methods:
        print(f'\n{method}')
        result = []
        if feature_id_val.shape[-1] >= 2048:
            DIM = num_cls
        elif feature_id_val.shape[-1] >= 768:
            DIM = 512
        else:
            DIM = feature_id_val.shape[-1] // 2
        print(f'{DIM=}')

        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        print('computing alpha...')
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')

        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        score_id = -vlogit_id_val + energy_id_val

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                                feature_oods.values()):
            energy_ood = logsumexp(logit_ood, axis=-1)
            vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
            score_ood = -vlogit_ood + energy_ood

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)

            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Residual'
    if method in args.methods:
        print(f'\n{method}')
        result = []
        if feature_id_val.shape[-1] >= 2048:
            DIM = 1000
        elif feature_id_val.shape[-1] >= 768:
            DIM = 512
        else:
            DIM = feature_id_val.shape[-1] // 2
        print(f'{DIM=}')

        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                                feature_oods.values()):
            score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)

            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'GradNorm'
    if method in args.methods:
        print(f'\n{method}')
        result = []
        score_id = gradnorm(feature_id_val, w, b, num_cls)

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, feature_ood in feature_oods.items():
            score_ood = gradnorm(feature_ood, w, b, num_cls)

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)

            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Mahalanobis'
    if method in args.methods:
        print(f'\n{method}')
        result = []

        print('computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(train_labels.max() + 1)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        print('go to gpu...')
        mean = torch.from_numpy(np.array(train_means)).cuda().float()
        prec = torch.from_numpy(ec.precision_).cuda().float()

        score_id = -np.array(
            [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
            for f in tqdm(torch.from_numpy(feature_id_val).cuda().float())])

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, feature_ood in feature_oods.items():
            score_ood = -np.array([
                (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                for f in tqdm(torch.from_numpy(feature_ood).cuda().float())
            ])

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)
                    
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'KL-Matching'
    if method in args.methods:
        print(f'\n{method}')
        result = []

        print('computing classwise mean softmax...')
        pred_labels_train = np.argmax(softmax_id_train, axis=-1)
        mean_softmax_train = []
        for i in tqdm(range(num_cls)):
            mean_softmax = softmax_id_train[pred_labels_train == i]
            if mean_softmax.shape[0] == 0:
                mean_softmax_train.append(np.zeros((num_cls)))
            else:
                mean_softmax_train.append(np.mean(mean_softmax, axis=0))
            
        score_id = -pairwise_distances_argmin_min(
            softmax_id_val, np.array(mean_softmax_train), metric=kl)[1]

        if score_path:
            os.makedirs(os.path.join(score_path, method), exist_ok=True)
            np.save(
                os.path.join(score_path, method, f'{args.id_data}_score.npy'), 
                score_id)

        for name, softmax_ood in softmax_oods.items():
            score_ood = -pairwise_distances_argmin_min(
                softmax_ood, np.array(mean_softmax_train), metric=kl)[1]

            if score_path:
                np.save(
                    os.path.join(score_path, method, f'{name}_score.npy'), score_ood)
                    
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(
                dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    if args.save_path:
        with open(args.save_path, 'w') as json_file:
            data = [df.to_dict(orient='records') for i, df in enumerate(dfs)]
            json.dump(data, json_file)

if __name__ == '__main__':
    main()
