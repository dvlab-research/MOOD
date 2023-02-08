import os
import numpy as np
import sklearn.metrics as skm
from torch.utils.data.dataset import Subset
from scipy import linalg
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

## utils ##
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


### dataset ###
def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)
    subdataset = Subset(dataset, indices)
    return subdataset


def get_superclass_list(dataset):
    CIFAR10_SUPERCLASS = list(range(10))  # one class
    IMAGENET_SUPERCLASS = list(range(30))  # one class
    CIFAR100_SUPERCLASS = [
        [4, 31, 55, 72, 95],
        [1, 33, 67, 73, 91],
        [54, 62, 70, 82, 92],
        [9, 10, 16, 29, 61],
        [0, 51, 53, 57, 83],
        [22, 25, 40, 86, 87],
        [5, 20, 26, 84, 94],
        [6, 7, 14, 18, 24],
        [3, 42, 43, 88, 97],
        [12, 17, 38, 68, 76],
        [23, 34, 49, 60, 71],
        [15, 19, 21, 32, 39],
        [35, 63, 64, 66, 75],
        [27, 45, 77, 79, 99],
        [2, 11, 36, 46, 98],
        [28, 30, 44, 78, 93],
        [37, 50, 65, 74, 80],
        [47, 52, 56, 59, 96],
        [8, 13, 48, 58, 90],
        [41, 69, 81, 85, 89],
    ]
    if dataset.lower() == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset.lower() == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset.lower() == 'imagenet' or dataset.lower() == 'imagenet30':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()

def get_scores_one_cluster(ftrain, ftest, food, args):
    methods = {
        'mahalanobis':  mahalanobis,        # Mahalanobis Distance
        'cos':          cosine_similarity,  # cosine similarity
        'projection':   projection,         # projection distance
        'gauss':        gauss_distribution, # distribution percentage of gauss distribution
        'kmeans':       kmeans,             # the distance to the nearest cluster
        'euclidean':    euclidean_distance, # euclidean distance
        'minkowski':    minkowski_distance, # minkowski distance
        'chebyshev':    chebyshev_distance, # chebyshev distance
    }
    din = methods[args.metric](ftrain, ftest, args)
    dood = methods[args.metric](ftrain, food, args)
    label = [0] * len(din) + [1] * len(dood)
    return din, dood, label


def mahalanobis(ftrain, ftest, args):
    cov = lambda x: np.cov(x.T, bias=True)
    if args.cc or args.avgcc:
        dtest = [[] for _ in range(args.nb_classes)]
        mean = [[] for _ in range(args.nb_classes)]
        std = [[] for _ in range(args.nb_classes)]
        for i in range(args.nb_classes):
            std[i] = np.linalg.pinv(cov(ftrain[i]))
            mean[i] = np.mean(ftrain[i], axis=0, keepdims=True)
            dtest[i] = np.sum((ftest - mean[i])* (std[i].dot((ftest - mean[i]).T)).T, axis=-1,)
        if args.cc:
            return np.min(dtest, axis=0)
        else:
            return np.mean(dtest, axis=0)

    else:
        std = np.linalg.pinv(cov(ftrain))
        mean = np.mean(ftrain, axis=0, keepdims=True)
        dtest = np.sum((ftest - mean)* (std.dot((ftest - mean).T)).T, axis=-1,)
        return dtest

## get features ###
def get_features(model, dataloader, name, args, is_train=False, max_num=1e10):
    model.eval()
    features = []
    for index, (img, label) in enumerate(dataloader):
        if index >= max_num:
            break
        img, label = img.cuda(), label.cuda()
        feature = model.forward_features(img)
        features += list(feature.data.cpu().numpy())
        if args.class_idx is None and (index + 1) % 100 == 0:
            shape = np.array(features).shape
            print('{}: ({}, {})/({}, {})'.format(name, index+1, shape[-1], len(dataloader), shape[-1]), end='\r')
    print('\n')
    features = np.array(features)
    return features


#### OOD detection ####
def get_roc_sklearn(xin, xood, labels):
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    # import pdb;pdb.set_trace()
    return auroc


def get_pr_sklearn(xin, xood, labels=None):
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood, labels):
    if labels == [0] * len(xin) + [1] * len(xood):
        return np.sum(xood < np.percentile(xin, 95)) / len(xood)
    elif labels == [1] * len(xin) + [0] * len(xood):
        return np.sum(xood > np.percentile(xin, 95)) / len(xood)
    else:
        raise


def projection(ftrain, ftest):
    from sklearn.metrics.pairwise import cosine_similarity
    matrix_in = cosine_similarity(ftrain, ftest)
    mod_in = np.linalg.norm(ftest)**2
    din = np.max(matrix_in, axis=1)*mod_in
    return din


def gauss_distribution(ftrain, ftest):
    shrunkcov = True
    if shrunkcov:
        from sklearn.covariance import ledoit_wolf
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    std = np.linalg.pinv(cov(ftrain))
    mean = np.mean(ftrain, axis=0, keepdims=True)
    D = len(ftrain[0])

    dtest = np.sum((ftest - mean)* (std.dot((ftest - mean).T)).T, axis=-1,)
    k = 1 / ((2*np.pi)**(D/2) * np.linalg.norm(std)**0.5)
    ptest = k * np.exp(-0.5 * dtest)
    return ptest

def kmeans(ftrain, ftest, ypred, nclusters):
    from sklearn.cluster import KMeans
    kMeansModel = KMeans(init='k-means++', n_clusters=nclusters, max_iter=100000)
    kMeansModel.fit(ftrain)

    distances = kMeansModel.transform(ftest)
    inDtC = np.min(distances, axis=1)

    return inDtC

def cosine_similarity(ftrain, ftest):
    from sklearn.metrics.pairwise import cosine_similarity
    matrix_in = cosine_similarity(ftrain, ftest)
    din = np.max(matrix_in, axis=1)
    return din

def euclidean_distance(ftrain, ftest):
    mean = np.mean(ftrain, axis=0, keepdims=True)
    dtest = np.sqrt(np.sum(np.power(ftest - mean, 2), axis=-1,))
    return dtest

def minkowski_distance(ftrain, ftest):
    mean = np.mean(ftrain, axis=0, keepdims=True)
    dtest = np.sum(np.abs(ftest - mean), axis=-1,)
    return dtest

def chebyshev_distance(ftrain, ftest):
    mean = np.mean(ftrain, axis=0, keepdims=True)
    dtest = np.max(np.abs(ftest - mean), axis=-1,)
    return dtest