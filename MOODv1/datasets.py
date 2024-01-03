# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms
from dataset_folder import ImageFolder, SegmentationDataset
import blobfile as bf
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform


from typing import TypeVar, Generic
from dall_e.utils import map_pixels
from masking_generator import MaskingGenerator
import ood_utils
from PIL import Image
import numpy as np

T_co = TypeVar('T_co', covariant=True)

class Dataset(Generic[T_co]):
    def __getitem__(self, index):
        raise NotImplementedError

class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_beit_pretraining_dataset(args, data_set=None):
    '''train set for beit'''
    if data_set == None:
        data_set = args.data_set
    transform = DataAugmentationForBEiT(args)
    
    # print("Data Aug = %s" % str(transform))
    data_path = args.data_path
    if data_set == 'cifar100':
        dataset = datasets.CIFAR100(data_path, train=True, transform=transform)
    elif data_set == 'cifar10':
        dataset = datasets.CIFAR10(data_path, train=True, transform=transform)
    elif data_set == 'imagenet30':
        dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    else:
        dataset = ImageFolder(data_path, transform=transform)
    return dataset


class Subset(Dataset[T_co]):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.array(indices)
        self.data = np.array(dataset.data)[self.indices, :]
        self.targets = np.array(dataset.targets)[self.indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class Mixup(Dataset[T_co]):
    def __init__(self, dataset, transform, alpha=0.2):
        self.dataset = dataset
        self.targets = dataset.targets
        self.data = dataset.data
        self.alpha = alpha
        self.baselenth = len(self.data)
        self.transform = transform


    def __getitem__(self, idx):
        if idx < self.baselenth:
            img, target =  self.data[idx], self.targets[idx]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        else:
            img, target =  self.data[idx-self.baselenth], self.targets[idx-self.baselenth]
            img = Image.fromarray(img)
            img = self.transform(img)

            lam = np.random.beta(self.alpha, self.alpha)
            mix_index = np.random.randint(0, self.baselenth-1)
            mix_img, mix_target =  self.data[mix_index], self.targets[mix_index]
            mix_img = Image.fromarray(mix_img)
            mix_img = self.transform(mix_img)

            img = lam * img + (1 - lam) * mix_img
            target = int(lam * target + (1 - lam) * mix_target)
            return img, target

    def __len__(self):
        return self.baselenth*2


class Rotation(Dataset[T_co]):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.targets = dataset.targets
        self.data = dataset.data
        self.baselenth = len(self.data)
        self.transform = transform

    def __getitem__(self, idx):
        rot_angle = idx // self.baselenth   # range: 0-3
        image_idx = idx % self.baselenth    # range: 0-baselenth
        img, target = self.data[image_idx], self.targets[image_idx]
        img = np.rot90(img, rot_angle)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return self.baselenth*4


class Transform(Dataset[T_co]):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.data = dataset.data
        self.targets = dataset.targets
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


def build_dataset(is_train, args, data_set=None, ood=False, is_trans=True, ood_data_path=None):
    if data_set == None:
        data_set = args.data_set

    if not is_trans:
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])
    else:
        transform = build_transform(is_train, args)
    
    if ood_data_path is None:
        data_path = args.data_path
    else:
        data_path = ood_data_path
        
    if data_set == 'cifar100':
        dataset = datasets.CIFAR100(data_path, train=is_train, transform=transform, download=False)
        nb_classes = 100
    elif data_set == 'cifar10':
        dataset = datasets.CIFAR10(data_path, train=is_train, transform=transform, download=False)
        nb_classes = 10
    elif data_set == 'svhn':
        dataset = datasets.SVHN(data_path, split='train' if is_train else 'test', transform=transform, download=False)
        nb_classes = 10
    elif data_set == 'imagenet30':
        split = 'train' if is_train else 'test'
        nb_classes = 30
        dataset = ImageFolder(os.path.join(data_path, split), transform=transform)
    elif data_set == 'caltech256':
        split = 'train' if is_train else 'test'
        nb_classes = 256
        dataset = ImageFolder(os.path.join(data_path, split), transform=transform)
    elif data_set == 'imagenet1k':
        root = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif data_set == "image_folder":
        root = data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes        
    else:
        nb_classes = None
        dataset = ImageFolder(data_path, transform=transform)

    if isinstance(args.class_idx, int) and not ood:
        print('Using one class idx (class idx:{})'.format(args.class_idx))
        cls_list = get_superclass_list(data_set)
        dataset = get_subclass_dataset(dataset, classes=cls_list[args.class_idx])

    return dataset, nb_classes


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results



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
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset.lower() == 'imagenet' or dataset.lower() == 'imagenet30':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)
    dataset = Subset(dataset, indices)
    return dataset


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


