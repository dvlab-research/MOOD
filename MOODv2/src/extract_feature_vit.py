#!/usr/bin/env python
import argparse
import pickle
from os.path import dirname
import os
import mmengine
import numpy as np
import torch
import torchvision as tv
from tqdm import tqdm

from .list_dataset import ImageFilelist
from mmpretrain.apis import init_model

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('data_root', help='Path to data')
    parser.add_argument('--out_file', help='Path to output file')
    parser.add_argument(
        '--cfg', default='configs/vit-base-p16-384.py', help='Path to config')
    parser.add_argument(
        '--checkpoint',
        default='checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-'
        '384_20210928-98e8652b.pth',
        help='Path to checkpoint')
    parser.add_argument('--img_list', help='Path to image list')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='num of workers')
    parser.add_argument('--fc_save_path', default=None, help='Path to save fc')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    cfg = mmengine.Config.fromfile(args.cfg)
    model = init_model(cfg, args.checkpoint, 0).cuda().eval()

    if args.fc_save_path is not None:
        if os.path.exists(args.fc_save_path):
            print(f'{args.fc_save_path} exists.')
            return
        mmengine.mkdir_or_exist(dirname(args.fc_save_path))
        if cfg.model.head.type == 'VisionTransformerClsHead':
            fc = model.head.layers.head
        elif cfg.model.head.type == 'LinearClsHead':
            fc = model.head.fc
        elif cfg.model.head.type in ['BEiTV1Head', 'BEiTV2Head']:
            fc = model.head.cls_head
        elif cfg.model.head.type in ['MoCoV3Head']:
            print(f'{cfg.model.head.type} utilize NonLinearNetwork which cannot be represented by a weight and a bias')
            raise 
        else:
            print(cfg.model.head.type)
            print(model.head)
            import pdb;pdb.set_trace()
            raise NotImplementedError(cfg.model.backbone.type)
        w = fc.weight.cpu().detach().numpy()
        b = fc.bias.cpu().detach().numpy()
        with open(args.fc_save_path, 'wb') as f:
            pickle.dump([w, b], f)
        return

    # if os.path.exists(out_file):
    #     print(f'{out_file} exists.')
    #     return

    if hasattr(cfg.model.backbone, 'img_size'):
        img_size = cfg.model.backbone.img_size
    else:
        img_size = 224

    transform = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.img_list not in [None, 'None']:
        dataset = ImageFilelist(args.data_root, args.img_list, transform)
    else:
        dataset = tv.datasets.ImageFolder(args.data_root, transform)
    print(f'lenth of dataset: {len(dataset)}')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    features = []
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(dataloader)):
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
            features.append(feat_batch.cpu().numpy())

    features = np.concatenate(features, axis=0)
    print(f'features: {features.shape}')
    mmengine.mkdir_or_exist(dirname(args.out_file))
    with open(args.out_file, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':
    main()
