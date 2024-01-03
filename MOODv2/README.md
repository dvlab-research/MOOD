# Official code for MOODv2: Masked Image Modeling for Out-of-Distribution (OOD) Detection

![framework](MOODv2/imgs/framework.png)

## Performance
![performance](MOODv2/imgs/distribution.png)

![table](MOODv2/imgs/moodv2_table.png)

## DataSets
Dataset source can be downloaded here.
- [ImageNet](https://www.image-net.org/). The ILSVRC 2012 dataset as In-distribution (ID) dataset. The training subset is [this file](datalists/imagenet2012_train_random_200k.txt).
- [OpenImage-O](https://github.com/openimages/dataset/blob/main/READMEV3.md). The OpenImage-O dataset is a subset of the OpenImage-V3 testing set. The filelist is [here](datalists/openimage_o.txt).
- [Texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/). The filelist ruled out four classes that coincides with ImageNet is [here](datalists/texture.txt).
- [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf). Follow the instructions in the [link](https://github.com/deeplearning-wisc/large_scale_ood) to prepare the iNaturalist OOD dataset.
- [ImageNet-O](https://github.com/hendrycks/natural-adv-examples). Follow the guide to download the ImageNet-O OOD dataset.

```bash
mkdir data
cd data
ln -s /path/to/imagenet imagenet
ln -s /path/to/openimage_o openimage_o
ln -s /path/to/texture texture
ln -s /path/to/inaturalist inaturalist
ln -s /path/to/imagenet_o imagenet_o
cd ..
```
## Environment
Install [mmpretrain](https://github.com/open-mmlab/mmpretrain)

## Pretrained Model Preparation
Please download the checkpoints and their corresponding configs

|  Model |  Paper  | Config  | Download|
|:------:|:-------:|:-------:|:-------:|
| BEiT   | [paper](https://arxiv.org/abs/2106.08254) | [config](configs/beit-base-p16_224px.py) | [ckpt](https://download.openmmlab.com/mmclassification/v0/beit/beit-base_3rdparty_in1k_20221114-c0a4df23.pth) |
| BEiTv2 | [paper](https://arxiv.org/abs/2208.06366) | [config](configs/beit-base-p16_224px.py) | [ckpt](https://download.openmmlab.com/mmclassification/v0/beit/beitv2-base_3rdparty_in1k_20221114-73e11905.pth) |
| ViT    | [paper](https://arxiv.org/abs/2010.11929) | [config](configs/vit-base-p16_224px.py) | [ckpt](https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth) |
| MoCov3 | [paper](https://arxiv.org/abs/2104.02057) | [config](configs/vit-base-p16_224px.py) | [ckpt](https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k_20220826-f1e6c442.pth) |
| DINOv2 | [paper](https://arxiv.org/abs/2304.07193) | [config](configs/vit-base-p14_224px.py) | [ckpt](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth) |

## Features and Logits Preparation
### Extract features
   ```bash
   python src/extract_feature_vit.py $DATA_ROOT $OUT_FILE --cfg $CFG --checkpoint $CHECKPOINT --img_list $IMG_LIST
   ```
   e.g.
   ```bash
   python extract_feature_vit.py data/imagenet outputs/vit_imagenet_val.pkl --cfg $CFG --checkpoint $CHECKPOINT --img_list datalists/imagenet2012_val_list.txt
   python extract_feature_vit.py data/imagenet outputs/vit_train_200k.pkl  --cfg $CFG --checkpoint $CHECKPOINT --img_list datalists/imagenet2012_train_random_200k.txt
   python extract_feature_vit.py data/openimage_o outputs/vit_openimage_o.pkl  --cfg $CFG --checkpoint $CHECKPOINT --img_list datalists/openimage_o.txt
   python extract_feature_vit.py data/texture outputs/vit_texture.pkl  --cfg $CFG --checkpoint $CHECKPOINT --img_list datalists/texture.txt
   python extract_feature_vit.py data/inaturalist outputs/vit_inaturalist.pkl  --cfg $CFG --checkpoint $CHECKPOINT 
   python extract_feature_vit.py data/imagenet_o outputs/vit_imagenet_o.pkl  --cfg $CFG --checkpoint $CHECKPOINT 
   python extract_feature_vit.py data/cifar10 outputs/vit_cifar10_train.pkl  --cfg $CFG --checkpoint $CHECKPOINT  --img_list datalists/cifar10_train.txt
   python extract_feature_vit.py data/cifar10 outputs/vit_cifar10_test.pkl  --cfg $CFG --checkpoint $CHECKPOINT  --img_list datalists/cifar10_test.txt
   ```
### Extract w and b in fc
   ```bash
   python src/extract_feature_vit.py $DATA_ROOT $OUT_FILE --cfg $CFG --checkpoint $CHECKPOINT --fc_save_path $FC_SAVE_PATH
   ```
   e.g.
   ```bash
   python src/extract_feature_vit.py $DATA_ROOT $OUT_FILE --cfg $CFG --checkpoint $CHECKPOINT --fc_save_path outputs/vit_fc.pkl 
   ```

### Evaluation
   ```bash
   python src/benchmark.py $FC_SAVE_PATH $ID_DATA $ID_TRAIN_FEATURE $ID_VAL_FEATURE $OOD_FEATURE
   ```
   e.g.
   ```bash
   python src/benchmark.py outputs/vit_fc.pkl outputs/vit_train_200k.pkl outputs/vit_imagenet_val.pkl outputs/vit_openimage_o.pkl outputs/vit_texture.pkl outputs/vit_inaturalist.pkl outputs/vit_imagenet_o.pkl
   python src/benchmark.py outputs/vit_fc.pkl outputs/vit_cifar10_train.pkl outputs/vit_cifar10_test.pkl outputs/vit_openimage_o.pkl outputs/vit_texture.pkl outputs/vit_inaturalist.pkl outputs/vit_imagenet_o.pkl
   ```

## Acknowledgement
Part of the code is modified from [ViM](https://github.com/haoqiwang/vim) repo.
