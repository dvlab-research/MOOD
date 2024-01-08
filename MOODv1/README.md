# MOODv1

<p align="center">
• 🤗 <a href="https://huggingface.co/JingyaoLi/MOODv2" target="_blank">Model </a> 
• 🐱 <a href="https://github.com/dvlab-research/MOOD" target="_blank">Code</a> 
• 📃 <a href="https://arxiv.org/abs/2302.02615" target="_blank">MOODv1</a>
• 📃 <a href="https://arxiv.org/abs/2401.02611" target="_blank">MOODv2</a> <br>
</p>
## Abstract
The core of out-of-distribution (OOD) detection is to learn the in-distribution (ID) representation, which is distinguishable from OOD samples. Previous work applied recognition-based methods to learn the ID features, which tend to learn shortcuts instead of comprehensive representations. In this work, we find surprisingly that simply using reconstruction-based methods could boost the performance of OOD detection significantly. We deeply explore the main contributors of OOD detection and find that reconstruction-based pretext tasks have the potential to provide a generally applicable and efficacious prior, which benefits the model in learning intrinsic data distributions of the ID dataset. Specifically, we take Masked Image Modeling as a pretext task for our OOD detection framework (MOOD). Without bells and whistles, MOOD outperforms previous SOTA of one-class OOD detection by 5.7%, multi-class OOD detection by 3.0%, and near-distribution OOD detection by 2.1%. It even defeats the 10-shot-per-class outlier exposure OOD detection, although we do not include any OOD samples for our detection.

## Setup
Follow official [BEiT](https://github.com/microsoft/unilm/tree/master/beit) to setup.

## Datasets
We suggest to organize datasets as following
```bash
- MOOD
    - data
        - cifar10
            - cifar-10-batches-py
        - cifar100
            - cifar-100-python
        - imagenet30
            - test
            - train
            - val
        - imagenet1k
            - test
            - train
            - val
        - $OOD_DATASET
            - images
        ...
```
In this case, for example, if you want to train on CIFAR-10, set the parameters `-- data_path ./data/cifar10 --data_set cifar10`. 

We provide `datasets/imagenet30.py` for you to create soft link for `imagenet30`.

## Pretrained models

Follow [BEiT](https://github.com/microsoft/unilm/tree/master/beit) to pre-train the model or directly utilize the official released weights pretrained on ImageNet-22k. The models were pretrained with 224x224 resolution.
- `BEiT-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)
- `BEiT-large`: #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16 (#parameters: 304M)

Download checkpoints that are **self-supervised pretrained and then intermediate fine-tuned** on ImageNet-22k (recommended):
- BEiT-base: [beit_base_patch16_224_pt22k_ft22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth)
- BEiT-large: [beit_large_patch16_224_pt22k_ft22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)

Download checkpoints that are **self-supervised pretrained** on ImageNet-22k:
- BEiT-base: [beit_base_patch16_224_pt22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth)
- BEiT-large: [beit_large_patch16_224_pt22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth)

## Fine-tuning on In-Distribution Dataset
### Multi-Class Fine-tuning
For ViT-large,
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_large_patch16_224 --data_path $ID_DATA_PATH --data_set $ID_DATASET \
    --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth \
    --batch_size 8 --lr 2e-5 --update_freq 2 \
    --warmup_epochs 5 --epochs 100 --layer_decay 0.9 --drop_path 0.4 \
    --weight_decay 1e-8 --enable_deepspeed
```
The hyper-parameters are the same with the official [BEiT](https://github.com/microsoft/unilm/tree/master/beit).

### One-class Fine-tuning
For one-class fine-tuning, please assign a class as in-distribution by adding command '--class_idx $CLASS_IDX'. Others are out-of-distribution. We support three in-distribution datasets, including `['cifar100', 'cifar10' and 'imagenet30']`. Noted that we only fine-tuned one-class imagenet30 in the original paper.
For ViT-large,
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_large_patch16_224 --data_path $ID_DATA_PATH --data_set $ID_DATASET \
    --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth \
    --batch_size 8 --lr 2e-5 --update_freq 2 \
    --warmup_epochs 5 --epochs 100 --layer_decay 0.9 --drop_path 0.4 \
    --weight_decay 1e-8 --enable_deepspeed --class_idx $CLASS_IDX
```

## OOD detection
### Multi-Class OOD Detection
With OOD detection metric using **features**, we support `['mahalanobis', 'cos', 'projection', 'gauss', 'kmeans', 'euclidean', 'minkowski', 'chebyshev']` with the following command
```bash
python eval_with_features.py --ckpt $CKPT_PATH --data_set $ID_DATASET --ood_dataset $OOD_DATASET --ood_data_path $OOD_DATA_PATH --metric $OOD_METRIC
```
With OOD detection metric using **logits**, we support `['softmax', 'entropy', 'energy', 'gradnorm']` with the following command
```bash
python eval_with_logits.py --ckpt $CKPT_PATH --data_set $ID_DATASET --ood_dataset $OOD_DATASET --ood_data_path $OOD_DATA_PATH --metric $OOD_METRIC
```

### One-Class OOD Detection
For one-class OOD detection, please assign a class as in-distribution by adding command '--class_idx $CLASS_IDX'. Others are out-of-distribution. We support three in-distribution datasets, including `['cifar100', 'cifar10' and 'imagenet30']`. 
With OOD detection metric using **features**, we support `['mahalanobis', 'cos', 'projection', 'gauss', 'kmeans', 'euclidean', 'minkowski', 'chebyshev']` with the following command
```bash
python eval_with_features.py --ckpt $CKPT_PATH --data_set $ID_DATASET --metric $OOD_METRIC --class_idx $CLASS_IDX
```
With OOD detection metric using **logits**, we support `['softmax', 'entropy', 'energy', 'gradnorm']` with the following command
```bash
python eval_with_logits.py --ckpt $CKPT_PATH --data_set $ID_DATASET --metric $OOD_METRIC --class_idx $CLASS_IDX
```

## Results
### Multi-class OOD detection
For CIFAR-10,
| CIFAR-10 	|   SVHN   	| CIFAR-100 	|   LSUN   	|  Avg  	| 
|:--------:	|:--------:	|:---------:	|:--------:	|:-----:	|
|    [ckpt](https://drive.google.com/file/d/1b_uWi2bty3tyspxEEM4jtyCh3WR9FpYm/view?usp=share_link), [distances](https://drive.google.com/drive/folders/1MeoEHArSeHc7D35-9vGEt782GKphU2PM?usp=share_link)  	|   99.8   	|    99.4   	|   99.9   	| 99.7  	|

For CIFAR-100,
| CIFAR-100 	|   SVHN   	| CIFAR-10 	|   LSUN   	|  Avg  	|
|:---------:	|:--------:	|:--------:	|:--------:	|:-----:	|
|    [ckpt](https://drive.google.com/file/d/1MCPUTnz5DjNmR8gyWGMAl7qW811cH13X/view?usp=share_link), [distances](https://drive.google.com/drive/folders/1CV4kpb3OKiCj9uN9fMj1reG59RPDcF_X?usp=share_link)   	|   96.5   	|   98.3   	|   96.3   	| 97.0  	|

For ImageNet-30,
| ImageNet30 	|   Dogs   	| Places365 	| Flowers102 	|   Pets   	|   Food   	| Caltech256 	|    Dtd   	|  Avg  	|
|:----------:	|:--------:	|:---------:	|:----------:	|:--------:	|:--------:	|:----------:	|:--------:	|:-----:	|
|     [ckpt](https://drive.google.com/file/d/1nTOimKRcNHlT_hKNfWJejDkYyDs4xd63/view?usp=share_link), [distances](https://drive.google.com/drive/folders/1CH3TjnohalbUIWgNcKzM3Swzy5BIo--f?usp=share_link)   	|  99.40   	|   98.90   	|   100.00   	|  99.10   	|  96.60   	|   99.50    	|   98.9   	| 98.9  	|

For ImageNet-1k,
| ImageNet1k 	| iNaturalist 	|    SUN   	|  Places  	| Textures 	| Average 	|
|:----------:	|:-----------:	|:--------:	|:--------:	|:--------:	|:-------:	|
|     [ckpt](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft1k.pth), [distances](https://drive.google.com/drive/folders/1-JT_81-a8mRMc_jikeuVKJYDbLCJz_mr?usp=share_link)   	|     86.9    	|   89.8   	|   88.5   	|   91.3   	|   89.1  	|

### One-class OOD detection
For CIFAR-10,
|   Method  	| Airplane 	| Automobile 	|   Bird   	|    Cat   	|   Dear   	|    Dog   	|   Frog   	|   Horse  	|   Ship   	|   Truck  	|
|:---------:	|:--------:	|:----------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|
|    ours   	| 98.6 	|  99.3  	| 94.3 	| 93.2 	| 98.1 	| 96.5 	| 99.3 	| 99.0 	| 98.8 	| 97.8 	|

For CIFAR-100,
|   Method  	|   AUROC  	|
|:---------:	|:--------:	|
|    ours   	| 96.4 	|

For ImageNet-30,
|         Method        	|   AUROC  	|
|:---------------------:	|:--------:	|
|          ours         	| 92.0 	|

## Acknowledgement

This repository is built using the [beit](https://github.com/microsoft/unilm/tree/master/beit) library and the [SSD](https://github.com/inspire-group/SSD) repository.

## Citation
If you find our research helpful, kindly cite
```
@inproceedings{li2023rethinking,
  title={Rethinking Out-of-distribution (OOD) Detection: Masked Image Modeling is All You Need},
  author={Li, Jingyao and Chen, Pengguang and He, Zexin and Yu, Shaozuo and Liu, Shu and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11578--11589},
  year={2023}
}
```