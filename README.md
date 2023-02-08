# [MOOD: Masked Image Modeling is All You Need]

Official PyTorch implementation and pretrained models of MOOD.

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
For ViT-base,
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS run_class_finetuning.py \
    --model beit_base_patch16_224 --data_path $ID_DATA_PATH --data_set $ID_DATASET \
    --nb_classes $NUM_OF_CLASSES --disable_eval_during_finetuning \
    --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth \
    --output_dir $OUTPUT_DIR --batch_size 256 --lr 3e-3 --update_freq 1 \
    --warmup_epochs 5 --epochs 90 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 --enable_deepspeed --layer_scale_init_value 0.1 --clip_grad 3.0
```

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

For ViT-base,
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS run_class_finetuning.py \
    --model beit_base_patch16_224 --data_path $ID_DATA_PATH --data_set $ID_DATASET \
    --nb_classes $NUM_OF_CLASSES --disable_eval_during_finetuning \
    --finetune https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth \
    --output_dir $OUTPUT_DIR --batch_size 256 --lr 3e-3 --update_freq 1 \
    --warmup_epochs 5 --epochs 90 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 --enable_deepspeed --layer_scale_init_value 0.1 --clip_grad 3.0 --class_idx $CLASS_IDX
``` 

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
|        OOD       	|   SVHN   	| CIFAR-100 	|   LSUN   	|  Average 	|
|:----------------:	|:--------:	|:---------:	|:--------:	|:--------:	|
|   Baseline OOD   	|   88.6   	|   85.8    	|   90.7   	|   88.4   	|
|       ODIN       	|   96.4   	|   89.6    	|     -    	|   93.0   	|
|    Mahalanobis   	|   99.4   	|   90.5    	|     -    	|   95.0   	|
|  Residual Flows  	|   99.1   	|   89.4    	|     -    	|   94.3   	|
|    Gram Matrix   	|   99.5   	|   79.0    	|     -    	|   89.3   	|
| Outlier exposure 	|   98.4   	|   93.3    	|     -    	|   95.9   	|
|   Rotation loss  	|   98.9   	|   90.9    	|     –    	|   94.9   	|
| Contrastive loss 	|   97.3   	|   88.6    	|   92.8   	|   92.9   	|
|        CSI       	|   97.9   	|   92.2    	|   97.7   	|   95.9   	|
|       SSD+       	| **99.9** 	|   93.4    	|   98.4   	|   97.2   	|
|       ours       	|   99.8   	|  **99.4** 	| **99.9** 	| **99.7** 	|

For CIFAR-100,
|        OOD       	|   SVHN   	| CIFAR-10 	|   LSUN   	|  Average 	|
|:----------------:	|:--------:	|:--------:	|:--------:	|:--------:	|
|   Baseline OOD   	|   81.9   	|   81.1   	|   86.6   	|   83.2   	|
|       ODIN       	|   60.9   	|   77.9   	|     -    	|   69.4   	|
|    Mahalanobis   	|   94.5   	|   55.3   	|     -    	|   74.9   	|
|  Residual Flows  	|   97.5   	|   77.1   	|     -    	|   87.3   	|
|    Gram Matrix   	|   96.0   	|   67.9   	|     -    	|   82.0   	|
| Outlier exposure 	|   86.9   	|   75.7   	|     -    	|   81.3   	|
|        CSI       	|   95.6   	|   78.3   	|     -    	|   87.0   	|
|       SSD+       	| **98.2** 	|   78.3   	|   79.8   	|   85.4   	|
|       ours       	|   96.5   	| **98.3** 	| **96.3** 	| **97.0** 	|

For ImageNet-30,
|      OOD      	|   Dogs   	| Places365 	| Flowers102 	|   Pets   	|   Food   	| Caltech256 	|    Dtd   	|  Average 	|
|:-------------:	|:--------:	|:---------:	|:----------:	|:--------:	|:--------:	|:----------:	|:--------:	|:--------:	|
| Cross Entropy 	|   96.7   	|   90.5    	|    89.7    	|   95.0   	|   79.8   	|    90.6    	|   90.1   	|   89.3   	|
|     SupCon    	|   95.6   	|   89.7    	|    92.2    	|   94.2   	|   81.2   	|    90.2    	|   92.1   	|   89.9   	|
|      CSI      	|   98.3   	|   94.0    	|    96.2    	|   97.4   	|   87.0   	|    93.2    	|   97.4   	|   94.2   	|
|      ours     	| **99.4** 	|  **98.9** 	|  **100.0** 	| **99.1** 	| **96.6** 	|  **99.5**  	| **98.9** 	| **98.9** 	|

For ImageNet-1k,
|      OOD     	| iNaturalist 	|    SUN   	|  Places  	| Textures 	|  Average 	|
|:------------:	|:-----------:	|:--------:	|:--------:	|:--------:	|:--------:	|
| Baseline OOD 	|     87.6    	|   78.3   	|   76.8   	|   74.5   	|   79.3   	|
|     ODIN     	|     89.4    	|   83.9   	|   80.7   	|   76.3   	|   82.6   	|
|    Energy    	|     88.5    	|   85.3   	|   81.4   	|   75.8   	|   82.7   	|
|  Mahalanobis 	|     46.3    	|   65.2   	|   64.5   	|   72.1   	|   62.0   	|
|   GradNorm   	|   **90.3**  	|   89.0   	|   84.8   	|   81.1   	|   86.3   	|
|     ours     	|     86.9    	| **89.8** 	| **88.5** 	| **91.3** 	| **89.1** 	|

### One-class OOD detection
For CIFAR-10,
|   Method  	| Airplane 	| Automobile 	|   Bird   	|    Cat   	|   Dear   	|    Dog   	|   Frog   	|   Horse  	|   Ship   	|   Truck  	|
|:---------:	|:--------:	|:----------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|:--------:	|
|   OC-SVM  	|   65.6   	|    40.9    	|   65.3   	|   50.1   	|   75.2   	|   51.2   	|   71.8   	|   51.2   	|   67.9   	|   48.5   	|
|  DeepSVDD 	|   61.7   	|    65.9    	|   50.8   	|   59.1   	|   60.9   	|   65.7   	|   67.7   	|   67.3   	|   75.9   	|   73.1   	|
|   AnoGAN  	|   67.1   	|    54.7    	|   52.9   	|   54.5   	|   65.1   	|   60.3   	|   58.5   	|   62.5   	|   75.8   	|   66.5   	|
|   OCGAN   	|   75.7   	|    53.1    	|   64.0   	|   62.0   	|   72.3   	|   62.0   	|   72.3   	|   57.5   	|   82.0   	|   55.4   	|
|    Geom   	|   74.7   	|    95.7    	|   78.1   	|   72.4   	|   87.8   	|   87.8   	|   83.4   	|   95.5   	|   93.3   	|   91.3   	|
|    Rot    	|   71.9   	|    94.5    	|   78.4   	|   70.0   	|   77.2   	|   86.6   	|   81.6   	|   93.7   	|   90.7   	|   88.8   	|
| Rot+Trans 	|   77.5   	|    96.9    	|   87.3   	|   80.9   	|   92.7   	|   90.2   	|   90.9   	|   96.5   	|   95.2   	|   93.3   	|
|    GOAD   	|   77.2   	|    96.7    	|   83.3   	|   77.7   	|   87.8   	|   87.8   	|   90.0   	|   96.1   	|   93.8   	|   92.0   	|
|    CSI    	|   89.9   	|    99.1    	|   93.1   	|   86.4   	|   93.9   	|   93.2   	|   95.1   	|   98.7   	|   97.9   	|   95.5   	|
|    ours   	| **98.6** 	|  **99.3**  	| **94.3** 	| **93.2** 	| **98.1** 	| **96.5** 	| **99.3** 	| **99.0** 	| **98.8** 	| **97.8** 	|

For CIFAR-100,
|   Method  	|   AUROC  	|
|:---------:	|:--------:	|
|   OC-SVM  	|   63.1   	|
|    Geom   	|   78.7   	|
|    Rot    	|   77.7   	|
| Rot+Trans 	|   79.8   	|
|    GOAD   	|   74.5   	|
|    CSI    	|   89.6   	|
|    ours   	| **96.4** 	|

For ImageNet-30,
|         Method        	|   AUROC  	|
|:---------------------:	|:--------:	|
|          Rot          	|   65.3   	|
|       Rot+Trans       	|   77.9   	|
|        Rot+Attn       	|   81.6   	|
|     Rot+Trans+Attn    	|   84.8   	|
| Rot+Trans+Attn+Resize 	|   85.7   	|
|          CSI          	|   91.6   	|
|          ours         	| **92.0** 	|

## Citation

If you find this repository useful, please consider citing our work:
```

```

## Acknowledgement

This repository is built using the [beit](https://github.com/microsoft/unilm/tree/master/beit) library and the [SSD](https://github.com/inspire-group/SSD) repository.

