# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', 
        arch='base', 
        img_size=224, 
        stage_cfgs=dict(block_cfgs=dict(window_size=7)),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        cal_acc=False
    ),
)
