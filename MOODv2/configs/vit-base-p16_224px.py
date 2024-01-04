model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
    ))


