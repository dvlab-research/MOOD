model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='huge',
        img_size=448,
        patch_size=14,
        drop_path_rate=0.3,  # set to 0.3
        out_type='avg_featmap',
        final_norm=False,),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
    )
)