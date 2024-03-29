model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BEiTViT',
        arch='base',
        img_size=224,
        patch_size=16,
        out_type='avg_featmap',
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False,
        ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        ),
    )
