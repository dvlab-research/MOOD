model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTPretrainViT',
        arch='base',
        patch_size=16,
        out_indices=[-4, -1],
        final_norm=False,
        out_type='raw',),
    neck=dict(
        type='BEiTV2Neck',
        num_layers=2,
        early_layers=9,
        backbone_arch='base',
    ),
    head=dict(
        type='BEiTV2Head',
        embed_dims=768,
        num_embed=8192,
        loss=dict(type='CrossEntropyLoss')),
)

