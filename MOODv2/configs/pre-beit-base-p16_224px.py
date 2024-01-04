# model settings
model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTPretrainViT',
        arch='base',
        patch_size=16,
        drop_path_rate=0.1,
        final_norm=True,
        out_type='raw',
        layer_scale_init_value=0.1,
        ),
    neck=None,
    head=dict(
        type='BEiTV1Head',
        embed_dims=768,
        num_embed=8192,
        loss=dict(type='CrossEntropyLoss'),
    )
)