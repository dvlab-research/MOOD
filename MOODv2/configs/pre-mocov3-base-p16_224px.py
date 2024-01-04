# model settings
temperature = 0.2
model = dict(
    type='MoCoV3',
    base_momentum=0.01,
    backbone=dict(
        type='MoCoV3ViT',
        arch='base',  # embed_dim = 768
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=768,
        hid_channels=4096,
        out_channels=256,
        num_layers=3,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=False),
    head=dict(
        type='MoCoV3Head',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=True,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        loss=dict(type='CrossEntropyLoss', loss_weight=2 * temperature),
        temperature=temperature,))

