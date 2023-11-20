_base_ = [
    '../../../configs/_base_/datasets/SR_daseg.py', '../../../configs/_base_/default_runtime.py',
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_forSTDASegNet',
    pretrained=None,
    dsk_neck=dict(
        type='ML_DSKNeck',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        r=8,
        L=8,
        dsk_type='ds2'),
    backbone_s=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    backbone_t=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head_s=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 576],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.25, 2.5, 1.25, 0.75])),
    decode_head_t=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 576],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.25, 2.5, 1.25, 0.75])),
    discriminator_s=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=dict(type='IN'),
        in_channels=512),
    discriminator_t=dict(
        type='AdapSegDiscriminator',
        num_conv=2,
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=dict(type='IN'),
        in_channels=512),
    cross_EMA = dict(
        ## three types: 'single_t', 'decoder_only_t', 'whole', 'single_decoder'
        #type='decoder_only_t',
        #decay=0.999,
        #pseudo_threshold=0.95,
        type='decoder_only_t',
        training_ratio=0.25,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=[1.01, 1.51, 1.51],
        backbone_EMA=dict(
            type='MixVisionTransformer',
            init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 6, 40, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),
        decode_head_EMA=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 576],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=4,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.25, 2.5, 1.25, 0.75]))
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

work_dir = './experiments/segformerb5/results_S2R/'

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU', pre_eval=True)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3)

# optimizer setting
optimizer = dict(
    backbone_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    backbone_t=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    dsk_neck=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01),
    decode_head_s=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    decode_head_t=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    discriminator_s=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99)),
    discriminator_t=dict(type='Adam', lr=0.00001, betas=(0.9, 0.99)))

runner = None
#use_ddp_wrapper = True
find_unused_parameters = True
