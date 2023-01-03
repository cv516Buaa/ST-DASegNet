_base_ = [
    '../../../configs/_base_/datasets/vp_daseg.py', '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_forSTDASegNet',
    pretrained='open-mmlab://resnet50_v1c',
     dsk_neck=dict(
        type='ML_DSKNeck',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        r=8,
        L=8,
        dsk_type='ds2'),
    backbone_s=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_t=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head_s=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2304,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        #c1_in_channels=0,
        #c1_channels=0,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])),
    decode_head_t=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2304,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        #c1_in_channels=0,
        #c1_channels=0,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])),
    discriminator_s=dict(
        type='AdapSegDiscriminator',
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=dict(type='IN'),
        in_channels=2048),
    discriminator_t=dict(
        type='AdapSegDiscriminator',
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.005),
        norm_cfg=dict(type='IN'),
        in_channels=2048),
     cross_EMA = dict(
        ## two types: 'single_t', 'decoder_only_t'
        type='decoder_only_t',
        training_ratio=0.25,
        decay=0.999,
        pseudo_threshold=0.975,
        pseudo_rare_threshold=0.8,
        pseudo_class_weight=[1.01, 1.01, 1.51, 1.51, 2.01, 2.01],
        backbone_EMA=dict(
                type='ResNetV1c',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                contract_dilation=True),
        decode_head_EMA=dict(
        type='DepthwiseSeparableASPPHead',
                in_channels=2304,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=256,
                c1_channels=48,
                #c1_in_channels=0,
                #c1_channels=0,
                dropout_ratio=0.1,
                num_classes=6,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5]))
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

# optimizer setting
optimizer = dict(
    backbone_s=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    backbone_t=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    dsk_neck=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    decode_head_s=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005),
    decode_head_t=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005),
    discriminator_s=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99)),
    discriminator_t=dict(type='Adam', lr=0.00025, betas=(0.9, 0.99))
    )

work_dir = './experiments/deeplabv3/ST-DASegNet_results/'
data = dict(samples_per_gpu=3, workers_per_gpu=3)
total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
runner = None
find_unused_parameters = True
