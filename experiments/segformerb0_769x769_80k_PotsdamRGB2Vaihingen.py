_base_ = [
    '../configs/_base_/models/segformer_mit-b0.py',
    '../configs/_base_/datasets/potsdam2vaihingen.py', '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b0.pth')),
    decode_head=dict(
        num_classes=6,
    ),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
data = dict(samples_per_gpu=1, workers_per_gpu=1, train=dict(img_dir='Potsdam_RGB_DA/img_dir/train', ann_dir='Potsdam_RGB_DA/ann_dir/train', split='Potsdam_RGB_DA/train.txt'))
workflow = [('train', 1), ('val', 1)]
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
