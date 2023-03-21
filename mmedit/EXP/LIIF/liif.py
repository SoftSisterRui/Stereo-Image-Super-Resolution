exp_name = 'liif'
scale_min = 1
scale_max = 4
model = dict(
    type='LIIF',
    generator=dict(
        type='LIIFRDN',
        encoder=dict(
            type='RDN',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=4,
            num_layers=8,
            channel_growth=64),
        imnet=dict(
            type='MLPRefiner',
            in_dim=64,
            out_dim=3,
            hidden_list=[256, 256, 256, 256]),
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
        eval_bsize=30000),
    rgb_mean=(0.5, 0.5, 0.5),
    rgb_std=(0.5, 0.5, 0.5),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=4)
train_dataset_type = 'SRFolderGTDataset'
val_dataset_type = 'SRFolderGTDataset'
test_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RandomDownSampling', scale_min=1, scale_max=4, patch_size=48),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', sample_quantity=2304),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]
valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RandomDownSampling', scale_min=4, scale_max=4),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell'),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', scale=4),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]
data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=8, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=20,
        dataset=dict(
            type='SRFolderGTDataset',
            gt_folder='data/Flickr1024_train/HR',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    io_backend='disk',
                    key='gt',
                    flag='color',
                    channel_order='rgb'),
                dict(
                    type='RandomDownSampling',
                    scale_min=1,
                    scale_max=4,
                    patch_size=48),
                dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='horizontal'),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='vertical'),
                dict(
                    type='RandomTransposeHW',
                    keys=['lq', 'gt'],
                    transpose_ratio=0.5),
                dict(type='ImageToTensor', keys=['lq', 'gt']),
                dict(type='GenerateCoordinateAndCell', sample_quantity=2304),
                dict(
                    type='Collect',
                    keys=['lq', 'gt', 'coord', 'cell'],
                    meta_keys=['gt_path'])
            ],
            scale=4)),
    val=dict(
        type='SRFolderGTDataset',
        gt_folder='data/Flickr1024_val/HR',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='color',
                channel_order='rgb'),
            dict(type='RandomDownSampling', scale_min=4, scale_max=4),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(type='ImageToTensor', keys=['lq', 'gt']),
            dict(type='GenerateCoordinateAndCell'),
            dict(
                type='Collect',
                keys=['lq', 'gt', 'coord', 'cell'],
                meta_keys=['gt_path'])
        ],
        scale=4),
    test=dict(
        type='SRFolderDataset',
        lq_folder='data/Flickr1024_val/LR',
        gt_folder='data/Flickr1024_val/HR',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='color',
                channel_order='rgb'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='color',
                channel_order='rgb'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(type='ImageToTensor', keys=['lq', 'gt']),
            dict(type='GenerateCoordinateAndCell', scale=4),
            dict(
                type='Collect',
                keys=['lq', 'gt', 'coord', 'cell'],
                meta_keys=['gt_path'])
        ],
        scale=4,
        filename_tmpl='{}'))
optimizers = dict(type='Adam', lr=0.0001)
iter_per_epoch = 1000
total_iters = 1000000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=0.5)
checkpoint_config = dict(interval=3000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=3000, save_image=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'EXP/LIIF'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
gpus = 1
