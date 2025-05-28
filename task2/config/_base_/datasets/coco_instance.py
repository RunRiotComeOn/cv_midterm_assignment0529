# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
data_root = '/home/huangyixu/data/VOCdevkit/VOC2012/'

metainfo = {
    'classes': (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)
    ]
}

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

###########
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs'),
]  

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_train2017.json',
        ann_file='annotations/annotations_train.json',
        # ann_file = 'subset_annotations/subset_train.json',
        # data_prefix=dict(img='train2017/'),
        data_prefix=dict(img='JPEGImages/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_val2017.json',
        ann_file='annotations/annotations_val.json',
        # ann_file = 'subset_annotations/subset_val.json',
        # data_prefix=dict(img='val2017/'),
        test_mode=True,
        data_prefix=dict(img='JPEGImages/'),
        metainfo=metainfo,
        # pipeline=test_pipeline,
        pipeline=val_pipeline,
        backend_args=backend_args))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_val2017.json',
        ann_file='annotations/annotations_test.json',
        # ann_file = 'subset_annotations/subset_test.json',
        # data_prefix=dict(img='val2017/'),
        test_mode=True,
        data_prefix=dict(img='JPEGImages/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + 'annotations/annotations_val.json',
    # ann_file = data_root + 'subset_annotations/subset_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
# test_evaluator = val_evaluator
test_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + 'annotations/annotations_test.json',
    # ann_file = data_root + 'subset_annotations/subset_test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric=['bbox', 'segm'],
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_instance/test')
