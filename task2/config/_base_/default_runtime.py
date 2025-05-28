default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='TensorboardVisBackend')],
    name='visualizer')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook',  # 添加TensorBoard记录
            #  log_dir='/home/sophiehhh/cv-midterm/task2.0/work_dirs/tensorboard',  # 日志目录
             interval=50)  # 记录间隔
    ])

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False


workflow = [('train', 1), ('val', 1)]