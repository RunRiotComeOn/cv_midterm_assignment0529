# training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  
    val_interval=1  
)

val_cfg = dict(
    type='ValLoop',
)

test_cfg = dict(type='TestLoop')

# 优化学习率策略
param_scheduler = [
    # Warmup 阶段（2000 次迭代）
    dict(
        type='LinearLR',
        start_factor=0.001,  # 初始学习率较小
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=2000  # 保持 Warmup 阶段
    ),
    # 使用 CosineAnnealingLR 平滑衰减学习率
    dict(
        type='CosineAnnealingLR',
        T_max=50, 
        eta_min=5e-6,  # 提高最小学习率，保持后期学习能力
        by_epoch=True,
        begin=0,
        end=50
    )
]

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',  # Adam 优化器
        lr=0.0002,  
        betas=(0.9, 0.999), 
        weight_decay=0.0005),
    clip_grad=dict(
        max_norm=10.0,  # 保持梯度裁剪
        norm_type=2
    )
)

# 自动学习率缩放
auto_scale_lr = dict(
    enable=True,
    base_batch_size=16
)
