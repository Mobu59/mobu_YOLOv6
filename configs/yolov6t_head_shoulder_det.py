# YOLOv6tiny head_det model
model = dict(
    type='YOLOv6t',
    pretrained='/world/data-gpu-94/liyang/Github_projects/YOLOv6/pretrain_weights/yolov6t.pt',
    depth_multiple=0.25,
    width_multiple=0.50,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        ),
    neck=dict(
        type='RepPAN',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        #out_channels=[256, 256, 256, 512, 512, 1024],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=1,
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        iou_type='ciou'
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.0032,
    lrf=0.12,
    momentum=0.843,
    weight_decay=0.00036,
    warmup_epochs=2.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.05
)

data_aug = dict(
    hsv_h=0.0138,
    hsv_s=0.664,
    hsv_v=0.464,
    degrees=0.373,
    #translate=0.245,
    translate=0.1,
    #scale=0.898,
    scale=0.2,
    #shear=0.602,
    shear=2.0,
    flipud=0.00856,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
    #mosaic=1.0,
    #mixup=0.243,
    #mixup=1.0,
)
