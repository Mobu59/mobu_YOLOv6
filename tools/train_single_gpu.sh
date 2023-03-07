#!/bin/bash
python tools/train.py \
--batch=16 \
--conf=configs/yolov6m_finetune.py \
--data=data/sens_det.yaml \
--fuse_ab \
--workers=4 \
--device=0 \
--img-size=416 \
--epochs=20 

#--data=data/hands_goods_det.yaml \
#--eval-interval 2 \
#--conf=configs/yolov6t_head_shoulder_det.py \
#--data=data/head_det.yaml \
#--conf=/world/data-gpu-94/liyang/Github_projects/YOLOv6/configs/yolov6n_head_shoulder_det.py \
#--data=/world/data-gpu-94/liyang/Github_projects/YOLOv6/data/head_shoulder_det.yaml \
#python tools/train.py \
#--data=data/head_shoulder_det.yaml \


