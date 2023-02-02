#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py \
--batch=64 \
--conf=configs/yolov6m_finetune.py \
--data=data/hands_goods_det.yaml \
--fuse_ab \
--device=3,4,5,7 \
--img-size=416 \
--epochs=50 
#--conf=configs/yolov6t_head_shoulder_det.py \
#--data=data/head_det.yaml \
#--conf=/world/data-gpu-94/liyang/Github_projects/YOLOv6/configs/yolov6n_head_shoulder_det.py \
#--data=/world/data-gpu-94/liyang/Github_projects/YOLOv6/data/head_shoulder_det.yaml \
#python tools/train.py \
#--data=data/head_shoulder_det.yaml \


