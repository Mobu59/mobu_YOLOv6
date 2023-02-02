#!/bin/bash
python tools/infer.py \
--weights runs/train/exp/weights/yolov6t_head_det.v1/last_ckpt.pt \
--source /world/data-gpu-94/liyang/pedDetection/head_detection/badcase/ped_head.badcase.gt.expansion.json \
--save-txt \
--hide-labels
#--save-img \
#--source /world/data-gpu-94/wyq/mobile_video_data/demo2020/hw_ped/ped3.mp4 \
#--source /world/data-gpu-94/liyang/pedDetection/head_detection/badcase/ped3_badcase_frames.fixed.label.expansion.json \
#--source /world/data-gpu-94/wyq/mobile_video_data/demo2020/huigou/pass3.mp4 \
#--source /world/data-gpu-94/liyang/pedDetection/head_detection/badcase/test.json \ #用于可视化
#--source /world/data-gpu-94/ped_detection_data/biped.v8.head.mix.shuf.test.json \
