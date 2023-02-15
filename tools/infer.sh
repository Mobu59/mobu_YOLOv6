#!/bin/bash
python tools/infer.py \
--weights /world/data-gpu-94/liyang/Github_projects/YOLOv6/runs/train/head_hs_weights/weights/yolov6t_head_det_nonorm/last_ckpt.pt \
--source /world/data-gpu-94/liyang/pedDetection/test_data/daily_data/太原钟楼街直营店_1F旗舰机台东北-新_20220923175900_20220923190001.mp4 \
--save-txt \
--conf-thres 0.4 \
--hide-conf
#--hide-labels
#--save-img \

#--weights runs/train/exp/weights/yolov6t_head_det.v1/last_ckpt.pt \
#--source /world/data-gpu-94/liyang/pedDetection/head_detection/badcase/ped_head.badcase.gt.expansion.json \

#--source /world/data-gpu-94/liyang/pedDetection/test_data/daily_data/20230214_test_data.json \
#--source /world/data-gpu-94/liyang/pedDetection/test_data/daily_data/20230215_0630-1000_test_data.json \

#--source /world/data-gpu-94/wyq/mobile_video_data/demo2020/hw_ped/ped3.mp4 \
#--source /world/data-gpu-94/liyang/pedDetection/head_detection/badcase/ped3_badcase_frames.fixed.label.expansion.json \
#--source /world/data-gpu-94/wyq/mobile_video_data/demo2020/huigou/pass3.mp4 \
#--source /world/data-gpu-94/liyang/pedDetection/head_detection/badcase/test.json \ #用于可视化
#--source /world/data-gpu-94/ped_detection_data/biped.v8.head.mix.shuf.test.json \
