trt_version=7.2.1.6
cuda_version=10.2
cudnn_version=8.0
#trt_version=8.0.3.4
#cuda_version=11.3.1
#cudnn_version=11.3


trt_lib=/world/data-gpu-94/wyq/sonic-3rdparty/tensorrt/${trt_version}/lib:/world/data-gpu-94/wyq/sonic-3rdparty/cuda/${cuda_version}/lib64:/world/data-gpu-94/wyq/sonic-3rdparty/cudnn/${cudnn_version}/lib64
export LD_LIBRARY_PATH=${trt_lib}
trt_bin=/world/data-gpu-94/wyq/sonic-3rdparty/tensorrt/${trt_version}/bin/trtexec
onnx_model=/world/data-gpu-94/liyang/Github_projects/YOLOv6/runs/train/exp/weights/best_ckpt.onnx
save_path=/world/data-gpu-94/liyang/Github_projects/YOLOv6/runs/train/exp/weights
${trt_bin} \
    --onnx=$onnx_model \
    --saveEngine=$save_path/hs_det_${trt_version}.trt \
    --explicitBatch=1 \
    --batch=1 \
    --verbose

