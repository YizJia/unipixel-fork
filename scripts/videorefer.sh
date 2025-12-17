#!/bin/bash

set -e

command -v npu-smi &>/dev/null && nproc=$(npu-smi info -l | grep "NPU ID" | wc -l) || nproc=$(nvidia-smi --list-gpus | wc -l)

export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((nproc-1)))
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

LOG_DIR="logs/videorefer_bench_d_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "log will be save in: $LOG_DIR"
ckpt_path="model_zoo/UniPixel-3B"
OUTPUT_DIR="outputs"

# ckpt_path=$1

# ===========================================================================

# IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
# CHUNKS=${#GPULIST[@]}

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     # define the log name
#     LOG_FILE="$LOG_DIR/gpu_${GPULIST[$IDX]}.log"
    
#     echo "launching taks on GPU ${GPULIST[$IDX]}... log file: $LOG_FILE"
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_videorefer_q.py \
#         --dataset videorefer_bench_q \
#         --split test \
#         --model_path $ckpt_path \
#         --res_pred_path $OUTPUT_DIR/videorefer_bench_q_single \
#         --vis_pred_path $OUTPUT_DIR/videorefer_bench_q_single_vis \
#         --single_frame_mode \
#         --chunk $CHUNKS \
#         --index $IDX \
#         --dump 100 \
#         > "$LOG_FILE" 2>&1 &
# done

# echo "All tasks are launched, waiting for completing..."
# wait
# echo "All tasks compledted!"

# python unipixel/eval/eval_general.py $OUTPUT_DIR/videorefer_bench_q_single

# ===========================================================================

# IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
# CHUNKS=${#GPULIST[@]}

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     # define the log name
#     LOG_FILE="$LOG_DIR/gpu_${GPULIST[$IDX]}.log"
    
#     echo "launching taks on GPU ${GPULIST[$IDX]}... log file: $LOG_FILE"
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python -u unipixel/eval/infer_videorefer_q.py \
#         --dataset videorefer_bench_q \
#         --split test \
#         --model_path $ckpt_path \
#         --res_pred_path $OUTPUT_DIR/videorefer_bench_q \
#         --vis_pred_path $OUTPUT_DIR/videorefer_bench_q_vis \
#         --chunk $CHUNKS \
#         --index $IDX \
#         --dump 100 \
#         > "$LOG_FILE" 2>&1 &
# done

# echo "All tasks are launched, waiting for completing..."
# wait
# echo "All tasks compledted!"

# python unipixel/eval/eval_general.py $OUTPUT_DIR/videorefer_bench_q

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    # define the log name
    LOG_FILE="$LOG_DIR/gpu_${GPULIST[$IDX]}.log"
    
    echo "launching taks on GPU ${GPULIST[$IDX]}... log file: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_videorefer_d.py \
        --dataset videorefer_bench_d \
        --split test \
        --model_path $ckpt_path \
        --res_pred_path $OUTPUT_DIR/videorefer_bench_d \
        --vis_pred_path $OUTPUT_DIR/videorefer_bench_d_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --dump 100 \
        > "$LOG_FILE" 2>&1 &
done

echo "All tasks are launched, waiting for completing..."
wait
echo "All tasks compledted!"