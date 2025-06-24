
set -x

MODEL=$(readlink -f $(dirname $0)/vllm_quant_model)
MODEL=$(readlink -f $(dirname $0)/dequant_model)
# TP=${1:-1}


for GPU_ID in {0..7}; do
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m sglang.launch_server \
    --model $MODEL \
    --tp 1 \
    --trust-remote-code \
    --torch-compile-max-bs 64 \
    --port $((9000+GPU_ID)) \
    --host 0.0.0.0 &
done

wait
