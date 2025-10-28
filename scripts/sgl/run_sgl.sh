set -x

TP=${1:-8}
shift || true


MODEL=$(readlink -f $(dirname $0))
if [ -d "$MODEL/vllm_quant_model" ]; then
    MODEL="$MODEL/vllm_quant_model"
elif [ -d "$MODEL/autoawq_quant_model" ]; then
    MODEL="$MODEL/autoawq_quant_model"
fi

# Determine visible GPU indices
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t ALL_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi

NUM_GPUS=${#ALL_GPUS[@]}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs detected."
    exit 1
fi

if [ "$TP" -le 0 ]; then
    echo "Invalid TP: $TP"
    exit 1
fi

NUM_INSTANCES=$(( NUM_GPUS / TP ))
if [ "$NUM_INSTANCES" -lt 1 ]; then
    echo "Not enough GPUs ($NUM_GPUS) for TP=$TP"
    exit 1
fi

BASE_PORT=${BASE_PORT:-8200}
HOST=${HOST:-0.0.0.0}

for (( i=0; i<NUM_INSTANCES; i++ )); do
    start=$(( i * TP ))
    end=$(( start + TP - 1 ))
    SLICE=("${ALL_GPUS[@]:start:TP}")
    GPU_LIST=$(IFS=, ; echo "${SLICE[*]}")
    PORT=$(( BASE_PORT + i ))

    echo "Starting instance $i on GPUs [$GPU_LIST], port $PORT"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" python3 -m sglang.launch_server \
        --model "$MODEL" \
        --tp "$TP" \
        --trust-remote-code \
        --enable-torch-compile \
        --torch-compile-max-bs 64 \
        --port "$PORT" \
        --host "$HOST" \
        "$@" &
done

wait