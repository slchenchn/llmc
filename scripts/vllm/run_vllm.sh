set -x

TP=${1:-8}

# export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/hpcx/ucx/lib
# export VLLM_USE_V1=0

MODEL=$(readlink -f $(dirname $0))
if [ -d "$MODEL/vllm_quant_model" ]; then
    MODEL="$MODEL/vllm_quant_model"
elif [ -d "$MODEL/autoawq_quant_model" ]; then
    MODEL="$MODEL/autoawq_quant_model"
fi

vllm serve $MODEL \
    --tensor-parallel-size $TP \
    --max-model-len 32768 \
    --trust-remote-code \
    --port 9000 \
    ${@:2}


    # --gpu-memory-utilization 0.85
