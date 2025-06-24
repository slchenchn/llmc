set -x

TP=${1:-8}


MODEL=$(readlink -f $(dirname $0))
if [ -d "$MODEL/vllm_quant_model" ]; then
    MODEL="$MODEL/vllm_quant_model"
elif [ -d "$MODEL/autoawq_quant_model" ]; then
    MODEL="$MODEL/autoawq_quant_model"
fi

python3 -m sglang.launch_server \
    --model $MODEL \
    --tp $TP \
    --trust-remote-code \
    --disable-cuda-graph \
    --port 30000 \
    --host 0.0.0.0 \
    ${@:2}

    # --enable-torch-compile \
    # --torch-compile-max-bs 8 \