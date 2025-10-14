set -x
#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1

llmc=$(dirname $0)/..    # llmc/scripts/csl_run_llmc.sh
export PYTHONPATH=$llmc:$PYTHONPATH


nnodes=1
nproc_per_node=1


# find_unused_port() {
#     while true; do
#         port=$(shuf -i 10000-60000 -n 1)
#         if ! ss -tuln | grep -q ":$port "; then
#             echo "$port"
#             return 0
#         fi
#     done
# }
# UNUSED_PORT=$(find_unused_port)
PORT=${PORT:-29556}



# # proxy&wandb
# export PROXY_ADDRESS="192.168.34.17"
# export PROXY_PORT="10808"

# export ALL_PROXY="http://$PROXY_ADDRESS:$PROXY_PORT"
# # export ALL_PROXY="socks5://$PROXY_ADDRESS:$PROXY_PORT"
# export HTTP_PROXY=$ALL_PROXY
# export HTTPS_PROXY=$ALL_PROXY
# echo "代理已启用: $ALL_PROXY"

# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # 替换成你自己的key


MASTER_ADDR=127.0.0.1
MASTER_PORT=$PORT
task_id=$PORT

CFG=${1}
task_name=$(basename $CFG .yml)
timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${2:-logs}
log_path=${LOG_DIR}/${task_name}_${timestamp}.log


mkdir -p ${LOG_DIR}
torchrun \
--nnodes $nnodes \
--master_port $MASTER_PORT \
--nproc_per_node $nproc_per_node \
${llmc}/llmc/__main__.py \
--config $CFG \
--task_id $task_id \
2>&1 | tee ${log_path}


--debugpy \
# --rdzv_id $task_id \
# --rdzv_backend c10d \
# --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \