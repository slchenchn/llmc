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


MASTER_ADDR=127.0.0.1
MASTER_PORT=$PORT
task_id=$PORT

CFG=${1}
task_name=$(basename $CFG .yml)
timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${2:-logs}
log_path=${LOG_DIR}/${task_name}_${timestamp}.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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