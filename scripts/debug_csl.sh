set -x

llmc=$(dirname $0)/.. # llmc/scripts/csl_run_llmc.sh
export PYTHONPATH=$llmc:$PYTHONPATH

nnodes=1
nproc_per_node=1

PORT=${PORT:-29545}

MASTER_ADDR=127.0.0.1
MASTER_PORT=$PORT
task_id=$PORT

SAVE_MODEL_PATH=checkpoints/debug
rm -rf $SAVE_MODEL_PATH
CFG=${1:-configs/csl/debug.yml}
# CFG=${1:-configs/csl/smoothquant_gptq/sq_dsv2_sym_w8a8_static.yml}
task_name=$(basename $CFG .yml)
timestamp=$(date +"%Y%m%d_%H%M%S")
log_path=logs/${task_name}_${timestamp}.log

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1
mkdir -p logs
torchrun \
    --nnodes $nnodes \
    --master_port $MASTER_PORT \
    --nproc_per_node $nproc_per_node \
    ${llmc}/llmc/__main__.py \
    --config $CFG \
    --task_id $task_id \
    2>&1 | tee ${log_path}

# cp ${llmc}/scripts/sgl/run_sgl.sh ${SAVE_MODEL_PATH}/run_sgl.sh

    --debugpy \
##############################################################################
# --rdzv_id $task_id \
# --rdzv_backend c10d \
# --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
