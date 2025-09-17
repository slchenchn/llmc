set -x

# rm -rf /data/chenshuailin/checkpoints/llmc/DeepSeek-R1/quarot/fp8_sym_w8a8-dynamic_ol-rotate/fake_quant_model
# bash scripts/csl_run.sh configs/csl/quarot_gptq_ol-rotate/fp8_quarot_r1_sym_w8a8_dynamic_ol-rotate.yml

N_SAMPLES=16
OUTPUT_ROOT=analysis_model
DS_NAME=pileval
DS_PATH=data/${DS_NAME}
MODEL_TYPE=Qwen2
SEQ_LEN=2048

############################# Qwen2.5-3B-Instruct #############################
CUDA_VISIBLE_DEVICES=1 python tools/quant_analysis/single_model_analysis.py \
    --n_samples $N_SAMPLES \
    --output_dir $OUTPUT_ROOT/Qwen2.5-3B-Instruct/bf16 \
    --dataset_name ${DS_NAME} \
    --data_path ${DS_PATH} \
    --model_type ${MODEL_TYPE} \
    --model_path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct \
    --seq_len ${SEQ_LEN} &

# CUDA_VISIBLE_DEVICES=2 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen2.5-3B-Instruct/bf16 \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path /data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B \
#     --seq_len ${SEQ_LEN} &

# CUDA_VISIBLE_DEVICES=1 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen2.5-3B-Instruct/quarot \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path checkpoints/Qwen2.5-3B-Instruct/quarot/sym_w8_a8-dynamic/transformed_model \
#     --seq_len ${SEQ_LEN} &
