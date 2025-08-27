set -x

# rm -rf /data/chenshuailin/checkpoints/llmc/DeepSeek-R1/quarot/fp8_sym_w8a8-dynamic_ol-rotate/fake_quant_model
# bash scripts/csl_run.sh configs/csl/quarot_gptq_ol-rotate/fp8_quarot_r1_sym_w8a8_dynamic_ol-rotate.yml

N_SAMPLES=32
OUTPUT_ROOT=analysis_model
DS_NAME=pileval
DS_PATH=data/${DS_NAME}
MODEL_TYPE=Qwen2
SEQ_LEN=2048


# ############################# Qwen3-0.6B #############################
# CUDA_VISIBLE_DEVICES=0 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen3-0.6B/bf16 \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path /data/chenshuailin/checkpoints/Qwen/Qwen3-0.6B \
#     --seq_len ${SEQ_LEN} &

CUDA_VISIBLE_DEVICES=0 python tools/quant_analysis/single_model_analysis.py \
    --n_samples $N_SAMPLES \
    --output_dir $OUTPUT_ROOT/Qwen3-0.6B/quarot \
    --dataset_name ${DS_NAME} \
    --data_path ${DS_PATH} \
    --model_type ${MODEL_TYPE} \
    --model_path checkpoints/Qwen3-0.6B/quarot/sym_w4_a8-dynamic/transformed_model \
    --seq_len ${SEQ_LEN} &


# ############################# Qwen3-1.7B #############################
# CUDA_VISIBLE_DEVICES=0 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen3-1.7B/bf16 \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path /data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B \
#     --seq_len ${SEQ_LEN} &

# CUDA_VISIBLE_DEVICES=1 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen3-1.7B/quarot \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path checkpoints/Qwen3-1.7B/quarot/sym_w8_a8-dynamic/transformed_model \
#     --seq_len ${SEQ_LEN} &



# ############################# Qwen3-8B #############################
# CUDA_VISIBLE_DEVICES=0 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen3-8B/bf16 \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path /data/chenshuailin/checkpoints/Qwen/Qwen3-8B \
#     --seq_len ${SEQ_LEN} &

# CUDA_VISIBLE_DEVICES=1 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen3-8B/quarot \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path checkpoints/Qwen3-8B/quarot/sym_w8_a8-dynamic/transformed_model \
#     --seq_len ${SEQ_LEN} &


############################# Qwen3-32B #############################
# CUDA_VISIBLE_DEVICES=1 python tools/quant_analysis/single_model_analysis.py \
#     --n_samples $N_SAMPLES \
#     --output_dir $OUTPUT_ROOT/Qwen3-32B/bf16 \
#     --dataset_name ${DS_NAME} \
#     --data_path ${DS_PATH} \
#     --model_type ${MODEL_TYPE} \
#     --model_path /data/chenshuailin/checkpoints/Qwen/Qwen3-32B \
#     --seq_len ${SEQ_LEN} &

CUDA_VISIBLE_DEVICES=1 python tools/quant_analysis/single_model_analysis.py \
    --n_samples $N_SAMPLES \
    --output_dir $OUTPUT_ROOT/Qwen3-32B/quarot \
    --dataset_name ${DS_NAME} \
    --data_path ${DS_PATH} \
    --model_type ${MODEL_TYPE} \
    --model_path checkpoints/Qwen3-32B/quarot/sym_w8_a8-dynamic/transformed_model \
    --seq_len ${SEQ_LEN} &
