set -x

# CUDA_VISIBLE_DEVICES=0 python tools/eval_ppl.py \
#     --model_path /data/chenshuailin/checkpoints/Qwen/Qwen3-8B \
#     --model_name Qwen3-8B &

# CUDA_VISIBLE_DEVICES=1 python tools/eval_ppl.py \
#     --model_path checkpoints/Qwen3-8B/gptq/quarot_gptq_w8a8_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-8B_quarot_gptq_w8a8_sym &

# CUDA_VISIBLE_DEVICES=2 python tools/eval_ppl.py \
#     --model_path checkpoints/Qwen3-8B/gptq/quarot_gptq_w4a8_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-8B_quarot_gptq_w4a8_sym &

CUDA_VISIBLE_DEVICES=5 python tools/eval_ppl.py \
    --model_path checkpoints/Qwen3-8B/gptq/quarot_gptq_w4a8_g64_sym_dynamic/fake_quant_model \
    --model_name Qwen3-8B_quarot_gptq_w4a8_g64 &
