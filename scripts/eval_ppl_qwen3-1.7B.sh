set -x

# CUDA_VISIBLE_DEVICES=0 python tools/w4_analysis/eval_ppl.py \
#     --model_path /data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B \
#     --model_name Qwen3-1.7B &

# CUDA_VISIBLE_DEVICES=1 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_fa_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_fa_sym &

# CUDA_VISIBLE_DEVICES=2 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g2_asym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_g2_asym &

# CUDA_VISIBLE_DEVICES=3 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g2_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_g2_sym &

# CUDA_VISIBLE_DEVICES=4 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g8_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_g8_sym &

# CUDA_VISIBLE_DEVICES=5 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g64_dq_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_g64_dg &

wait

# CUDA_VISIBLE_DEVICES=0 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g64_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_g64_sym &

# CUDA_VISIBLE_DEVICES=1 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_sym &

# CUDA_VISIBLE_DEVICES=2 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w8a8_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w8a8_sym &

# CUDA_VISIBLE_DEVICES=3 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w8a8_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w8a8_sym &

    
# CUDA_VISIBLE_DEVICES=3 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g64_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_gptq_w4a8_g64_sym_small_update &


# CUDA_VISIBLE_DEVICES=4 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/gptq/quarot_sq_gptq_w4a8_g64_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_sq_gptq_w4a8_g64_sym &

# CUDA_VISIBLE_DEVICES=0 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/omniq/quarot_omniq_w4_sym_dynamic/fake_quant_model \
#     --model_name Qwen3-1.7B_quarot_omniq_w4_sym_fake &

    
# CUDA_VISIBLE_DEVICES=1 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/omniq/quarot_omniq_w4_sym_dynamic/transformed_model \
#     --model_name Qwen3-1.7B_quarot_omniq_w4_sym_transformed &


# CUDA_VISIBLE_DEVICES=2 python tools/w4_analysis/eval_ppl.py \
#     --model_path checkpoints/Qwen3-1.7B/omniq/quarot_omniq_w4_sym_dynamic/dequant_model \
#     --model_name Qwen3-1.7B_quarot_omniq_w4_sym_dequant &


        
CUDA_VISIBLE_DEVICES=3 python tools/w4_analysis/eval_ppl.py \
    --model_path checkpoints/Qwen3-1.7B/omniq/quarot_ol_rotate_omniq_v2_e5_w4a16_g128_sym_dynamic/fake_quant_model \
    --model_name Qwen3-1.7B_quarot_omniq_v3_e5_w4a16_g128_sym_fake &

    
CUDA_VISIBLE_DEVICES=1 python tools/w4_analysis/eval_ppl.py \
    --model_path checkpoints/Qwen3-1.7B/omniq/quarot_ol_rotate_omniq_v2_e5_w4a16_g128_sym_dynamic/transformed_model \
    --model_name Qwen3-1.7B_quarot_omniq_v3_e5_w4a16_g128_sym_transformed &


CUDA_VISIBLE_DEVICES=2 python tools/w4_analysis/eval_ppl.py \
    --model_path checkpoints/Qwen3-1.7B/omniq/quarot_ol_rotate_omniq_v2_e5_w4a16_g128_sym_dynamic/dequant_model \
    --model_name Qwen3-1.7B_quarot_omniq_v3_e5_w4a16_g128_sym_dequant &