import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
import os
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="The model id of a pretrained model from huggingface.",
        default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="The directory to save the quantized model.",
        default="checkpoints/Qwen2.5-3B-Instruct/gptqv2/llmcompressor_nvfp4_w4a16_sgs/vllm_nvfp4_quant_model",
    )
    parser.add_argument(
        "--max_seq_len",
        type=str,
        help="The maximum sequence length.",
        default=2048,
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        help="The number of calibration samples.",
        default=128,
    )
    parser.add_argument("--method", type=str, default="gptq")
    parser.add_argument("--a_bit", default=16, type=int)
    args = parser.parse_args()

    return args


def get_data(tokenizer, num_calib_samples, max_seq_len):
    # DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train"

    # Load dataset and preprocess.
    # ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    # ds = load_dataset("json", data_files="/ms/FM/tangqie/0723/LLM-QAT/zh_c4_sampled_0724.jsonl", split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = load_dataset(
        "json",
        data_files="data/tulu_alpaca/QAT_mix_tulu_alpaca_50k_qwen25_3b_distill_v2.jsonl",
        split=f"{DATASET_SPLIT}[:{num_calib_samples}]",
    )

    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # Tokenize inputs.
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize)
    ds = ds.remove_columns(["messages", "text", "idx"])
    return ds


if __name__ == "__main__":
    args = get_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    ds = get_data(tokenizer, args.n_calib_samples, args.max_seq_len)

    print(ds[0])
    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp4 with per group 16 via ptq
    #   * calibrate a global_scale for activations, which will be used to
    #       quantize activations to fp4 on the fly
    print(model)

    if args.a_bit == 4:
        scheme = "NVFP4"
    elif args.a_bit in [16, 32]:
        scheme = "NVFP4A16"
    else:
        raise ValueError(f"Invalid activation bit: {args.a_bit}")

    print(f"using scheme {scheme}")
    quant_args = dict(targets="Linear", scheme=scheme, ignore=["lm_head"])
    quant_method = args.method.lower()
    if quant_method == "gptq":
        recipe = GPTQModifier(**quant_args)
    elif quant_method == "rtn":
        recipe = QuantizationModifier(**quant_args)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    # Apply quantization.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.n_calib_samples,
    )

    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save to disk in compressed-tensors format.
    model.save_pretrained(args.save_dir, save_compressed=True)
    tokenizer.save_pretrained(args.save_dir)
    print(f"model saved to {args.save_dir}")
