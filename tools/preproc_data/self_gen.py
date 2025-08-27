from pathlib import Path
from datasets import Dataset
import json
from collections import defaultdict
from transformers import AutoTokenizer


def filter_right_dialog(details):
    messages = []
    for dialog in details.values():
        if dialog["predictions"] != dialog["references"]:
            continue
        cur_msg = [
            {"role": "user", "content": dialog["prompt"][0]["prompt"]},
            {"role": "assistant", "content": dialog["origin_prediction"]},
        ]
        messages.append(cur_msg)
    return messages


def proc_ceval(pred_res_dir, save_root, sampled_budget, tokenizer_path):
    cnt = defaultdict(int)
    messages = []
    for res_path in pred_res_dir.glob("ceval*.json"):
        pred_res = json.load(res_path.open())["details"]
        filtered_msg = filter_right_dialog(pred_res)
        print(
            f"{res_path.stem}: total {len(pred_res)}, filtered {len(filtered_msg)}, {len(filtered_msg) / len(pred_res):.2%}"
        )
        cnt["all"] += len(pred_res)
        cnt["filtered"] += len(filtered_msg)
        messages.extend(filtered_msg)

    print(
        f"totally find {len(messages)} valid messages, {len(messages) / cnt['all']:.2%}"
    )
    ds = Dataset.from_dict({"messages": messages})
    if len(ds) > sampled_budget:
        ds = ds.shuffle().select(range(sampled_budget))

    save_dir = save_root / "ceval"
    ds.save_to_disk(save_dir)
    print(f"save to {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    texts = tokenizer.apply_chat_template(
        ds["messages"], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(texts).input_ids
    print(f"check done")


def proc_mmlu_pro(pred_res_dir, save_root, sampled_budget, tokenizer_path):
    cnt = defaultdict(int)
    messages = []
    for res_path in pred_res_dir.glob("mmlu_pro*.json"):
        pred_res = json.load(res_path.open())["details"]
        filtered_msg = filter_right_dialog(pred_res)
        print(
            f"{res_path.stem}: total {len(pred_res)}, filtered {len(filtered_msg)}, {len(filtered_msg) / len(pred_res):.2%}"
        )
        cnt["all"] += len(pred_res)
        cnt["filtered"] += len(filtered_msg)
        messages.extend(filtered_msg)

    print(
        f"totally find {len(messages)} valid messages, {len(messages) / cnt['all']:.2%}"
    )
    ds = Dataset.from_dict({"messages": messages})
    if len(ds) > sampled_budget:
        ds = ds.shuffle().select(range(sampled_budget))

    save_dir = save_root / "mmlu_pro"
    ds.save_to_disk(save_dir)
    print(f"save to {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    texts = tokenizer.apply_chat_template(
        ds["messages"], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(texts).input_ids
    print(f"check done")


if __name__ == "__main__":
    pred_res_dir = Path(
        "/ms/FM/chenshuailin/code/opencompass_0303_bk/outputs/data4k/eval_0303_4k_qwen3.1.7B/20250306_122735/results/Qwen3"
    )
    save_root = Path("data/self_gen")
    sampled_budget = 10240
    tokenizer_path = "/data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B"

    # proc_ceval(pred_res_dir, save_root, sampled_budget, tokenizer_path)
    proc_mmlu_pro(pred_res_dir, save_root, sampled_budget, tokenizer_path)
    print("done")
