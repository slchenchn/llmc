import json
import os
import random

import torch
from tqdm import tqdm
from loguru import logger

from llmc.utils.registry_factory import PREPROC_REGISTRY
from llmc.utils import warning_once


@PREPROC_REGISTRY
def wikitext2_gptq(calib_dataset, tokenizer, n_samples, seq_len, prefix_token_ids=None):
    assert prefix_token_ids is None, (
        "prefix_token_ids is not supported for wikitext2_gptq yet"
    )
    trainenc = tokenizer("\n\n".join(calib_dataset["text"]), return_tensors="pt")
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def ptb_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer(" ".join(calib_dataset["sentence"]), return_tensors="pt")
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def c4_gptq(calib_dataset, tokenizer, n_samples, seq_len, prefix_token_ids=None):
    if prefix_token_ids is not None:
        warning_once("prefix_token_ids is not supported for c4_gptq yet")
    samples = []
    calibtexts = [t for t in calib_dataset["text"] if len(t) > seq_len]
    all_ids = tokenizer(calibtexts, return_attention_mask=False).input_ids
    all_ids = [t for t in all_ids if len(t) >= seq_len]
    if len(all_ids) < n_samples:
        raise ValueError(
            f"Not enough samples to calibrate, {len(all_ids)=} < {n_samples=}"
        )
    for _ in range(n_samples):
        while True:
            cur_ids = all_ids[random.randint(0, len(all_ids) - 1)]
            if len(cur_ids) >= seq_len:
                break
        i = random.randint(0, len(cur_ids) - seq_len - 1)
        j = i + seq_len
        inp = torch.tensor(cur_ids[i:j], dtype=torch.int64).unsqueeze(0)
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def pileval_awq(calib_dataset, tokenizer, n_samples, seq_len, prefix_token_ids=None):
    if prefix_token_ids is not None:
        warning_once("prefix_token_ids is not supported for pileval_awq yet")
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    samples = torch.cat(samples, dim=1)
    n_split = samples.shape[1] // seq_len
    samples = [samples[:, i * seq_len : (i + 1) * seq_len] for i in range(n_split)]
    return samples


@PREPROC_REGISTRY
def pileval_smooth(calib_dataset, tokenizer, n_samples, seq_len):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        trainenc = tokenizer(
            line, return_tensors="pt", max_length=seq_len, truncation=True
        )
        line_encoded = trainenc.input_ids
        samples.append(line_encoded)
        n_run += 1
        if n_run == n_samples:
            break
    return samples


@PREPROC_REGISTRY
def pileval_omni(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer("\n\n".join(calib_dataset["text"][:1000]), return_tensors="pt")
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def img_general(calib_dataset, tokenizer, batch_process, n_samples):
    random.shuffle(calib_dataset)
    if len(calib_dataset) > n_samples:
        calib_dataset = calib_dataset[:n_samples]
    samples = batch_process(calib_dataset)
    return samples


@PREPROC_REGISTRY
def random_truncate_txt(calib_dataset, tokenizer, n_samples, seq_len):
    random.shuffle(calib_dataset)
    trainenc = tokenizer("\n\n".join(calib_dataset), return_tensors="pt")
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def ultrachat_general(
    calib_dataset, tokenizer, n_samples, seq_len, prefix_token_ids=None
):
    """
    对 ultrachat 格式的数据集进行预处理，生成用于校准的样本。处理流程：
    1. 过滤掉长度小于 seq_len 的数据；
    2. 若数据量不足 n_samples，则自动将长样本拆分为多个样本；
    3. 若拆分后仍然不足 n_samples，则抛出异常
    4. 截取每个样本的前 seq_len 个 token
    """
    if prefix_token_ids is not None:
        warning_once("prefix_token_ids is not supported for ultrachat_general yet")
    # calib_dataset = calib_dataset.shuffle(seed=42).select(range(n_samples))

    # 定义处理函数：检查长度并应用chat template
    def process_example(example):
        messages = example["messages"]
        str_len = sum(len(msg["content"]) for msg in messages)
        # 如果长度达标（>seq_len），则处理；否则标记为需要过滤
        if str_len > seq_len * 1:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            return {"formatted_text": text, "keep": True}
        else:
            return {"formatted_text": "", "keep": False}

    # 批量处理数据集
    processed_dataset = calib_dataset.map(
        process_example,
        desc="formatting ultrachat dataset",
        # num_proc=1,
    )

    # 过滤掉不达标的数据
    filtered_dataset = processed_dataset.filter(lambda x: x["keep"])

    # 提取格式化后的文本
    texts = filtered_dataset["formatted_text"]

    all_ids = tokenizer(texts, return_attention_mask=False).input_ids
    all_ids = [t for t in all_ids if len(t) >= seq_len]
    if len(all_ids) < n_samples:
        """ 如果超出n倍长度，那就可以拆成n个样本"""
        # 尝试将长样本拆分成多个样本
        expanded_ids = []
        for ids in all_ids:
            # 计算可以拆分成多少个seq_len长度的样本
            num_splits = len(ids) // seq_len
            for i in range(num_splits):
                start_idx = i * seq_len
                end_idx = start_idx + seq_len
                expanded_ids.append(ids[start_idx:end_idx])

        # 更新all_ids为拆分后的样本
        all_ids = expanded_ids

        # 如果拆分后仍然不够，则抛出错误
        if len(all_ids) < n_samples:
            raise ValueError(
                f"Not enough samples to calibrate even after splitting, {len(all_ids)=} < {n_samples=}"
            )
    samples = []
    for _ in range(n_samples):
        while True:
            cur_ids = all_ids[random.randint(0, len(all_ids) - 1)]
            if len(cur_ids) >= seq_len:
                break
        inp = torch.tensor(cur_ids[:seq_len], dtype=torch.int64).unsqueeze(0)
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def txt_general_preproc(
    calib_dataset, tokenizer, n_samples, seq_len, key, prefix_token_ids=None
):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    longest_line_encoded = None
    print(f"{prefix_token_ids=}")
    for data in dataset:
        line = data[key]
        # logger.info(f"{n_run}/{n_samples} calib data: {data}")
        trainenc = tokenizer(
            line, return_tensors="pt", max_length=seq_len, truncation=True
        )
        line_encoded = trainenc.input_ids

        if (
            longest_line_encoded is None
            or line_encoded.shape[1] > longest_line_encoded.shape[1]
        ):
            longest_line_encoded = line_encoded

        if line_encoded.shape[1] < seq_len:
            continue

        if prefix_token_ids:
            assert line_encoded.ndim == 2 and line_encoded.shape[0] == 1
            line_encoded[0][len(prefix_token_ids) :] = line_encoded[0][
                : -len(prefix_token_ids)
            ].clone()
            line_encoded[0][: len(prefix_token_ids)] = torch.tensor(
                prefix_token_ids, dtype=line_encoded.dtype, device=line_encoded.device
            )
            logger.info(f"add prefix token ids: {prefix_token_ids}")

        samples.append(line_encoded)
        n_run += 1
        if n_run == n_samples:
            break
    if not samples:
        samples = [longest_line_encoded]
    logger.info(f"{len(samples)=}, {len(samples[0])=}")
    return samples
