import json
import os
from abc import ABCMeta

import torch
from datasets import load_dataset, load_from_disk
from loguru import logger
from PIL import Image
from torch.nn import functional as F

from .specified_preproc import PREPROC_REGISTRY


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, tokenizer, calib_cfg, batch_process=None):
        # calib_cfg
        logger.info(f"calib_cfg : {calib_cfg}")
        self.tokenizer = tokenizer
        self.batch_process = batch_process
        self.calib_dataset_name = calib_cfg["name"]
        self.padding = calib_cfg.get("padding", False)
        # if self.calib_dataset_name == "ultrachat":
        #     assert self.padding
        self.download = calib_cfg["download"]
        self.load_from_txt = calib_cfg.get("load_from_txt", False)
        self.calib_dataset_path = calib_cfg.get("path", None)
        self.apply_chat_template = calib_cfg.get("apply_chat_template", False)
        self.n_samples = calib_cfg["n_samples"]
        self.calib_bs = calib_cfg["bs"]
        self.seq_len = calib_cfg.get("seq_len", None)
        self.preproc = calib_cfg.get("preproc", False)
        if self.calib_dataset_name == "ultrachat":
            assert self.preproc == "ultrachat_general"
        if self.preproc == "original_txt":
            assert self.seq_len is None
        self.seed = calib_cfg["seed"]
        self.dataset_key = {
            "pileval": "text",
            "c4": "text",
            "wikitext2": "text",
            "ptb": "sentence",
            "r1-fail": "text",
            "r1-fail-50": "text",
        }
        if self.calib_dataset_name in self.dataset_key:
            self.key = self.dataset_key[self.calib_dataset_name]
        self.build_calib_dataset()

    def build_calib_dataset(self):
        if self.download:
            if self.calib_dataset_name == "pileval":
                self.calib_dataset = load_dataset(
                    "mit-han-lab/pile-val-backup", split="validation"
                )
            elif self.calib_dataset_name == "c4":
                self.calib_dataset = load_dataset(
                    "allenai/c4",
                    data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                    split="train",
                )
            elif self.calib_dataset_name == "wikitext2":
                self.calib_dataset = load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split="train"
                )
            elif self.calib_dataset_name == "ptb":
                self.calib_dataset = load_dataset(
                    "ptb_text_only", "penn_treebank", split="train"
                )
            elif self.calib_dataset_name == "ultrachat":
                ultrachat_dataset = load_dataset(
                    "HuggingFaceH4/ultrachat_200k", split="train_sft"
                )
                # self.calib_dataset = self.convert_ultrachat_to_custom_format(
                #     ultrachat_dataset
                # )
                self.calib_dataset = ultrachat_dataset
            else:
                raise Exception(f"Not support {self.calib_dataset_name} dataset.")
        else:
            if (
                self.calib_dataset_name == "custom_txt"
                or self.calib_dataset_name == "custom_mm"
                or self.calib_dataset_name == "images"
            ):  # noqa
                self.calib_dataset = self.get_cutomdata(self.calib_dataset_path)
            elif self.calib_dataset_name == "ultrachat":
                ultrachat_dataset = load_from_disk(self.calib_dataset_path)
                # self.calib_dataset = self.convert_ultrachat_to_custom_format(
                #     ultrachat_dataset
                # )
                self.calib_dataset = ultrachat_dataset
            else:
                self.calib_dataset = load_from_disk(self.calib_dataset_path)

    def get_calib_model_inputs(self, samples, prefix_token_ids=None):
        if not self.padding:
            if self.calib_dataset_name == "images":
                calib_model_inputs = self.get_batch_process(samples)
            else:
                assert not self.calib_dataset_name == "custom_mm"
                if self.calib_dataset_name == "custom_txt":
                    txts = self.batch_process(
                        samples,
                        calib_or_eval="calib",
                        apply_chat_template=self.apply_chat_template,
                        return_inputs=False,
                    )
                else:
                    txts = self.calib_dataset
                preproc = PREPROC_REGISTRY[self.preproc]
                preproc_param_dict = {
                    "calib_dataset": txts,
                    "tokenizer": self.tokenizer,
                    "n_samples": self.n_samples,
                    "seq_len": self.seq_len,
                    "prefix_token_ids": prefix_token_ids,
                }
                if self.preproc == "txt_general_preproc":
                    preproc_param_dict["key"] = self.key
                samples = preproc(**preproc_param_dict)
                calib_model_inputs = []
                if self.calib_bs < 0:
                    batch = torch.cat(samples, dim=0)
                    calib_model_inputs.append({"input_ids": batch})
                elif self.calib_bs == 1:
                    for i in range(len(samples)):
                        calib_model_inputs.append({"input_ids": samples[i]})
                elif self.calib_bs > 1:
                    for i in range(0, len(samples), self.calib_bs):
                        start = i
                        end = min(i + self.calib_bs, len(samples))
                        batch = samples[start:end]
                        batch = torch.cat(batch, dim=0)
                        calib_model_inputs.append({"input_ids": batch})
        else:
            assert (
                self.calib_dataset_name == "custom_txt"
                or self.calib_dataset_name == "custom_mm"
                or self.calib_dataset_name == "ultrachat"
            )
            calib_model_inputs = self.get_batch_process(samples)
        return calib_model_inputs

    def get_batch_process(self, samples):
        calib_model_inputs = []
        if self.calib_bs < 0:
            calib_model_inputs.append(
                self.batch_process(
                    samples,
                    calib_or_eval="calib",
                    apply_chat_template=self.apply_chat_template,
                )
            )
        elif self.calib_bs == 1:
            calib_model_inputs = [
                self.batch_process(
                    [sample],
                    calib_or_eval="calib",
                    apply_chat_template=self.apply_chat_template,
                )
                for sample in samples
            ]  # noqa
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                calib_model_inputs.append(
                    self.batch_process(
                        batch,
                        calib_or_eval="calib",
                        apply_chat_template=self.apply_chat_template,
                    )
                )
        return calib_model_inputs

    def get_calib_dataset(self, prefix_token_ids=None):
        samples = self.calib_dataset[
            int(os.environ["RANK"]) :: int(os.environ["WORLD_SIZE"])
        ]
        logger.info(f"len(samples) rank : {len(samples)}")

        calib_model_inputs = self.get_calib_model_inputs(samples, prefix_token_ids)
        logger.info(f"len(calib_model_inputs) : {len(calib_model_inputs)}")
        if self.padding:
            padding_mask = [
                calib_model_input["attention_mask"]
                for calib_model_input in calib_model_inputs
            ]  # noqa
        else:
            padding_mask = None
        return calib_model_inputs, padding_mask

    def get_cutomdata(self, custom_dataset):
        audio_img_qa_json = os.path.join(custom_dataset, "samples.json")
        fp = open(audio_img_qa_json)
        custom_data_samples = json.load(fp)
        for idx in range(len(custom_data_samples)):
            if "audio" in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]["audio"], list):
                    for audio_idx in range(len(custom_data_samples[idx]["audio"])):
                        custom_data_samples[idx]["audio"][audio_idx] = os.path.join(
                            custom_dataset, custom_data_samples[idx]["audio"][audio_idx]
                        )
                else:
                    custom_data_samples[idx]["audio"] = os.path.join(
                        custom_dataset, custom_data_samples[idx]["audio"]
                    )
            else:
                custom_data_samples[idx]["audio"] = None
            if "image" in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]["image"], list):
                    for img_idx in range(len(custom_data_samples[idx]["image"])):
                        custom_data_samples[idx]["image"][img_idx] = os.path.join(
                            custom_dataset, custom_data_samples[idx]["image"][img_idx]
                        )
                else:
                    custom_data_samples[idx]["image"] = os.path.join(
                        custom_dataset, custom_data_samples[idx]["image"]
                    )
            else:
                custom_data_samples[idx]["image"] = None
            if "question" not in custom_data_samples[idx]:
                custom_data_samples[idx]["question"] = ""
            if "answer" not in custom_data_samples[idx]:
                custom_data_samples[idx]["answer"] = ""
        return custom_data_samples

    def convert_ultrachat_to_custom_format(self, ultrachat_dataset):
        """Convert ultrachat dataset to custom format compatible with get_cutomdata"""
        custom_data_samples = []
        for idx in range(len(ultrachat_dataset)):
            sample = ultrachat_dataset[idx]
            messages = sample["messages"]

            # Extract question and answer from messages
            question = ""
            answer = ""

            if messages:
                # Find the first user message as question
                for msg in messages:
                    if msg["role"] == "user":
                        question = msg["content"]
                        break

                # Find the first assistant message as answer
                for msg in messages:
                    if msg["role"] == "assistant":
                        answer = msg["content"]
                        break

            # Create custom format sample
            custom_sample = {
                "audio": None,
                "image": None,
                "question": question,
                "answer": answer,
            }
            custom_data_samples.append(custom_sample)

        return custom_data_samples
