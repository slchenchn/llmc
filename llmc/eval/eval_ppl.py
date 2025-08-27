import gc
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm, trange

from .eval_base import BaseEval


class PerplexityEval(BaseEval):
    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        testenc = testenc.input_ids
        try:
            nsamples = testenc.numel() // seq_len
        except:
            nsamples = len(testenc)
            assert bs == 1  # only support batch size 1 for ultrachat

        nlls = []

        # Loop through each batch
        all_seq_len = []
        for i in trange(0, nsamples, bs, desc="Evaluating perplexity"):
            # logger.info(f'index : {(i + 1) // bs}/{nsamples // bs}')
            # Calculate end index
            j = min(i + bs, nsamples)
            if isinstance(testenc, torch.Tensor):
                inputs = testenc[:, (i * seq_len) : (j * seq_len)]
                inputs = inputs.reshape(j - i, seq_len)
            else:
                inputs = testenc[i:j]
                inputs = torch.tensor(inputs, dtype=torch.int64)
            inputs = inputs.cuda()

            # Forward pass through the model
            lm_logits = model.model(inputs).logits
            model.reset_kv()

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            # Calculate negative log likelihood
            cur_seq_len = inputs.shape[1]
            all_seq_len.append(cur_seq_len)
            neg_log_likelihood = loss.float() * cur_seq_len * (j - i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (sum(all_seq_len)))

        # Empty CUDA cache to save memory
        # testenc.cpu()
        torch.cuda.empty_cache()

        return ppl.item()


class DecodePerplexityEval(BaseEval):
    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        num_eval_tokens = 0
        num_samples = 1 if self.num_samples is None else self.num_samples
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        nlls = []

        for text in testenc[:num_samples]:
            logger.info(text)
            encodings = self.tokenizer(text, return_tensors="pt")
            seq_len = encodings.input_ids.size(1)
            logger.info(f"seq_len: {seq_len}")
            pbar = tqdm(range(0, seq_len - 1))

            for idx in pbar:
                input_ids = encodings.input_ids[:, idx : idx + 1].cuda()
                with torch.no_grad():
                    outputs = model.model(
                        input_ids,
                    )
                    logits = outputs.logits.view(-1, model.model.config.vocab_size)
                    label = (
                        encodings.input_ids[:, idx + 1 : idx + 2]
                        .to(logits.device)
                        .view(-1)
                    )
                    neg_log_likelihood = loss_fn(logits, label)
                nlls.append(neg_log_likelihood)
                num_eval_tokens += 1
                if (
                    self.num_eval_tokens is not None
                    and num_eval_tokens >= self.num_eval_tokens
                ):
                    break
            if (
                self.num_eval_tokens is not None
                and num_eval_tokens >= self.num_eval_tokens
            ):
                break
        model.reset_kv()
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
