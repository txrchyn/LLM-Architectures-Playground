from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import torch
import torch.nn.functional as F
import tiktoken

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from benchmark.utils import load_config_from_py_file, resolve_device, resolve_dtype
from models import create_config, create_model


def _chunk_list(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _cleanup_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    unwanted_prefix = "_orig_mod."
    if any(key.startswith(unwanted_prefix) for key in state_dict.keys()):
        return {
            (k[len(unwanted_prefix) :] if k.startswith(unwanted_prefix) else k): v
            for k, v in state_dict.items()
        }
    return state_dict


@register_model("local")
class LocalCheckpointLM(TemplateLM):
    """
    lm-evaluation-harness adapter for locally trained models in this repository.
    """

    AUTO_MODEL_CLASS = None  # keeps TemplateLM encoding assumptions simple

    def __init__(
        self,
        model_name: str = "modern",
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        dtype: str = "float32",
        batch_size: int | str = 8,
        tokenizer: str = "gpt2",
        max_new_tokens: int | str = 128,
    ) -> None:
        super().__init__()
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(str(dtype))
        self._batch_size = int(batch_size)
        self.default_max_new_tokens = int(max_new_tokens)

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        if ckpt_path is not None and not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config_dict = checkpoint.get("config")
            if config_dict is None:
                raise ValueError("Checkpoint does not contain a `config` dictionary.")
            model_name = config_dict.get("model_name", model_name)
            config = create_config(model_name, config_dict)
        else:
            if config_path is None:
                raise ValueError("Either `checkpoint_path` or `config_path` must be provided.")
            config = load_config_from_py_file(config_path)

        self.model_name = model_name
        self.config = config
        self.max_seq_len = config.block_size

        self.model = create_model(model_name, config).to(self.device, dtype=self.dtype)
        self.model.eval()

        if ckpt_path is not None:
            state_dict = _cleanup_state_dict(checkpoint["model"])
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
            if unexpected:
                raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")

        self._encoder = tiktoken.get_encoding(tokenizer)
        try:
            self._eot_token = self._encoder.eot_token
        except AttributeError:
            self._eot_token = self._encoder.encode("<|endoftext|>")[0]

        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            self._autocast_kwargs = dict(device_type="cuda", dtype=self.dtype)
        else:
            self._autocast_kwargs = None

    @property
    def tokenizer_name(self) -> str:
        return "tiktoken/" + self._encoder.name

    @property
    def eot_token_id(self) -> int:
        return self._eot_token

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        allowed = kwargs.pop("allowed_special", {"<|endoftext|>"})
        return self._encoder.encode(string, allowed_special=allowed)

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self._encoder.decode(list(tokens))

    def _maybe_autocast(self):
        if self._autocast_kwargs:
            return torch.autocast(**self._autocast_kwargs)
        return contextlib.nullcontext()

    def _prepare_sequence(
        self, context_ids: list[int], continuation_ids: list[int]
    ) -> Tuple[list[int], int]:
        if not context_ids:
            context_ids = [self.eot_token_id]
        full = context_ids + continuation_ids
        if len(full) <= self.max_seq_len:
            return full, len(continuation_ids)

        overflow = len(full) - self.max_seq_len
        ctx_len = len(context_ids)
        drop = min(overflow, ctx_len)
        full = full[drop:]
        return full, len(continuation_ids)

    def _score_long_sequence(
        self, context_ids: list[int], continuation_ids: list[int]
    ) -> Tuple[float, bool]:
        if not context_ids:
            context_ids = [self.eot_token_id]
        tokens = context_ids + continuation_ids
        score_start = len(tokens) - len(continuation_ids)
        total_logprob = 0.0
        greedy = True

        for idx in range(len(tokens) - 1):
            start = max(0, idx - self.max_seq_len + 1)
            input_tokens = tokens[start : idx + 1]
            input_tensor = torch.tensor(input_tokens, device=self.device).unsqueeze(0)
            with self._maybe_autocast():
                logits, _ = self.model(input_tensor)
            next_logits = logits[0, -1].float()
            next_token = tokens[idx + 1]
            if idx + 1 >= score_start:
                logprob = F.log_softmax(next_logits, dim=-1)[next_token]
                total_logprob += logprob.item()
                greedy = greedy and (next_logits.argmax().item() == next_token)
        return total_logprob, greedy

    def _loglikelihood_tokens(
        self, requests: list[tuple[tuple[str, str], list[int], list[int]]], disable_tqdm=False
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool] | None] = [None] * len(requests)
        short_entries: list[tuple[int, list[int], int]] = []

        for idx, (_, ctx_ids, cont_ids) in enumerate(requests):
            ctx_list = list(ctx_ids)
            cont_list = list(cont_ids)
            full_len = len(ctx_list) + len(cont_list) if ctx_list else len(cont_list) + 1
            if full_len <= self.max_seq_len:
                full, cont_len = self._prepare_sequence(ctx_list, cont_list)
                short_entries.append((idx, full, cont_len))
            else:
                results[idx] = self._score_long_sequence(ctx_list, cont_list)

        for batch in _chunk_list(short_entries, self._batch_size):
            active = [entry for entry in batch if entry[2] > 0]

            for idx, _, cont_len in batch:
                if cont_len == 0:
                    results[idx] = (0.0, True)

            if not active:
                continue

            sequences = [entry[1] for entry in active]
            cont_lens = [entry[2] for entry in active]
            max_len = max(len(seq) for seq in sequences)
            input_ids = torch.full(
                (len(sequences), max_len),
                self.eot_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, seq in enumerate(sequences):
                input_ids[row, -len(seq) :] = torch.tensor(
                    seq, dtype=torch.long, device=self.device
                )

            with self._maybe_autocast():
                logits, _ = self.model(input_ids[:, :-1])
            log_probs = F.log_softmax(logits.float(), dim=-1)

            for row, (idx, _, _) in enumerate(active):
                cont_len = cont_lens[row]
                target_tokens = input_ids[row, -cont_len:]
                token_log_probs = log_probs[row, -cont_len:]
                gathered = token_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
                total_logprob = gathered.sum().item()
                greedy_tokens = token_log_probs.argmax(dim=-1)
                is_greedy = bool(torch.equal(greedy_tokens, target_tokens))
                results[idx] = (total_logprob, is_greedy)

        return [r if r is not None else (0.0, True) for r in results]

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        scores: list[float] = []
        for (string,) in [req.args for req in requests]:
            tokens = self.tok_encode(string)
            if not tokens:
                scores.append(0.0)
                continue
            context_tokens = [self.eot_token_id]
            total = 0.0
            for token in tokens:
                context_tokens = (context_tokens + [token])[-self.max_seq_len :]
                input_tensor = torch.tensor(context_tokens[:-1], device=self.device).unsqueeze(0)
                with self._maybe_autocast():
                    logits, _ = self.model(input_tensor)
                next_logits = logits[0, -1].float()
                total += F.log_softmax(next_logits, dim=-1)[token].item()
            scores.append(total)
        return scores

    def generate_until(
        self, requests, disable_tqdm: bool = False
    ) -> list[str]:
        generations: list[str] = []
        for context, gen_kwargs in [req.args for req in requests]:
            tokens = self.tok_encode(context)
            if not tokens:
                tokens = [self.eot_token_id]
            input_ids = torch.tensor(
                tokens[-self.max_seq_len :], dtype=torch.long, device=self.device
            ).unsqueeze(0)
            max_new = int(gen_kwargs.get("max_new_tokens", self.default_max_new_tokens))
            temperature = float(gen_kwargs.get("temperature", 1.0))
            top_k_val = gen_kwargs.get("top_k")
            top_k = int(top_k_val) if top_k_val is not None else None
            top_p = gen_kwargs.get("top_p")
            if top_p is not None and float(top_p) < 1.0:
                raise ValueError("top_p sampling is not supported by the local adapter yet.")
            with torch.no_grad(), self._maybe_autocast():
                generated = self.model.generate(
                    idx=input_ids,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_k=top_k,
                )
            new_tokens = generated[0, input_ids.size(1) :].tolist()
            completion = self.tok_decode(new_tokens)

            for stop_seq in gen_kwargs.get("until", []) or []:
                stop_index = completion.find(stop_seq)
                if stop_index != -1:
                    completion = completion[:stop_index]
                    break
            generations.append(completion)
        return generations
