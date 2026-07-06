import json
import hashlib
from pathlib import Path
from typing import Optional, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm.auto import tqdm
from jaxtyping import Float

# ---------------------------------------------------------------------------
# Hashing utility
# ---------------------------------------------------------------------------


def _stable_hash(obj, algo: str = "sha256") -> str:
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    h = hashlib.new(algo)
    h.update(payload)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Generic math dataset loader: GSM8K + MATH
# ---------------------------------------------------------------------------


class MathWordProblemDataset(Dataset):
    """
    Supports:
      - gsm8k (question -> answer)
      - math  (problem  -> solution)

    For MATH:
      - HF dataset only has a train split, so we deterministically carve out
        a validation set from train using train_test_split().
      - split="train" -> train portion
      - split="test"  -> held-out validation portion

    IMPORTANT:
    - Examples longer than max_length are SKIPPED (not truncated).
      This is especially important for MATH so we don't train on cut-off solutions.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "gsm8k",
        split: str = "train",
        max_length: int = 512,
        val_fraction: float = 0.1,
        split_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == "gsm8k":
            raw = load_dataset("gsm8k", "main", split=split)

        elif self.dataset_name == "math":
            # MATH only has train, so create our own deterministic val split
            full = load_dataset("qwedsacf/competition_math", split="train")
            split_ds = full.train_test_split(
                test_size=val_fraction,
                seed=split_seed,
                shuffle=True,
            )

            if split == "train":
                raw = split_ds["train"]
            elif split in ("test", "validation", "val"):
                raw = split_ds["test"]
            else:
                raise ValueError(
                    f"For dataset_name='math', split must be 'train' or 'test' (got {split})"
                )

        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        processed = [self._process(ex) for ex in raw]
        self.examples = [ex for ex in processed if ex is not None]

        print(
            f"[{self.dataset_name}:{split}] kept {len(self.examples)}/{len(raw)} examples "
            f"(max_length={self.max_length}, skipped {len(raw) - len(self.examples)} overlong)"
        )

    def _find_response_start(
        self, question: str, answer: str, input_ids: torch.Tensor
    ) -> int:
        """
        Locates the token index where the assistant response starts.
        Uses the tokenizer's chat template instead of searching for a
        model-specific assistant marker, so it works for both Phi and Qwen.
        """
        prompt_messages = [{"role": "user", "content": question}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if prompt_ids.shape[0] <= input_ids.shape[0] and torch.equal(
            input_ids[: prompt_ids.shape[0]], prompt_ids
        ):
            return prompt_ids.shape[0]

        full_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer(
            full_text,
            truncation=False,
            return_tensors="pt",
        )[
            "input_ids"
        ].squeeze(0)

        if full_ids.shape[0] <= input_ids.shape[0]:
            # Fallback: compare against the full rendered text and mask after
            # the prompt-only prefix length, which is stable across models.
            return min(prompt_ids.shape[0], full_ids.shape[0])

        return len(input_ids)  # fallback: no answer found → mask everything

    def _get_qa(self, example: dict) -> tuple[str, str]:
        if self.dataset_name == "gsm8k":
            return example["question"], example["answer"]
        elif self.dataset_name == "math":
            return example["problem"], example["solution"]
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

    def _process(self, example: dict) -> Optional[dict]:
        question, answer = self._get_qa(example)

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # First tokenize WITHOUT truncation so we can skip overlong examples
        tokenized = self.tokenizer(
            text,
            truncation=False,
            return_tensors="pt",
        )

        seq_len = tokenized["input_ids"].shape[1]
        if seq_len > self.max_length:
            return None

        # Re-tokenize with padding to fixed length
        tokenized = self.tokenizer(
            text,
            truncation=False,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze()  # (seq_len,)
        attention_mask = tokenized["attention_mask"].squeeze()  # (seq_len,)

        # --- Response masking for CE loss (prompt tokens → -100) ---
        labels = input_ids.clone()
        response_start = self._find_response_start(question, answer, input_ids)
        labels[:response_start] = -100  # ignore prompt in CE loss
        labels[attention_mask == 0] = -100  # ignore padding

        # --- Answer mask for ratio loss ---
        answer_mask = torch.zeros_like(attention_mask)
        answer_mask[response_start:] = attention_mask[response_start:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_mask": answer_mask,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_math_dataloader(
    tokenizer: AutoTokenizer,
    dataset_name: str = "gsm8k",
    split: str = "train",
    max_length: int = 512,
    batch_size: int = 4,
    shuffle: bool = True,
    val_fraction: float = 0.1,
    split_seed: int = 42,
) -> DataLoader:
    dataset = MathWordProblemDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        max_length=max_length,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Reference-augmented dataloader
# ---------------------------------------------------------------------------


class ReferenceAugmentedARDataLoader:
    """
    Wraps a DataLoader and attaches pre-computed baseline log-likelihoods.
    The answer_mask is already part of each batch from the dataset.
    """

    def __init__(
        self,
        base: DataLoader,
        ref_logL: Float[torch.Tensor, "M"],
    ):
        self.base = base
        self.ref_logL = ref_logL

    def __len__(self):
        return len(self.base)

    def __iter__(self) -> Iterator:
        offset = 0
        for batch in self.base:
            bsz = batch["input_ids"].shape[0]
            batch["ref_logL"] = self.ref_logL[offset : offset + bsz]
            offset += bsz
            yield batch


# ---------------------------------------------------------------------------
# Log-likelihood computation — answer_mask-aware
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compute_sequence_log_likelihood(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    answer_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-sequence log-likelihood, optionally restricted to answer tokens.

    If `answer_mask` is provided (shape: batch x seq_len, values 0/1),
    only positions where answer_mask == 1 contribute to the logL sum and
    the normalisation denominator.

    Returns:
        log_likelihoods: shape (batch_size,) — mean logL per answer token
    """
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, V)
            log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

            # Shift for next-token prediction
            shifted_ids = input_ids[:, 1:].contiguous()  # (B, T-1)
            shifted_log_probs = log_probs[:, :-1, :]  # (B, T-1, V)

            B, T = shifted_ids.shape
            b_idx = torch.arange(B, device=device).unsqueeze(1)
            t_idx = torch.arange(T, device=device).unsqueeze(0)

            token_log_probs = shifted_log_probs[b_idx, t_idx, shifted_ids]  # (B, T-1)

            # Choose which mask to use for averaging
            if answer_mask is not None:
                mask = answer_mask[:, 1:].float()  # align with shifted positions
            else:
                mask = attention_mask[:, 1:].float()

            token_log_probs = token_log_probs * mask

            # Mean logL per (answer) token — safe divide
            denom = mask.sum(dim=1).clamp(min=1.0)
            return token_log_probs.sum(dim=1) / denom


# ---------------------------------------------------------------------------
# Caching wrapper (dataset/split/length-aware cache key)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def get_reference_augmented_ar_dataloader(
    baseline_model_id: str,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    *,
    dataset_name: str = "gsm8k",
    split_name: str = "train",
    cache_dir: Path = Path(".cache/baseline_ar_logL"),
    ignore_cache: bool = False,
    device: str = "cuda",
) -> ReferenceAugmentedARDataLoader:
    """
    Computes (or loads cached) baseline logL over the full dataset, then
    wraps the dataloader. Uses answer_mask so that baseline logL is also
    restricted to response tokens.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_payload = {
        "fn": "get_reference_augmented_ar_dataloader_v4_chatmask",
        "baseline_model_id": baseline_model_id,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "num_examples": len(dataloader.dataset),
    }
    key = _stable_hash(key_payload)
    cache_path = cache_dir / f"{key}.pt"

    if cache_path.exists() and not ignore_cache:
        print(f"Loading cached baseline logL from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        logL_all = cached["logL"]
        return ReferenceAugmentedARDataLoader(dataloader, logL_all)

    print(
        f"Computing baseline logL with {baseline_model_id} (will cache to {cache_path})"
    )
    baseline = (
        AutoModelForCausalLM.from_pretrained(
            baseline_model_id,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    logL_chunks = []
    for batch in tqdm(dataloader, desc=f"Computing baseline logL ({split_name})"):
        ids = batch["input_ids"].to(device)
        amask = batch["attention_mask"].to(device)
        amask_ans = batch["answer_mask"].to(device)

        logL = compute_sequence_log_likelihood(
            baseline, ids, amask, device, answer_mask=amask_ans
        )
        logL_chunks.append(logL.cpu())

    logL_all = torch.cat(logL_chunks, dim=0)
    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save({"logL": logL_all, "meta": key_payload}, tmp_path)
    tmp_path.replace(cache_path)

    del baseline
    torch.cuda.empty_cache()

    print(f"Cached {len(logL_all)} baseline logL values to {cache_path}")
    return ReferenceAugmentedARDataLoader(dataloader, logL_all)
