import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from tqdm import tqdm

def _ensure_repo_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = None
    for parent in (here.parent, *here.parents):
        if (parent / "src").is_dir():
            repo_root = parent
            break
    if repo_root is None:
        return
    src_root = repo_root / "src"
    for path in (repo_root, src_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_repo_on_path()

from experiments.reasoning.power_sampling_reasoning.power_samp_utils import format_prompt

# Use shared ratio computation
from src.evaluation.sequence_metrics import sequence_likelihood_ratios


# Monkeypatch for Phi-3.5 compatibility with newer transformers versions
if not hasattr(DynamicCache, "get_max_length"):

    def get_max_length(self):
        return self.get_seq_length()

    DynamicCache.get_max_length = get_max_length

# Add seen_tokens property for newer transformers versions that expect it
if not hasattr(DynamicCache, "seen_tokens"):

    @property
    def seen_tokens(self):
        return self.get_seq_length()

    DynamicCache.seen_tokens = seen_tokens

# Add get_usable_length method for newer transformers versions that expect it
if not hasattr(DynamicCache, "get_usable_length"):

    def get_usable_length(self, seq_length, layer_idx=None):
        # Return the cached sequence length (what's already in the cache)
        # The attention mechanism will handle adding the current seq_length
        return self.get_seq_length()

    DynamicCache.get_usable_length = get_usable_length

# Monkey-patch __getitem__ to return cached KV properly for Phi model compatibility
if not hasattr(DynamicCache, "_phi_patched"):
    # Just ensure the cache behaves correctly during generation
    DynamicCache._phi_patched = True


def _stable_hash(obj: Dict) -> str:
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def _find_response_start(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> int:
    marker_ids = tokenizer.encode("<|assistant|>", add_special_tokens=False)
    marker_len = len(marker_ids)
    ids = input_ids.tolist()
    for i in range(len(ids) - marker_len):
        if ids[i : i + marker_len] == marker_ids:
            return i + marker_len
    return len(ids)


def _build_example(
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    model_name: str,
    max_length: int,
    cot: bool,
) -> Optional[Dict[str, torch.Tensor]]:
    prompt = format_prompt(question, model_name, tokenizer, cot=cot)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    tokenized = tokenizer(text, truncation=False, return_tensors="pt")
    seq_len = tokenized["input_ids"].shape[1]
    if seq_len > max_length:
        return None

    tokenized = tokenizer(
        text,
        truncation=False,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)

    response_start = _find_response_start(tokenizer, input_ids)
    answer_mask = torch.zeros_like(attention_mask)
    answer_mask[response_start:] = attention_mask[response_start:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_mask": answer_mask,
    }


def _load_math500(
    dataset_json: Path,
    tokenizer: AutoTokenizer,
    model_name: str,
    max_length: int,
    cot: bool,
) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    data = json.loads(dataset_json.read_text())
    examples: List[Dict[str, torch.Tensor]] = []
    skipped = 0
    for item in data:
        ex = _build_example(
            tokenizer,
            question=item["prompt"],
            answer=item["answer"],
            model_name=model_name,
            max_length=max_length,
            cot=cot,
        )
        if ex is None:
            skipped += 1
        else:
            examples.append(ex)
    return examples, skipped


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "answer_mask": torch.stack([b["answer_mask"] for b in batch], dim=0),
    }


@torch.inference_mode()
def _compute_logL(
    model,
    dataloader: DataLoader,
    device: str,
) -> torch.Tensor:
    logL_chunks = []
    for batch in tqdm(dataloader, desc="logL"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answer_mask = batch["answer_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            shifted_ids = input_ids[:, 1:].contiguous()
            shifted_log_probs = log_probs[:, :-1, :]

            bsz, t = shifted_ids.shape
            b_idx = torch.arange(bsz, device=device).unsqueeze(1)
            t_idx = torch.arange(t, device=device).unsqueeze(0)
            token_log_probs = shifted_log_probs[b_idx, t_idx, shifted_ids]

            mask = answer_mask[:, 1:].float()
            token_log_probs = token_log_probs * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            logL = token_log_probs.sum(dim=1) / denom

        logL_chunks.append(logL.detach().cpu())

    return torch.cat(logL_chunks, dim=0)


def _load_or_compute_logL(
    model_id: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    dataset_json: Path,
    max_length: int,
    cot: bool,
    batch_size: int,
    device: str,
    cache_dir: Path,
    ignore_cache: bool,
) -> Tuple[torch.Tensor, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_payload = {
        "fn": "math500_logL_v1",
        "model_id": model_id,
        "dataset_json": str(dataset_json),
        "max_length": max_length,
        "cot": cot,
    }
    cache_key = _stable_hash(key_payload)
    cache_path = cache_dir / f"{cache_key}.pt"

    if cache_path.exists() and not ignore_cache:
        cached = torch.load(cache_path, map_location="cpu")
        return cached["logL"], cached.get("skipped", 0)

    tokenizer.pad_token = tokenizer.eos_token
    examples, skipped = _load_math500(
        dataset_json, tokenizer, model_name, max_length=max_length, cot=cot
    )
    dataloader = DataLoader(
        examples, batch_size=batch_size, shuffle=False, collate_fn=_collate
    )

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    logL = _compute_logL(model, dataloader, device=device)

    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save({"logL": logL, "meta": key_payload, "skipped": skipped}, tmp_path)
    tmp_path.replace(cache_path)

    return logL, skipped


def _calc_likelihood_log_ratio_scaling_laplace(
    base_model_likelihoods: torch.Tensor,
    target_model_likelihoods: torch.Tensor,
    pairing_mode: str = "pairs",
    minimal_abs_ratio_difference: float = 0.0,
    laplace_alpha: float = 0.1,
) -> Tuple[float, float]:
    """
    Computes mean/variance of smoothed likelihood-ratio scaling.
    Additive (Laplace-style) smoothing is applied as:
      (base_diff + alpha) / (target_diff + alpha)
    """
    base_differences = sequence_likelihood_ratios(base_model_likelihoods, pairing_mode)
    target_differences = sequence_likelihood_ratios(
        target_model_likelihoods, pairing_mode
    )

    good_ratios = (base_differences.abs() >= minimal_abs_ratio_difference) & (
        target_differences.abs() >= minimal_abs_ratio_difference
    )

    base_sel = base_differences[good_ratios]
    target_sel = target_differences[good_ratios]
    ratios = (base_sel + laplace_alpha) / (target_sel + laplace_alpha)
    return ratios.mean().item(), ratios.var().item()


def _resolve_models(args: argparse.Namespace) -> List[str]:
    models: List[str] = []
    if args.models:
        models.extend(args.models)
    if args.models_file:
        for line in Path(args.models_file).read_text().splitlines():
            line = line.strip()
            if line:
                models.append(line)
    if args.models_glob:
        models.extend(sorted(str(p) for p in Path(".").glob(args.models_glob)))

    if not models:
        models = [
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k-temperature1.0-alpha0.01-seed42/checkpoint_25.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k-temperature1.0-alpha0.01-seed42/checkpoint_50.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k-temperature1.0-alpha0.01-seed42/checkpoint_75.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k-temperature1.0-alpha0.01-seed42/checkpoint_100.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.25-alpha0.01-seed42/checkpoint_25.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.25-alpha0.01-seed42/checkpoint_50.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.25-alpha0.01-seed42/checkpoint_75.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.25-alpha0.01-seed42/checkpoint_100.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.0-alpha0.01-seed42/checkpoint_25.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.0-alpha0.01-seed42/checkpoint_50.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.0-alpha0.01-seed42/checkpoint_75.pt",
            "/m/cs/scratch/temperature_diffusion/models/phi35-gsm8k/-temperature0.0-alpha0.01-seed42/checkpoint_100.pt",
        ]
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_model_id",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="data/MATH500.json",
    )
    parser.add_argument("--models", type=str, nargs="*")
    parser.add_argument("--models_file", type=str, default=None)
    parser.add_argument("--models_glob", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no_cot", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/m/cs/scratch/temperature_diffusion/.cache/math500_ratio",
    )
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument(
        "--out_csv", type=str, default="results/metrics_math500_ratio.csv"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ratio_thr", type=float, default=0)
    parser.add_argument(
        "--laplace_alpha",
        type=float,
        default=0.1,
        help="Additive smoothing constant for ratio computation",
    )
    args = parser.parse_args()

    dataset_json = Path(args.dataset_json)
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_json}")

    cot = True
    if args.no_cot:
        cot = False
    elif args.cot:
        cot = True

    models = _resolve_models(args)

    cache_dir = Path(args.cache_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.baseline_model_id, trust_remote_code=True
    )

    base_logL, base_skipped = _load_or_compute_logL(
        model_id=args.baseline_model_id,
        tokenizer=tokenizer,
        model_name="phi",
        dataset_json=dataset_json,
        max_length=args.max_length,
        cot=cot,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=cache_dir / "baseline",
        ignore_cache=args.ignore_cache,
    )

    rows = []
    for model_id in models:
        model_logL, skipped = _load_or_compute_logL(
            model_id=model_id,
            tokenizer=tokenizer,
            model_name=model_id,
            dataset_json=dataset_json,
            max_length=args.max_length,
            cot=cot,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=cache_dir / "models",
            ignore_cache=args.ignore_cache,
        )

        ratio_mean, ratio_var = _calc_likelihood_log_ratio_scaling_laplace(
            base_logL,
            model_logL,
            pairing_mode="pairs",
            minimal_abs_ratio_difference=args.ratio_thr,
            laplace_alpha=args.laplace_alpha,
        )

        row = {
            "model_id": model_id,
            "baseline_model_id": args.baseline_model_id,
            "dataset": "MATH500",
            "max_length": args.max_length,
            "cot": cot,
            "ratio_scale_mean": ratio_mean,
            "ratio_scale_var": ratio_var,
            "laplace_alpha": args.laplace_alpha,
            "skipped": skipped,
            "baseline_skipped": base_skipped,
            "model_fp": _stable_hash({"model_id": model_id}),
            "data_fp": _stable_hash(
                {
                    "dataset_json": str(dataset_json),
                    "max_length": args.max_length,
                    "cot": cot,
                }
            ),
        }
        rows.append(row)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
