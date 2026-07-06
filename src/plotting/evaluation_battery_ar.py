"""
Evaluation battery for Autoregressive (AR) language models.

Evaluates AR models (e.g., GPT-2 variants) by computing sequence log-likelihoods
and comparing them to a baseline AR model. Outputs results in CSV format compatible
with the diffusion model evaluation battery.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from typing import Iterable, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
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


import utils.config_classes as config_classes

# Handle both relative and absolute imports for flexibility
try:
    from .evaluate_perplexity_temp_scaling import (
        calc_likelihood_log_ratio_scaling,
        kendall_tau,
        regress_effective_temperature_scaling,
    )
except ImportError:
    from plotting.evaluate_perplexity_temp_scaling import (
        calc_likelihood_log_ratio_scaling,
        kendall_tau,
        regress_effective_temperature_scaling,
    )


def _short_model_name(model_id: str) -> str:
    short_name = (
        model_id.split("models/", 1)[1]
        if "models/" in model_id
        else model_id.split("/")[-1]
    )
    for separator in ("/", "\\", ":"):
        short_name = short_name.replace(separator, "_")
    return short_name or "model"


class ARTemperatureScalingWrapper(nn.Module):
    """
    Temperature scaling wrapper for autoregressive language models.
    Scales logits before softmax computation.
    """

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("_temp", torch.tensor(float(temperature)))

    @property
    def temperature(self) -> float:
        return float(self._temp.item())

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with temperature scaling."""
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        logits = outputs.logits

        t = self._temp.to(dtype=logits.dtype, device=logits.device)
        if t == 1:
            return outputs

        # Scale logits by temperature
        outputs.logits = logits / t
        return outputs

    def __getattr__(self, name: str):
        if name != "model" and hasattr(self.model, name):
            return getattr(self.model, name)
        return super().__getattr__(name)


def trimmed_mean_flat(x, trim_fraction=0.1):
    """
    Compute the trimmed mean of a tensor after flattening it.

    Args:
        x (torch.Tensor): Input tensor (any shape)
        trim_fraction (float): Fraction to trim from each end (0 <= trim_fraction < 0.5)

    Returns:
        torch.Tensor: Trimmed mean (scalar)
    """
    # Flatten the tensor
    x_flat = x.flatten()

    # Sort the flattened tensor
    x_sorted, _ = torch.sort(x_flat)

    # Compute number of elements to trim
    n = x_sorted.numel()
    k = int(n * trim_fraction)

    # Slice to remove k elements from both ends
    if k > 0:
        x_sorted = x_sorted[k : n - k]

    # Compute mean of remaining elements
    return x_sorted.mean()


def compute_sequence_log_likelihood(model, input_ids, attention_mask, device):
    """
    Compute the log likelihood of sequences given an autoregressive model.

    Args:
        model: The autoregressive language model
        input_ids: Token IDs of shape (batch_size, seq_length)
        attention_mask: Attention mask of shape (batch_size, seq_length)
        device: Device to run on

    Returns:
        log_likelihoods: Normalized log likelihoods for each sequence in batch, shape (batch_size,)
                         Returns per-token average log likelihood
    """
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )

            # Get logits from model output
            logits = outputs.logits  # shape: (batch_size, seq_length, vocab_size)

            # Compute log probabilities via log_softmax
            log_probs = F.log_softmax(
                logits, dim=-1
            )  # shape: (batch_size, seq_length, vocab_size)

            # Get log probability of the target tokens
            # Shift input_ids and log_probs by one position for next-token prediction
            shifted_input_ids = input_ids[
                :, 1:
            ].contiguous()  # shape: (batch_size, seq_length-1)
            shifted_log_probs = log_probs[
                :, :-1, :
            ]  # shape: (batch_size, seq_length-1, vocab_size)

            # Gather log probs for ground truth tokens
            batch_size, seq_length = shifted_input_ids.shape
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            seq_indices = torch.arange(seq_length, device=device).unsqueeze(0)
            token_log_probs = shifted_log_probs[
                batch_indices, seq_indices, shifted_input_ids
            ]

            # Mask out padding tokens
            shifted_attention_mask = attention_mask[:, 1:]
            token_log_probs = token_log_probs * shifted_attention_mask

            # Sum log probabilities across sequence to get total log likelihood
            sequence_log_likelihoods = token_log_probs.sum(
                dim=1
            )  # shape: (batch_size,)

            # Average logL per token (per-token likelihood)
            return sequence_log_likelihoods / shifted_attention_mask.sum(dim=1)


def compute_metrics(logL: torch.Tensor, var: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics from log-likelihoods.

    Args:
        logL: (M,) per-example log likelihood
        var: (M,) variance (placeholder for consistency with diffusion battery)

    Returns:
        Dictionary of scalar metrics suitable for CSV
    """
    x = logL.detach()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {
            "n": 0.0,
            "mean_logL": float("nan"),
            "mean_nll": float("nan"),
            "std_nll": float("nan"),
            "mean_var": float("nan"),
            "std_var": float("nan"),
            "median_nll": float("nan"),
            "ppl": float("nan"),
            "min_logL": float("nan"),
            "max_logL": float("nan"),
        }

    x = x.to("cpu", dtype=torch.float64)
    nll = -x

    mean_logL = x.mean().item()
    mean_nll = nll.mean().item()
    std_nll = nll.std(unbiased=False).item()
    var = var.to("cpu", dtype=torch.float64)
    mean_var = var.mean().item()
    std_var = var.std(unbiased=False).item()
    median_nll = nll.median().item()
    ppl = 0  # placeholder, matching diffusion battery

    return {
        "n": float(x.numel()),
        "mean_logL": float(mean_logL),
        "mean_nll": float(mean_nll),
        "std_nll": float(std_nll),
        "mean_var": float(mean_var),
        "std_var": float(std_var),
        "median_nll": float(median_nll),
        "ppl": float(ppl),
        "min_logL": float(x.min().item()),
        "max_logL": float(x.max().item()),
    }


def get_ar_model_logL_cached(
    model_spec: config_classes.ModelSpec,
    data_spec: config_classes.OpenWebTextLoaderSpec,
    cache_dir: Path = Path(".cache/ar_eval"),
    ignore_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute or load cached log-likelihoods for an AR model.

    Args:
        model_spec: ModelSpec with model_id, max_length, myopic_temperature, device
        data_spec: DataLoader specification for evaluation data
        cache_dir: Directory to cache results
        ignore_cache: If True, recompute even if cache exists

    Returns:
        Tuple of (logL, var) tensors where logL is per-sequence log-likelihood
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache key from model spec and data fingerprint
    cache_key_dict = {
        "fn": "get_ar_model_logL_cached_v1",
        "model": model_spec.fingerprint(),
        "data": data_spec.fingerprint(),
    }
    cache_key = config_classes._stable_hash(cache_key_dict)
    cache_path = cache_dir / f"{cache_key}.pt"

    # Try loading from cache
    if cache_path.exists() and not ignore_cache:
        print(f"Loading cached AR logL from {cache_path}")
        cached_data = torch.load(cache_path, map_location="cpu")
        return cached_data["logL"], cached_data["var"]

    # Cache miss: compute logL over full dataset
    device = model_spec.device or "cuda"
    print(f"Computing AR logL for model: {model_spec.model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_spec.model_id)

    # Apply temperature scaling if specified
    if (
        model_spec.myopic_temperature is not None
        and model_spec.myopic_temperature != 1.0
    ):
        model = ARTemperatureScalingWrapper(
            model, temperature=model_spec.myopic_temperature
        )

    model.to(device)
    model.eval()

    # Get dataloader
    dl = data_spec.lazy().get()
    logL_chunks = []
    var_chunks = []
    batch_sizes = []

    for batch in tqdm(
        dl, desc=f"Computing AR logL for {Path(model_spec.model_id).name}"
    ):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_sizes.append(len(input_ids))

        logL = compute_sequence_log_likelihood(model, input_ids, attention_mask, device)
        logL_chunks.append(logL.detach().to("cpu"))
        # For AR models, we compute variance as 0 since it's deterministic
        var_chunks.append(torch.zeros_like(logL).to("cpu"))

    logL_all = torch.cat(logL_chunks, dim=0)
    var_all = torch.cat(var_chunks, dim=0)

    # Save to cache
    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "logL": logL_all,
            "var": var_all,
            "meta": cache_key_dict,
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)  # atomic on POSIX

    print(f"Cached AR logL to {cache_path}")
    print(
        f"Computed: {len(logL_chunks)} batches, batch sizes: {batch_sizes[:5]}... (total {sum(batch_sizes)} sequences)"
    )

    return logL_all, var_all


def run_ar_battery_to_csv(
    model_specs: list[config_classes.ModelSpec],
    data_spec: config_classes.OpenWebTextLoaderSpec,
    base_model_spec: config_classes.ModelSpec,
    out_dir: Path = Path("results"),
    cache_dir: Path = Path(".cache/ar_eval"),
    ignore_cache: bool = False,
    run_id: Optional[int] = None,
    save_pair_csvs: bool = False,
    pair_csv_dir: Optional[Path] = None,
) -> Path:
    """
    Evaluate a list of AR models and write results to CSV in diffusion battery format.

    Args:
        model_specs: List of ModelSpec objects with model configuration
        data_spec: DataLoader specification for evaluation
        base_model_spec: Baseline model spec for comparison
        out_dir: Output directory for CSV
        cache_dir: Cache directory for computed likelihoods
        ignore_cache: If True, recompute all likelihoods
        run_id: Optional run ID for parallel execution

    Returns:
        Path to output CSV file (matches diffusion battery format except no block_size/mask_token)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # File naming
    ss = data_spec.slice_start if data_spec.slice_start is not None else 0
    se = data_spec.slice_end if data_spec.slice_end is not None else "end"
    filename = (
        f"metrics_ar_openwebtext_{data_spec.split}"
        f"_slice{ss}-{se}"
        f"_L{data_spec.max_length}"
    )
    if run_id is not None:
        filename += f"_run{run_id}"
    filename += ".csv"
    out_csv = out_dir / filename

    # Compute baseline logL once (at temperature 1.0)
    print("Computing baseline AR logL...")
    base_logL, base_var = get_ar_model_logL_cached(
        base_model_spec,
        data_spec,
        cache_dir=cache_dir,
        ignore_cache=ignore_cache,
    )
    base_logL_cpu = base_logL.detach().to("cpu", dtype=torch.float64)

    rows = []
    for ms in model_specs:
        print(
            f"\nEvaluating model: {ms.model_id}, temperature: {ms.myopic_temperature}"
        )
        logL, var = get_ar_model_logL_cached(
            ms,
            data_spec,
            cache_dir=cache_dir,
            ignore_cache=ignore_cache,
        )
        logL_cpu = logL.detach().to("cpu", dtype=torch.float64)

        if save_pair_csvs:
            target_dir = pair_csv_dir or out_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            short_model_name = _short_model_name(ms.model_id)
            pair_filename = (
                f"pairs_ar_openwebtext_{data_spec.split}"
                f"_slice{ss}-{se}"
                f"_L{data_spec.max_length}"
                f"_{short_model_name}"
                f"_myopic_temp{ms.myopic_temperature}"
            )
            if run_id is not None:
                pair_filename += f"_run{run_id}"
            pair_filename += ".csv"
            pair_csv = target_dir / pair_filename
            with pair_csv.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["example_idx", "base_logL", "model_logL", "logL_diff"],
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "example_idx": idx,
                        "base_logL": float(base_val.item()),
                        "model_logL": float(model_val.item()),
                        "logL_diff": float((model_val - base_val).item()),
                    }
                    for idx, (base_val, model_val) in enumerate(
                        zip(base_logL_cpu, logL_cpu)
                    )
                )

        metrics = compute_metrics(logL, var)

        # Compute baseline-to-model ratio (simple per-example ratio)
        # Only compute where both are finite
        valid_mask = torch.isfinite(base_logL) & torch.isfinite(logL)
        # if valid_mask.sum() > 0:
        #     ratio_scale_mean = (base_logL[valid_mask] / logL[valid_mask]).mean().item()
        #     ratio_scale_trimmed_mean = trimmed_mean_flat(
        #         base_logL[valid_mask] / logL[valid_mask], trim_fraction=0.05
        #     ).item()
        #     ratio_scale_var = (base_logL[valid_mask] / logL[valid_mask]).var().item()
        # else:
        #     ratio_scale_mean = float("nan")
        #     ratio_scale_trimmed_mean = float("nan")
        #     ratio_scale_var = float("nan")
        ratio_mean, ratio_var = calc_likelihood_log_ratio_scaling(
            base_logL,
            logL,
            pairing_mode="pairs",
            minimal_abs_ratio_difference=0.1,
        )
        tau = kendall_tau(base_logL, logL, pairing_mode="pairs")
        effective_temp = regress_effective_temperature_scaling(
            base_logL, logL, pairing_mode="pairs"
        )

        # Build row matching diffusion battery format (except no block_size/mask_token)
        row = {
            "model_id": ms.model_id,
            "max_length": ms.max_length,
            "temperature": (
                ms.myopic_temperature.item()
                if ms.myopic_temperature is not None
                else None
            ),
            "revision": ms.revision,
            "model_fp": ms.fingerprint(),
            "data_fp": data_spec.fingerprint(),
            "slice_start": data_spec.slice_start,
            "slice_end": data_spec.slice_end,
            "ratio_scale_mean": ratio_mean,
            "ratio_scale_var": ratio_var,
            "kendall_tau": tau,
            "effective_temperature": effective_temp.item(),
            # "ratio_scale_trimmed_mean": ratio_scale_trimmed_mean,
        }
        row.update(metrics)
        rows.append(row)

    # Write CSV
    fieldnames: List[str] = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote results to: {out_csv}")
    return out_csv


if __name__ == "__main__":

    # Create base model spec

    # Create model specs from model_ids and temperatures
    # If single temperature given with multiple models, apply to all
    # temperatures = args.temperatures
    # if len(temperatures) == 1 and len(args.model_ids) > 1:
    #     temperatures = temperatures * len(args.model_ids)
    # elif len(temperatures) != len(args.model_ids):
    #     raise ValueError(
    #         f"Number of temperatures ({len(temperatures)}) must match number of models ({len(args.model_ids)}) "
    #         "or be a single temperature to apply to all"
    #     )

    device = "cuda"

    model_id = "gpt2"
    max_length = 1024
    batch_size = 32

    base_model_spec = config_classes.ModelSpec(
        model_id=model_id,
        max_length=max_length,
        device=device,
        myopic_temperature=None,
    )

    model_specs = [
        config_classes.ModelSpec(
            model_id=model_id,
            max_length=max_length,
            device=device,
            myopic_temperature=myopic_temp,
        )
        for myopic_temp in torch.logspace(math.log10(0.1), 0.5, 30 + 1)
    ]

    offset = 10**5
    size = 9000

    data_spec = config_classes.OpenWebTextLoaderSpec(
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        slice_start=offset,
        slice_end=offset + size,
    )

    out_csv = run_ar_battery_to_csv(
        model_specs=model_specs,
        data_spec=data_spec,
        base_model_spec=base_model_spec,
        out_dir=Path("/m/cs/scratch/temperature_diffusion/results"),
        cache_dir=Path("/m/cs/scratch/temperature_diffusion/.cache/bd3lm_eval"),
        save_pair_csvs=True,
        pair_csv_dir=Path("/m/cs/scratch/temperature_diffusion/results/pair_csvs"),
    )
    print(f"Results written to: {out_csv}")
