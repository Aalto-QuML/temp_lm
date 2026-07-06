from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from typing import Iterable, List, Dict, Any, Optional


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

import torch

from plotting.evaluate_perplexity_temp_scaling import (
    calc_likelihood_log_ratio_scaling,
    kendall_tau,
    regress_effective_temperature_scaling,
)

import utils.config_classes as config_classes


def _short_model_name(model_id: str) -> str:
    short_name = (
        model_id.split("models/", 1)[1]
        if "models/" in model_id
        else model_id.split("/")[-1]
    )
    for separator in ("/", "\\", ":"):
        short_name = short_name.replace(separator, "_")
    return short_name or "model"


# IMPORTANT: we only EVER touch this, the rest stays fixed
def compute_metrics(logL: torch.Tensor, var: torch.Tensor) -> Dict[str, float]:
    """
    logL: (M,) per-example log expected likelihood (whatever your cached function returns).
    Returns scalar metrics (python floats) suitable for CSV.
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
    ppl = 0  # math.exp(mean_nll)

    return {
        "n": float(x.numel()),
        "mean_logL": float(mean_logL),
        "mean_nll": float(mean_nll),
        "std_nll": float(std_nll),
        "mean_var": float(var.mean().item()),
        "std_var": float(var.std(unbiased=False).item()),
        "median_nll": float(median_nll),
        "ppl": float(ppl),
        "min_logL": float(x.min().item()),
        "max_logL": float(x.max().item()),
    }


def run_battery_to_csv(
    model_specs,
    data_spec,
    base_model_spec,  # <-- add this
    block_size=4,
    mask_token=50257,
    out_dir=Path("results"),
    cache_dir=Path(".cache/bd3lm_eval"),
    ignore_cache=False,
    pairing_mode="pairs",  # <-- add if you want configurable
    ratio_thr=1.0,  # <-- add if you want configurable
    run_id=None,  # <-- add for parallel-safe naming
    save_pair_csvs: bool = False,
    pair_csv_dir: Optional[Path] = None,
):

    out_dir.mkdir(parents=True, exist_ok=True)  # File naming
    ss = data_spec.slice_start if data_spec.slice_start is not None else 0
    se = data_spec.slice_end if data_spec.slice_end is not None else "end"
    block_sizes = sorted({getattr(ms, "block_size", 4) for ms in model_specs})
    bs_tag = f"bs{block_sizes[0]}" if len(block_sizes) == 1 else "bsMIXED"
    filename = (
        f"metrics_openwebtext_{data_spec.split}"
        f"_slice{ss}-{se}"
        f"_L{data_spec.max_length}"
        f"_{bs_tag}"
    )
    # For parallel runs, use a run-specific filename to avoid race conditions
    if run_id is not None:
        filename += f"_run{run_id}"
    filename += ".csv"
    out_csv = out_dir / filename

    lazy_data = data_spec.lazy()

    cache_key_extra = None  # {"block_size": block_size, "mask_token": int(mask_token)}

    rows = []
    base_logL, base_var = config_classes.get_full_log_exp_likelihood_cached(
        base_model_spec.lazy(),
        lazy_data,
        cache_dir=cache_dir,
        ignore_cache=ignore_cache,
        cache_key_extra=cache_key_extra,
    )
    base_logL = base_logL[:, 0, 0]  # (M, 16, 32) -> (M,)
    # print(base_logL.shape)
    base_logL_cpu = base_logL.detach().to("cpu", dtype=torch.float64)
    for ms in model_specs:
        logL, var = config_classes.get_full_log_exp_likelihood_cached(
            ms.lazy(),
            lazy_data,
            cache_dir=cache_dir,
            ignore_cache=ignore_cache,
            cache_key_extra=cache_key_extra,
        )
        logL = logL[:, 0, 0]  # (M, 16, 32) -> (M,)
        var = var[:, 0, 0]  # (M, 16, 32) -> (M,)
        # print(logL.shape)
        logL_cpu = logL.detach().to("cpu", dtype=torch.float64)

        if save_pair_csvs:
            target_dir = pair_csv_dir or out_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            short_model_name = _short_model_name(ms.model_id)
            pair_filename = (
                f"pairs_openwebtext_{data_spec.split}"
                f"_slice{ss}-{se}"
                f"_L{data_spec.max_length}"
                f"_{bs_tag}"
                f"_{short_model_name}"
                f"_myopic_temp{ms.myopic_temperature}"
            )
            if run_id is not None:
                pair_filename += f"_run{run_id}"
            pair_filename += ".csv"
            pair_csv = target_dir / pair_filename
            pair_rows = [
                {
                    "example_idx": idx,
                    "base_logL": float(base_val.item()),
                    "model_logL": float(model_val.item()),
                    "logL_diff": float((model_val - base_val).item()),
                }
                for idx, (base_val, model_val) in enumerate(
                    zip(base_logL_cpu, logL_cpu)
                )
            ]
            with pair_csv.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["example_idx", "base_logL", "model_logL", "logL_diff"],
                )
                writer.writeheader()
                writer.writerows(pair_rows)

        # ---- compute base once ----

        metrics = compute_metrics(logL, var)

        ratio_mean, ratio_var = calc_likelihood_log_ratio_scaling(
            base_logL,
            logL,
            pairing_mode=pairing_mode,
            minimal_abs_ratio_difference=ratio_thr,
        )

        effective_temperature = regress_effective_temperature_scaling(
            base_logL, logL, "pairs"
        )
        tau = kendall_tau(base_logL, logL, "pairs")

        row = {
            "model_id": ms.model_id,
            "max_length": ms.max_length,
            "temperature": ms.myopic_temperature,
            "revision": ms.revision,
            "model_fp": ms.fingerprint(),
            "data_fp": data_spec.fingerprint(),
            "slice_start": data_spec.slice_start,
            "slice_end": data_spec.slice_end,
            "block_size": block_size,
            "mask_token": mask_token,
            "ratio_scale_mean": ratio_mean,
            "ratio_scale_var": ratio_var,
            "effective_temperature": effective_temperature.item(),
            "kendall_tau": tau,
        }
        row.update(metrics)
        rows.append(row)

    # Write CSV (separate file per run_id to avoid race conditions)
    fieldnames: List[str] = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

        return out_csv


if __name__ == "__main__":
    # Argument parsing
    # parser = argparse.ArgumentParser(description="Train transformer model")
    # parser.add_argument("--model_id", type=int, help="Model identifier", required=True)
    # args = parser.parse_args()

    model_name = "kuleshov-group/bd3lm-owt-block_size4"
    max_length = 128
    batch_size = 64
    model_specs = [
        config_classes.ModelSpec(
            model_id=model_name,
            max_length=max_length,
            device="cuda",
            myopic_temperature=float(myopic_temperature),
        )
        # for myopic_temperature in torch.linspace(0.1, 3.1, 31)
        for myopic_temperature in torch.logspace(math.log10(0.1), 0.5, 30 + 1)
    ]
    base_model_spec = model_specs[20]

    offset = 10**5
    size = 9000
    dataloader_spec = config_classes.OpenWebTextLoaderSpec(
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        slice_start=offset,
        slice_end=offset + size,
    )
    # print(
    #     [model_spec.fingerprint for model_spec in model_specs],
    #     dataloader_spec.fingerprint,
    # )
    # for model_spec in model_specs:
    #     cache_payload = {
    #         "fn": "get_full_log_exp_likelihood_cached_v1",
    #         "model": model_spec.fingerprint(),
    #         "data": dataloader_spec.fingerprint(),
    #         "extra": {},
    #     }
    #     cache_key = config_classes._stable_hash(cache_payload)

    # id = args.model_id
    out_csv = run_battery_to_csv(
        model_specs=model_specs,  # [id : id + 1],
        base_model_spec=base_model_spec,
        data_spec=dataloader_spec,
        out_dir=Path("/m/cs/work/scheufh1/results"),
        cache_dir=Path("/m/cs/work/scheufh1/.cache/bd3lm_eval"),
        save_pair_csvs=True,
        pair_csv_dir=Path("/m/cs/scratch/temperature_diffusion/results/pair_csvs"),
    )
    print("Wrote:", out_csv)
