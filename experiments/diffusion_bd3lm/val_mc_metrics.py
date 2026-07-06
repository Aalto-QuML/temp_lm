from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from evaluation.sequence_metrics import (
    log_expected_likelihood_and_elbo_mc,
    sample_permutations,
    sequence_likelihood_ratios,
)
from plotting.evaluate_perplexity_temp_scaling import (
    regress_effective_temperature_scaling,
    kendall_tau,
)


def run_mc_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    *,
    block_size: int,
    mc_samples: int,
    mask_token: int,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Run MC validation and return aggregate metrics.

    Expects val_loader batches to contain:
      - input_ids
      - attention_mask
      - ref_mu (baseline MC logL)
    """
    if device is None:
        device = next(model.parameters()).device
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    model.eval()
    model_logLs = []
    baseline_logLs = []

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.autocast("cpu", dtype=torch.float32)
    )

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                torch.bool
            ).to(device, non_blocking=True)
            baseline_logL = batch["ref_mu"].to(device, non_blocking=True)

            permutations = sample_permutations(
                mc_samples, input_ids, block_size=block_size
            )

            with amp_ctx:
                logL, _ = log_expected_likelihood_and_elbo_mc(
                    model,
                    input_ids,
                    attention_mask,
                    block_size=block_size,
                    num_samples=mc_samples,
                    mask_token=mask_token,
                    permutations=permutations,
                )

            model_logLs.append(logL.detach())
            baseline_logLs.append(baseline_logL.detach())

    model_logL_all = torch.cat(model_logLs)
    baseline_logL_all = torch.cat(baseline_logLs)

    model_ratio = sequence_likelihood_ratios(model_logL_all, "full")
    baseline_ratio = sequence_likelihood_ratios(baseline_logL_all, "full")

    val_ratio = baseline_ratio / model_ratio
    val_avg_nll = (-model_logL_all.sum() / len(model_logL_all)).item()

    effective_temperature = regress_effective_temperature_scaling(
        baseline_logL_all, model_logL_all, pairing_mode="full"
    )
    tau = kendall_tau(
        baseline_logL_all, model_logL_all, pairing_mode="full"
    )

    return {
        "val_ratio": val_ratio.median().item(),
        "val_avg_nll": val_avg_nll,
        "val_ratio_mean": val_ratio.mean().item(),
        "val_ratio_std": val_ratio.std().item(),
        "val_effective_temperature": effective_temperature,
        "val_kendall_tau": tau,
    }
