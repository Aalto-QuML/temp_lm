from typing import List
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from pathlib import Path


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
from utils.cached_reference_dataloader import get_reference_augmented_dataloader
import utils.config_classes as config_classes

from evaluation.sequence_metrics import (
    PairingMode,
    sequence_likelihood_ratios,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def calc_avg_perplexity(
    model_likelihoods: Float[torch.Tensor, "M"],
) -> Float[torch.Tensor, "1"]:

    return torch.exp(-model_likelihoods).sum() / len(model_likelihoods)  # type: ignore


@torch.inference_mode()
def calc_likelihood_log_ratio_scaling(
    base_model_likelihoods: Float[torch.Tensor, "M"],
    target_model_differences: Float[torch.Tensor, "M"],
    pairing_mode: PairingMode,
    minimal_abs_ratio_difference: float = 1,
):
    """
    Calculates the scaling in log_ratios i.e. the avg of base_model log_ratio / target_model log_ratio
    returns average and variance
    """
    base_differences = sequence_likelihood_ratios(base_model_likelihoods, pairing_mode)
    target_model_differences = sequence_likelihood_ratios(
        target_model_differences, pairing_mode
    )
    good_ratios = (base_differences >= minimal_abs_ratio_difference) & (
        target_model_differences >= minimal_abs_ratio_difference
    )
    ratios = base_differences[good_ratios] / target_model_differences[good_ratios]
    return ratios.mean().item(), ratios.var().item()

@torch.inference_mode()
def regress_effective_temperature_scaling(
    base_model_likelihoods: Float[torch.Tensor, "M"],
    target_model_likelihoods: Float[torch.Tensor, "M"],
    pairing_mode: PairingMode,
) -> float:
    """
    Calculates the effective temperature scaling as least squares regression solution over the log ratios.
    """
    base_differences = sequence_likelihood_ratios(base_model_likelihoods,   pairing_mode)
    target_model_differences = sequence_likelihood_ratios(
        target_model_likelihoods, pairing_mode
    )
    return (base_differences*target_model_differences).sum() / (target_model_differences**2).sum()

@torch.inference_mode()
def kendall_tau(
    base_model_likelihoods: Float[torch.Tensor, "M"],
    target_model_likelihoods: Float[torch.Tensor, "M"],
    pairing_mode: PairingMode,
) -> float:
    """
    Calculates the Kendall tau correlation between the base model likelihoods and the target model likelihoods.
    """
    base_differences = sequence_likelihood_ratios(base_model_likelihoods,   pairing_mode)
    target_model_differences = sequence_likelihood_ratios(
        target_model_likelihoods, pairing_mode
    )
    return (base_differences.sign() * target_model_differences.sign()).mean().item()

if __name__ == "__main__":

    model_name = "kuleshov-group/bd3lm-owt-block_size4"
    max_length = 64
    model_spec = config_classes.ModelSpec(
        model_id=model_name, max_length=max_length, device="cuda"
    )
    lazy_model = model_spec.lazy()

    dataloader_spec = config_classes.OpenWebTextLoaderSpec(
        max_length=64, batch_size=1024, shuffle=False, slice_start=0, slice_end=9000
    )
    dataloader = dataloader_spec.lazy()
    print("testing laziness")
    # temp_model = model
    with torch.nn.attention.sdpa_kernel(
        [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]
    ), torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ref_dl = get_reference_augmented_dataloader(
            reference_model=lazy_model,
            data=dataloader_spec.lazy(),
            block_size=4,
            mask_token=50257,
            sparse_inference=True,
            cumulative=True,
        )

        for batch in ref_dl:
            print(batch)

    # with torch.backends.cuda.sdp_kernel(
    #     enable_flash=True, enable_mem_efficient=True, enable_math=True
    # ), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

    #     for myopic_temperature in torch.linspace(0.1, 3.1, 31):
    #         model_spec = config_classes.ModelSpec(
    #             model_id=model_name,
    #             max_length=max_length,
    #             device="cuda",
    #             myopic_temperature=float(myopic_temperature),
    #         )
    #         lazy_model = model_spec.lazy()
    #         base_log_likelihoods = config_classes.get_full_log_exp_likelihood_cached(
    #             lazy_model,
    #             dataloader,
    #             ignore_cache=myopic_temperature >= 2.5,
    #             block_size=8,
    #         )

    #         print(
    #             myopic_temperature,
    #             base_log_likelihoods,
    #             model_spec.fingerprint,
    #             dataloader_spec.fingerprint,
    #         )
