from jaxtyping import Float, Int
from typing import Tuple, Dict
import torch
from evaluation.sequence_metrics import expected_elbo

MASK_TOKEN = 50257


def update_baseline(
    baseline: Dict[str, Float[torch.Tensor, "num_subsets num_blocks"]],
    baseline_logL: Float[torch.Tensor, "B num_subsets num_blocks"],
    baseline_var: Float[torch.Tensor, "B num_subsets num_blocks"],
) -> Tuple[
    Dict[str, Float[torch.Tensor, "num_subsets num_blocks"]],
    Float[torch.Tensor, "num_subsets num_blocks"],
    Float[torch.Tensor, "num_subsets num_blocks"],
]:
    """
    Updates the baseline statistics with the new batch log-likelihoods and variances.

    Args:
        baseline: Dictionary containing the current baseline statistics.
        baseline_logL: Log-likelihoods for the current batch.
        baseline_var: Variances for the current batch.
    Returns:
        updated_baseline: Updated baseline statistics.
    """
    batch_size, num_subsets, num_blocks = baseline_logL.shape

    batch_logL_sum: Float[torch.Tensor, "num_subsets num_blocks"] = baseline_logL.sum(
        dim=0
    )
    batch_var_sum: Float[torch.Tensor, "num_subsets num_blocks"] = baseline_var.sum(
        dim=0
    )

    # TODO: Should I a implement a version that works with padding tokens?
    print("logL sum", batch_logL_sum.shape)
    print("baseline", baseline["logL_sum"].shape)
    baseline["count"] += batch_size
    baseline["logL_sum"] += batch_logL_sum
    baseline["var_sum"] += batch_var_sum

    return (
        baseline,
        baseline["logL_sum"] / baseline["count"],
        baseline["var_sum"] / baseline["count"],
    )


def lths_loss(
    transition_probs: Float[torch.Tensor, "B num_subsets N"],
    block_length: int,
    temperature: float,
    baseline: Dict[str, Float[torch.Tensor, "num_subsets num_blocks"]],
    baseline_logL: Float[torch.Tensor, "B num_subsets num_blocks"],
    baseline_var: Float[torch.Tensor, "B num_subsets num_blocks"],
) -> Tuple[
    Float[torch.Tensor, "1"],
    Dict[str, Float[torch.Tensor, "num_subsets num_blocks"]],
]:
    """
    Calculates the LHTS loss given transition probabilities and baseline statistics.

    Args:
        transition_probs: Transition probabilities from the model.
        block_size: Size of the blocks used in the model.
        baseline_sum: Sum of baseline log-likelihoods.
        baseline_count: Count of baseline log-likelihoods.
        baseline_logL: Baseline log-likelihoods for the current batch.
        baseline_var: Variance of baseline log-likelihoods for the current batch.
    Returns:
        loss: The computed LHTS loss.
        updated_baseline_sum: Updated sum of baseline log-likelihoods.
        updated_baseline_count: Updated count of baseline log-likelihoods.
    """

    batch_size, num_subsets, num_blocks = baseline_logL.shape

    # Baseline
    baseline, baseline_mean_logL, baseline_mean_var = update_baseline(
        baseline, baseline_logL, baseline_var
    )

    # Calculate weights
    updated_logL = baseline_logL - baseline_mean_logL.unsqueeze(
        0
    )  # / baseline_mean_var.sqrt().unsqueeze(0).clamp(min=1e-5)
    # batch_max = baseline_logL.max(dim=0, keepdim=True).values
    # normalized_logL = baseline_logL - batch_max + 10.0 / (1 - temperature) * temperature
    exponent = (updated_logL * (1 - temperature) / temperature).clamp(max=10)
    weight: Float[torch.Tensor, "B num_subsets num_blocks"] = torch.exp(exponent)
    weight = weight.flip(dims=[1])

    # Calculate loss
    transition_probs_reshaped = transition_probs.view(
        batch_size, num_subsets, num_blocks, block_length
    )
    weighted_transition_probs = transition_probs_reshaped * weight.unsqueeze(-1)
    weighted_transition_probs = weighted_transition_probs.view(
        batch_size, num_subsets, -1
    )
    loss = -expected_elbo(weighted_transition_probs, block_size=block_length).mean()
    return loss, baseline, exponent


