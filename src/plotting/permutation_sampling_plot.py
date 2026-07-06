# Plot that takes a small batch of sequences and *uniformly* estimates unmasking permutations (that adhere to the block structure).
# Needs a modelspec and a dataspec and then needs to get the transition probabilities, sample permutations, and get the corresponding likelihood from the table.
from pathlib import Path
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
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


from evaluation import sequence_metrics

import math
import torch

from typing import Tuple
from jaxtyping import Float

import utils.config_classes as config_classes
from evaluation.sequence_metrics import (
    MASK_TOKEN,
    all_transition_likelihoods_batched,
)


def permutation_sampling_running_LSE(
    transition_probs: Float[torch.Tensor, "B num_subsets N"],
    block_size: int = 4,
    num_permutations: int = 10000,  # M
    seed: int = 4,
) -> Float[torch.Tensor, "B M"]:

    B, num_subsets, N = transition_probs.shape
    num_blocks = N // block_size
    device = transition_probs.device
    M = num_permutations

    # set random seed in torch
    torch.manual_seed(seed)

    # obtain M permutations: per block, sequential over blocks
    perms = torch.stack(
        [torch.randperm(block_size, device=device) for _ in range(M * num_blocks)]
    ).reshape(M, num_blocks, block_size)
    inv = perms.argsort(-1)  # step at which each position is unmasked

    # k = subset index whose binary repr indicates which positions are still masked
    # for position p: bits set for positions q where step(q) > step(p)
    later = inv.unsqueeze(-1) > inv.unsqueeze(-2)
    bits = 2 ** torch.arange(block_size, device=device)
    k = (later * bits).sum(-1).reshape(M, N).long()

    # gather: log_probs[b, m, n] = transition_probs[b, k[m,n], n]
    b_idx = torch.arange(B, device=device)[:, None, None]
    n_idx = torch.arange(N, device=device)
    log_probs = transition_probs[b_idx, 15 - k[None], n_idx]

    # sum over N to get log-likelihood per permutation: (B, M)
    log_L = log_probs.sum(-1)

    # running log-mean-exp: cumulative LSE minus log count
    counts = torch.arange(1, M + 1, device=device, dtype=log_L.dtype)
    return torch.logcumsumexp(log_L, dim=1) - counts.log()
    # return torch.cumsum(log_L, dim=1) / counts


# def permutation_sampling_running_LSE(
#     transition_probs: Float[torch.Tensor, "B num_subsets N"],
#     block_size: int = 4,
#     mask_token: int = MASK_TOKEN,
#     num_permutations: int = 100,  # M
#     seed: int = 4,
#     block_restricted: bool = True,
# ) -> Float[torch.tensor, " B M"]:

#     B, num_subsets, N = transition_probs.shape
#     num_blocks = N // block_size

#     # set random seed in torch.

#     # obtain a set of M permutations uniform at random with the following restriction (if_block_restricted):
#     # permutation is obtained PER BLOCK, and sequential over blocks

#     # permutations are index vectors of shape [1 M N] (first dimension will be batched)
#     permutations=torch.zeros((1,num_permutations,N))

#     # per block: for permutations[:, i]  compute the sum of transisition_probs[B,k,i]
#     #IDEA: forget about the block_restricted flag and just do the permutation sampling in this loop.

#     # here k is the index whose binary representation corresponds to permutations[:,0:i]

#     # the result of this computation should be (B M num_blocks) and then we sum over num_blocks

#     # the B M matrix now contains the log-likelihood for each model under each permutation.

#     # we now return the cumulative-LSE over the permutations for each model and divide by the permutation index to get the running expectation.


#     pass


@torch.inference_mode()
def get_full_log_exp_likelihood_perm_plot(
    model: config_classes.LazyLoaded[torch.nn.Module],  # LazyLoaded[torch.nn.Module]
    data: config_classes.LazyLoaded[
        config_classes.DataLoader[config_classes.Batch]
    ],  # LazyLoaded[DataLoader[Batch]]
    block_size: int = 4,
    mask_token: int = MASK_TOKEN,
) -> Tuple[Float[torch.Tensor, "M"], Float[torch.Tensor, "M"]]:
    print(model.fingerprint(), data.fingerprint())
    mdl = model.get()
    dl: config_classes.DataLoader[config_classes.Batch] = data.get()

    # # Allow passing eval knobs through cache_key_extra, but keep this minimal.
    # block_size = extra.get("block_size", 4)
    # mask_token = extra.get("mask_token", None)

    device = next(mdl.parameters()).device
    chunks = []

    for batch in tqdm(dl):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        transition_probs: Float[torch.Tensor, "B num_subsets N"] = (
            all_transition_likelihoods_batched(
                mdl,
                input_ids,
                attention_mask.to(torch.bool),
                block_size,
                mask_token,
                sparse_inference=True,
                subset_chunk=16,
            )
        )

        chunks.append(transition_probs.detach().to("cpu"))

    logL = torch.cat(chunks, dim=0)  # (M,)

    return logL


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Train transformer model")
    # parser.add_argument("--model_id", type=int, help="Model identifier", required=True)
    # args = parser.parse_args()

    model_name = "kuleshov-group/bd3lm-owt-block_size4"
    # model_name = "/m/cs/work/scheufh1/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.1_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300/checkpoint_0.5287_14336"

    max_length = 64
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
    # base_model_spec = config_classes.ModelSpec(
    #     model_id=model_name,
    #     max_length=max_length,
    #     device="cuda",
    #     myopic_temperature=1.0,
    # )

    base_model_spec = config_classes.ModelSpec(
        model_id=model_name,
        max_length=max_length,
        device="cuda",
        myopic_temperature=0.5,
    )

    offset = 10**5
    size = 64

    dataloader_spec = config_classes.OpenWebTextLoaderSpec(
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        slice_start=offset,
        slice_end=offset + size,
    )

    transition_probs = get_full_log_exp_likelihood_perm_plot(
        base_model_spec.lazy(),
        dataloader_spec.lazy(),
    )
    print(transition_probs.shape)
    perms = permutation_sampling_running_LSE(transition_probs, num_permutations=1000)
    print(perms.shape)
    data = perms.numpy()  # or tensor.detach().cpu().numpy() if on GPU/requires grad
    np.savetxt("temp_myopic_model.csv", data, delimiter=",")

    print(data)
    plt.figure(figsize=(10, 4))
    for row in data:
        # plt.plot(np.abs(row - row[-1]), alpha=0.2, linewidth=0.5, color="blue")
        plt.plot((row - row[-1]), alpha=0.2, linewidth=0.5, color="blue")

    plt.xlabel("Number of Unmaskings")
    plt.ylabel("Running avg NLL deviation from mean (nats)")
    plt.show()

    # TRY THESE VARIANTS!!
    # WITH/WITHOUT ABS
    # WITH logcumsumexp in function above instead of normal sum
