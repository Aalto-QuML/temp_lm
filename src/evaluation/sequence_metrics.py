# this module defines functions that measure sequence likelihoods according to a given model
# B always denotes a batch size
# V is the vocab size
# N is the sequence length, for inference the model expects 2N tokens with the first half clean, then noisy

import math
from typing import Literal, Tuple, Union, Optional
import torch
from jaxtyping import Float, Bool, Int64
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

from src.utils.block_diffusion_classes import SparseBD3LMWrapper

MASK_TOKEN = 50257


def all_transition_likelihoods_batched(
    block_diffusion_model: torch.nn.Module,
    sequences,
    attention_mask,
    block_size: int = 4,
    mask_token: int = MASK_TOKEN,
    sparse_inference: bool = True,
    subset_chunk: int | None = None,
):
    num_subsets = 2**block_size
    B, N = sequences.shape
    num_blocks = math.ceil(N / block_size)
    if subset_chunk is None:
        subset_chunk = num_subsets

    # IMPORTANT: correct dtype/value
    mask_tensor = torch.tensor(
        mask_token, device=sequences.device, dtype=sequences.dtype
    )
    sigma = torch.ones(B, device=sequences.device)

    sparse = None
    if sparse_inference:
        sparse = SparseBD3LMWrapper(
            block_diffusion_model, vocab_chunk=4096, force_float32_softmax=False
        ).to(sequences.device)

    bitpos = torch.arange(block_size, device=sequences.device)
    bit_table = (
        (torch.arange(num_subsets, device=sequences.device)[:, None] >> bitpos) & 1
    ).bool()
    patterns = (
        bit_table[:, None, :]
        .expand(num_subsets, num_blocks, block_size)
        .reshape(num_subsets, num_blocks * block_size)
    )[:, :N]

    chunks = []  # <-- training-safe accumulation

    for s0 in range(0, num_subsets, subset_chunk):
        s1 = min(s0 + subset_chunk, num_subsets)
        Sg = s1 - s0

        pat = patterns[s0:s1]  # (Sg, N)
        masked = attention_mask[:, None, :] & pat[None, :, :]  # (B, Sg, N)

        first = torch.where(masked, mask_tensor, sequences[:, None, :])  # (B, Sg, N)
        inputs = torch.empty(
            (B * Sg, 2 * N), device=sequences.device, dtype=sequences.dtype
        )
        inputs[:, :N] = first.reshape(B * Sg, N)
        inputs[:, N:] = sequences[:, None, :].expand(B, Sg, N).reshape(B * Sg, N)

        sigma_rep = sigma.repeat_interleave(Sg, dim=0)

        if sparse_inference:
            logp = sparse.forward_masked_target_logprobs(
                input_ids=inputs,
                timesteps=sigma_rep,
                targets=sequences[:, None, :].expand(B, Sg, N).reshape(B * Sg, N),
                masked=masked.reshape(B * Sg, N),
                sample_mode=False,
            )
        else:
            logits = block_diffusion_model(inputs, sigma_rep, sample_mode=False)[:, :, :-1]
            targets = sequences[:, None, :].expand(B, Sg, N).reshape(B * Sg, N)
            logp = (
                logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            )

        logp = logp.reshape(B, Sg, N)

        # training-safe: no in-place write into a preallocated buffer
        chunks.append(logp.masked_fill(~masked, float("-inf")))

        del masked, first, inputs, sigma_rep, logp

    return torch.cat(chunks, dim=1)  # (B, num_subsets, N)


def expected_elbo(
    transition_probs: Float[torch.Tensor, "B num_subsets N"],
    block_size: int = 4,
    length_normalize: bool = True,
) -> Float[torch.Tensor, "B"]:

    B, num_subsets, seq_len = transition_probs.shape
    device = transition_probs.device
    dtype = transition_probs.dtype

    # Hamming weight k of each subset index in {0, ..., 2**block_size - 1}
    subset_indices = torch.arange(num_subsets, device=device, dtype=torch.int64)
    subset_sizes: Int64[torch.Tensor, "num_subsets"] = _bitcount(subset_indices).to(
        dtype
    )

    # Weight for a subset of size k is proportional to 1 / k!
    # (log-space: -lgamma(k+1))
    log_weights = -(
        torch.lgamma(torch.Tensor([block_size + 1]).to(device))
        - torch.lgamma(subset_sizes + 1)
        - torch.lgamma(block_size - subset_sizes + 1)
    )  # (num_subsets,)
    weights = (torch.exp(log_weights) / subset_sizes).view(
        1, num_subsets, 1
    )  # (1, num_subsets, 1)

    # Positions where the subset does not predict this token were set to -inf;
    # for E[log p] they should contribute 0, not -inf.
    finite_mask = torch.isfinite(transition_probs)
    contrib = torch.where(
        finite_mask,
        weights * transition_probs,  # w_k * log p
        torch.zeros_like(transition_probs),
    )

    # Sum over subsets and positions -> (B,)
    # (optionally divide by seq_len if you want per-token ELBO instead of total)
    elbo = contrib.sum(dim=(1, 2)) / (seq_len if length_normalize else 1.0)

    return elbo


def full_log_expected_likelihood(
    block_diffusion_model: torch.nn.Module,
    sequences: Int64[torch.Tensor, "B N"],
    attention_mask: Bool[torch.Tensor, "B N"],
    block_size: int = 4,
    mask_token: int = MASK_TOKEN,
    permutation_normalize: bool = True,
    length_normalize: bool = True,
) -> Float[torch.Tensor, "B"]:
    transition_probs: Float[torch.Tensor, "B num_subsets N"] = (
        all_transition_likelihoods_batched(
            block_diffusion_model,
            sequences,
            attention_mask,
            block_size,
            mask_token,
            subset_chunk=16,
        )
    )
    return log_expected_likelihood(
        transition_probs, block_size, permutation_normalize, length_normalize
    )


def log_expected_likelihood(
    transition_probs: Float[torch.Tensor, "B num_subsets N"],
    block_size: int = 4,
    permutation_normalize: bool = True,
    length_normalize: bool = True,
) -> Float[torch.Tensor, "B"]:

    sequence_length = transition_probs.shape[2]
    total_log_prob = _transition_probability_sum(
        transition_probs, normalize=permutation_normalize, block_size=block_size
    )
    return total_log_prob[:, 0, :].sum(dim=-1) / (
        sequence_length if length_normalize else 1
    )


def sample_permutations(
    num_samples: int,
    sequences: Int64[torch.Tensor, "B N"],
    block_size: int,
    generator: Optional[torch.Generator] = None,
) -> Int64[torch.Tensor, "K B N"]:
    """Sample K random block permutations for each sequence."""
    B, N = sequences.shape
    device = sequences.device
    M = N // block_size
    rand = torch.rand((num_samples, B, M, block_size), device=device, generator=generator)
    return torch.argsort(rand, dim=3).flatten(2)


def log_expected_likelihood_mc(
    block_diffusion_model: torch.nn.Module,
    sequences: Int64[torch.Tensor, "B N"],
    attention_mask: Bool[torch.Tensor, "B N"],
    block_size: int = 4,
    num_samples: int = 8,
    mask_token: int = MASK_TOKEN,
    sparse_inference: bool = True,
    permutations: Optional[Int64[torch.Tensor, "K B N"]] = None,
    permutation_generator: Optional[torch.Generator] = None,
    length_normalize: bool = True,
) -> Float[torch.Tensor, "B"]:
    """
    Monte Carlo estimate of log E_{pi}[p(x; pi)] from K sampled permutations.

    Args:
        block_diffusion_model: Model used to score masked sequences.
        sequences: Token ids (B, N).
        attention_mask: Attention mask (B, N).
        block_size: Block size used for permutations.
        num_samples: Number of permutation samples K.
        mask_token: Token id used for masking.
        sparse_inference: Use sparse wrapper for log prob computation.
        permutations: Optional pre-sampled permutations, shape (K, B, N) or (B, K, N).
        length_normalize: If True, return per-token average logL.

    Returns:
        logL estimate per sequence, shape (B,).
    """
    if not sparse_inference:
        raise NotImplementedError("log_expected_likelihood_mc requires sparse_inference")

    B, N = sequences.shape
    if N % block_size != 0:
        raise ValueError("Sequence length must be divisible by block_size")

    device = sequences.device
    perm = _resolve_permutations(
        permutations,
        num_samples=num_samples,
        sequences=sequences,
        block_size=block_size,
        generator=permutation_generator,
    )
    num_samples = perm.shape[0]

    logp_perm = torch.stack(
        [
            one_order_probabilities(
                block_diffusion_model,
                sequences,
                attention_mask,
                block_size=block_size,
                mask_token=mask_token,
                length_normalize=False,
                sparse_inference=sparse_inference,
                permutation=perm[k],
            )
            for k in range(num_samples)
        ],
        dim=0,
    )

    log_mean = torch.logsumexp(logp_perm, dim=0) - torch.log(
        torch.tensor(float(num_samples), device=device)
    )
    if length_normalize:
        return log_mean / N
    return log_mean


def expected_elbo_mc(
    block_diffusion_model: torch.nn.Module,
    sequences: Int64[torch.Tensor, "B N"],
    attention_mask: Bool[torch.Tensor, "B N"],
    block_size: int = 4,
    num_samples: int = 8,
    mask_token: int = MASK_TOKEN,
    sparse_inference: bool = True,
    permutations: Optional[Int64[torch.Tensor, "K B N"]] = None,
    permutation_generator: Optional[torch.Generator] = None,
    length_normalize: bool = True,
) -> Float[torch.Tensor, "B"]:
    """
    Monte Carlo estimate of ELBO = E_{pi}[log p(x; pi)] using K permutations.
    """
    if not sparse_inference:
        raise NotImplementedError("expected_elbo_mc requires sparse_inference")

    B, N = sequences.shape
    if N % block_size != 0:
        raise ValueError("Sequence length must be divisible by block_size")

    perm = _resolve_permutations(
        permutations,
        num_samples=num_samples,
        sequences=sequences,
        block_size=block_size,
        generator=permutation_generator,
    )
    num_samples = perm.shape[0]

    logp_perm = torch.stack(
        [
            one_order_probabilities(
                block_diffusion_model,
                sequences,
                attention_mask,
                block_size=block_size,
                mask_token=mask_token,
                length_normalize=length_normalize,
                sparse_inference=sparse_inference,
                permutation=perm[k],
            )
            for k in range(num_samples)
        ],
        dim=0,
    )
    return logp_perm.mean(dim=0)


def log_expected_likelihood_and_elbo_mc(
    block_diffusion_model: torch.nn.Module,
    sequences: Int64[torch.Tensor, "B N"],
    attention_mask: Bool[torch.Tensor, "B N"],
    block_size: int = 4,
    num_samples: int = 8,
    mask_token: int = MASK_TOKEN,
    sparse_inference: bool = True,
    permutations: Optional[Int64[torch.Tensor, "K B N"]] = None,
    permutation_generator: Optional[torch.Generator] = None,
    length_normalize: bool = True,
) -> Tuple[Float[torch.Tensor, "B"], Float[torch.Tensor, "B"]]:
    """
    Returns (log E_pi[p(x;pi)], E_pi[log p(x;pi)]) from shared MC logp.
    """
    if not sparse_inference:
        raise NotImplementedError(
            "log_expected_likelihood_and_elbo_mc requires sparse_inference"
        )

    B, N = sequences.shape
    if N % block_size != 0:
        raise ValueError("Sequence length must be divisible by block_size")

    device = sequences.device
    perm = _resolve_permutations(
        permutations,
        num_samples=num_samples,
        sequences=sequences,
        block_size=block_size,
        generator=permutation_generator,
    )
    num_samples = perm.shape[0]

    logp_perm = torch.stack(
        [
            one_order_probabilities(
                block_diffusion_model,
                sequences,
                attention_mask,
                block_size=block_size,
                mask_token=mask_token,
                length_normalize=False,
                sparse_inference=sparse_inference,
                permutation=perm[k],
            )
            for k in range(num_samples)
        ],
        dim=0,
    )

    log_mean = torch.logsumexp(logp_perm, dim=0) - torch.log(
        torch.tensor(float(num_samples), device=device)
    )
    elbo_mean = logp_perm.mean(dim=0)
    if length_normalize:
        return log_mean / N, elbo_mean / N
    return log_mean, elbo_mean


def _resolve_permutations(
    permutations: Optional[Int64[torch.Tensor, "K B N"]],
    *,
    num_samples: int,
    sequences: Int64[torch.Tensor, "B N"],
    block_size: int,
    generator: Optional[torch.Generator] = None,
) -> Int64[torch.Tensor, "K B N"]:
    B, N = sequences.shape
    device = sequences.device
    if permutations is None:
        return sample_permutations(
            num_samples, sequences, block_size=block_size, generator=generator
        )

    perm = permutations
    if perm.shape == (B, num_samples, N):
        perm = perm.permute(1, 0, 2)
    if perm.shape[1:] != (B, N):
        raise ValueError(f"Expected permutations shape (K,B,N), got {perm.shape}")
    return perm


def one_order_probabilities(
    block_diffusion_model: torch.nn.Module,
    sequences: Int64[torch.Tensor, "B N"],
    attention_mask: Bool[torch.Tensor, "B N"],
    block_size: int = 4,
    mask_token: int = MASK_TOKEN,
    length_normalize: bool = True,
    sparse_inference: bool = True,
    permutation: Optional[Int64[torch.Tensor, "B N"]] = None,
) -> Float[torch.Tensor, "B"]:
    """
    Computes the log-likelihood along a single unmasking order per sequence.

    Args:
        block_diffusion_model: Model used to score masked sequences.
        sequences: Token ids (B, N).
        attention_mask: Attention mask (B, N).
        block_size: Block size used for permutations.
        mask_token: Token id used for masking.
        length_normalize: If True, return per-token average logL.
        sparse_inference: Use sparse wrapper for log prob computation.
        permutation: Optional pre-sampled permutation, shape (B, N).

    Returns:
        logL per sequence, shape (B,).
    """
    if not sparse_inference:
        raise NotImplementedError("one_order_probabilities requires sparse_inference")

    B, N = sequences.shape
    K = block_size
    M = N // K
    device = sequences.device

    sigma = torch.ones(B, device=sequences.device)
    sparse = SparseBD3LMWrapper(
        block_diffusion_model, vocab_chunk=4096, force_float32_softmax=True
    ).to(sequences.device)

    if permutation is None:
        permutation = torch.argsort(
            torch.rand((B, M, K), device=device), dim=2
        ).flatten(1)
    else:
        if permutation.shape != (B, N):
            raise ValueError(
                f"Expected permutation shape {(B, N)} but got {permutation.shape}"
            )
    assert permutation.shape == (B, N)
    assert permutation.max() == K - 1

    logp_sum = torch.zeros(B, device=device)
    mask = torch.ones((B, N), device=device, dtype=torch.bool)
    for i in range(K):
        masked_sequences = torch.where(
            attention_mask.bool() & mask,
            torch.full_like(sequences, mask_token),
            sequences,
        )
        log_probs = sparse.forward_masked_target_logprobs(
            input_ids=torch.cat((masked_sequences, sequences), dim=1),
            timesteps=sigma,
            targets=sequences,
            masked=masked_sequences,
            sample_mode=False,
        )
        logp_sum += torch.nan_to_num(log_probs * (permutation == i)).sum(dim=1)
        mask[permutation == i] = False

    if length_normalize:
        return logp_sum / N
    return logp_sum



def _subset_convolution_step(
    logp_full: Float[torch.Tensor, "B 2**N-1 N num_blocks"],
    F_prev_full: Float[torch.Tensor, "B 2**N-1 num_blocks"],
    k: int,
) -> Float[torch.Tensor, "B 2**N-1 num_blocks"]:
    """
    logp_full: (2**n, n) tensor, logp_full[mask, i] = log p(i | mask).
              (typically set to -inf when i is already in mask)
    F_prev_full: (2**n,) tensor holding F on size-k masks and -inf elsewhere
    k: current subset size

    Returns:
      masks_k1: (num_{k+1},) int64 masks of size k+1
      F_next:   (num_{k+1},) F on those masks, in the same order
    """
    device = logp_full.device
    n = logp_full.shape[2]
    N = 1 << n
    num_blocks = logp_full.shape[3]
    batch_size = logp_full.shape[0]
    # print("block size:", n)
    masks0 = torch.arange(N, device=device, dtype=torch.int64)
    pop = _bitcount(masks0)

    k_indices = pop == (k)  # next indices, not calced yet
    k1_indices = pop == (k + 1)  # successor indices, already calced
    # print(masks_k1)
    bitpos = torch.arange(n, device=device, dtype=torch.int64)
    assert bitpos.shape[0] == n, bitpos.shape
    bitmasks = torch.zeros(n, dtype=torch.bool, device=logp_full.device)
    bitmasks[bitpos] = 1  # (n,)
    kernel = masks0[k_indices, None] | (1 << bitpos) == masks0[k1_indices, None, None]
    assert kernel.shape == (k1_indices.sum(), k_indices.sum(), n)  # nice!
    assert F_prev_full.shape[1] == len(k1_indices)
    transitioned_likelihoods = torch.einsum(  # let g=k1
        "bgl,gkn->bknl", F_prev_full[:, k1_indices, :], kernel.to(F_prev_full.dtype)
    )  # B k n l

    vals = (
        torch.nan_to_num(
            transitioned_likelihoods, nan=0.0, neginf=-float("inf")
        )  # (B, k, n,l)
        + logp_full[:, (masks0 ^ ((1 << n) - 1))[k_indices], :, :]  # (B, k, n, l)
    )  # (B, k, n, l)
    assert vals.shape == (batch_size, k_indices.sum(), n, num_blocks)
    F_next = torch.logsumexp(
        vals, dim=2
    )
    F_prev_full[:, k_indices, :] = F_next
    return F_prev_full


def _bitcount(masks: Int64[torch.Tensor, "N"]) -> Int64[torch.Tensor, "N"]:
    # SWAR popcount
    masks = masks - ((masks >> 1) & 0x5555555555555555)
    masks = (masks & 0x3333333333333333) + ((masks >> 2) & 0x3333333333333333)
    masks = (masks + (masks >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (masks * 0x0101010101010101) >> 56  # popcount per element, uint64


def _transition_probability_sum(
    transition_probs: Float[torch.Tensor, "B num_subsets N"],
    block_size: int = 4,
    normalize: bool = True,
) -> Float[torch.Tensor, "B num_subsets num_blocks"]:
    """
    Docstring for _transition_probability_sum
    num_subset = 2**block_size-1
    :param transition_probs: Description
    :type transition_probs: Float[torch.Tensor, "B num_subsets N"]
    :param block_size: Description
    :param normalize: divides the output by the subset size
    :type block_size: int

    """
    batch_size, num_subsets, sequence_length = transition_probs.shape
    num_blocks = sequence_length // block_size
    blocked_probs = transition_probs.view(
        batch_size, num_subsets, num_blocks, block_size
    ).transpose(2, 3)
    # blocked_probs = torch.nn.functional.pad(
    #     blocked_probs, (0, 0, 0, 0, 0, 1), value=float("-inf")
    # )
    # the dp table should be of size equal to transition probs?
    # compute block-batched expected prob over all permutations
    f = torch.full(
        (batch_size, num_subsets, num_blocks),
        -torch.inf,
        device=transition_probs.device,
        dtype=transition_probs.dtype,
    )
    f[:, -1, :] = 0.0
    for prefix_size in reversed(range(block_size)):
        # input(f[0, :, 0])

        f = _subset_convolution_step(blocked_probs, f, prefix_size)
        # f[:, prefix_size, :][f[:, prefix_size, :] == float("-inf")] = 0

        assert f.shape == (batch_size, num_subsets, num_blocks)
    # compute sums over all blocks for the full sequence
    if normalize:
        f = (
            f
            - torch.lgamma(
                block_size
                - _bitcount(torch.arange(1 << block_size, device=f.device))
                + 1
            )[None, :, None]
        )
    return f


PairingMode = Union[Literal["pairs", "full"], int]


def sequence_likelihood_ratios(
    logL: Float[torch.Tensor, "B"],
    pairing_mode: PairingMode = "pairs",
) -> Float[torch.Tensor, "K"]:
    """
    pairing_mode:
      - "pairs": (0,1), (2,3), ... -> returns per-item ratio vs its partner (last item gets 0 if unpaired)
      - "full":  all vs all
      - int k:   for each i, sample k partners j, return mean_j (logL[i] - logL[j])
    """

    B = logL.shape[0]
    out: Float[torch.Tensor, "B"] = torch.zeros_like(logL)

    match pairing_mode:
        case "pairs":
            m = (B // 2) * 2
            a = torch.arange(0, m, 2, device=logL.device)
            b = a + 1
            out[a] = logL[a] - logL[b]
            out[b] = logL[b] - logL[a]
            return out

        case "full":
            logR = logL[:, None] - logL[None, :]
            logR = logR[~torch.eye(B, dtype=torch.bool, device=logL.device)]
            return logR

        case k if isinstance(k, int):
            # sample k partners for each i, compute mean_j (logL[i] - logL[j])
            idx = torch.arange(B, device=logL.device)[:, None].expand(B, k)
            partners = torch.randint(0, B - 1, (B, k), device=logL.device)
            partners = partners + (partners >= idx).long()  # avoid self by shifting
            return (logL[idx] - logL[partners]).mean(dim=1)

        case _:
            raise ValueError("pairing_mode must be 'pairs', 'full', or an int")

    pass


def _transition_score_mean_var(
    transition_probs: Float[
        torch.Tensor, "B num_subsets N"
    ],  # (B, 2**n, N) as in your code
    block_size: int = 4,
) -> tuple[
    Float[torch.Tensor, "B num_subsets num_blocks"],
    Float[torch.Tensor, "B num_subsets num_blocks"],
]:
    """
    Returns (mean_S, var_S) per batch element for uniform permutations.
    Assumes N is a multiple of block_size (same assumption as your DP).
    """
    B, num_subsets, seq_len = transition_probs.shape
    n = block_size
    full = (1 << n) - 1
    num_blocks = seq_len // n

    # (B, 2**n, num_blocks, n) -> (B, 2**n, n, num_blocks)
    logp = transition_probs.view(B, num_subsets, num_blocks, n).transpose(2, 3)

    masks = torch.arange(1 << n, device=logp.device, dtype=torch.int64)
    pop = _bitcount(masks)
    comp = masks ^ full

    # DP tables over masks, per block
    mu = torch.zeros((B, 1 << n, num_blocks), device=logp.device, dtype=torch.float64)
    m2 = torch.zeros((B, 1 << n, num_blocks), device=logp.device, dtype=torch.float64)

    # base state: mask=full already has 0,0

    bitpos = torch.arange(n, device=logp.device, dtype=torch.int64)

    for k in reversed(range(n)):  # k = n-1 ... 0
        idx_k = (pop == k).nonzero(as_tuple=True)[0]  # masks of size k
        m_k = masks[idx_k]  # (K,)
        r = n - k  # remaining count (scalar)

        # next masks for each choice bit: (K, n)
        next_masks = m_k[:, None] | (1 << bitpos)[None, :]

        # which bits are actually addable: only those not already in mask
        addable = (m_k[:, None] & (1 << bitpos)[None, :]) == 0  # (K, n) bool

        # gather a_i = log p(i | mask) with your complement convention:
        # logp: (B, subset, i, block); subset index is comp(mask)
        a = logp[:, comp[idx_k], :, :]  # (B, K, n, num_blocks)
        a = a.to(torch.float64)

        # gather mu/m2 for next state per choice
        mu_next = mu[:, next_masks, :]  # (B, K, n, num_blocks)
        m2_next = m2[:, next_masks, :]  # (B, K, n, num_blocks)

        # mask out non-addable choices (so they don't contribute)
        # set them to 0 before summing, since we average over only addable count=r
        addable_f = addable[None, :, :, None]  # (1,K,n,1)
        valid = addable_f & torch.isfinite(a)  # (B,K,n,num_blocks)
        den = valid.sum(dim=2).clamp(min=1).to(a.dtype)  # (B,K,num_blocks)

        mu[:, idx_k, :] = ((a + mu_next).masked_fill(~valid, 0).sum(dim=2)) / den
        m2[:, idx_k, :] = (
            (a * a + 2 * a * mu_next + m2_next).masked_fill(~valid, 0).sum(dim=2)
        ) / den

        # term_mu = torch.nan_to_num((a + mu_next) * addable_f)
        # mu[:, idx_k, :] = term_mu.sum(dim=2) / r

        # term_m2 = torch.nan_to_num((a * a + 2 * a * mu_next + m2_next) * addable_f)
        # m2[:, idx_k, :] = term_m2.sum(dim=2) / r

    # start from empty mask=0, sum over blocks (same way you sum block contributions)
    # mean_S_blocks = mu[:, 0, :]  # (B, num_blocks)
    # m2_blocks = m2[:, 0, :]  # (B, num_blocks)
    # var_blocks = m2_blocks - mean_S_blocks**2  # (B, num_blocks)

    # mean_S = mean_S_blocks.sum(dim=-1)  # (B,)
    # var_S = var_blocks.sum(dim=-1).clamp_min(0.0)  # (B,)
    # return mean_S.to(transition_probs.dtype), var_S.to(transition_probs.dtype)
    assert mu.shape == (B, num_subsets, num_blocks), mu.shape
    return mu.to(transition_probs.dtype), m2.to(
        transition_probs.dtype
    )  # has shape B, num_subsets


def subset_permutation_means_variances(
    transition_probs: Float[torch.Tensor, "B num_subsets N"],
    block_size: int = 4,
    cumulative: bool = True,
) -> Tuple[
    Float[torch.Tensor, "B num_subsets num_blocks"],
    Float[torch.Tensor, "B num_subsets num_blocks"],
]:
    mu_table, m2_table = _transition_score_mean_var(
        transition_probs=transition_probs, block_size=block_size
    )
    mu_table = mu_table.clamp(max=0.0)
    m2_table = m2_table.clamp(min=0.0)
    var_table = m2_table - mu_table**2
    mean_table = _transition_probability_sum(transition_probs, block_size=block_size)

    if cumulative:
        return _suffix_cumulate_blocks(mean_table), _suffix_cumulate_blocks(var_table)
    return mean_table, var_table


def _suffix_cumulate_blocks(dp_table: torch.Tensor) -> torch.Tensor:
    # dp_table: (B, S, num_blocks)
    full_block = dp_table[
        :, 0, :
    ]  # subset=0 in your NEW DP means "empty chosen mask" = full remaining block
    suffix = torch.flip(torch.cumsum(torch.flip(full_block, dims=[1]), dim=1), dims=[1])
    offset = torch.zeros_like(full_block)
    offset[:, :-1] = suffix[:, 1:]  # sum of future blocks only
    return dp_table + offset[:, None, :]
