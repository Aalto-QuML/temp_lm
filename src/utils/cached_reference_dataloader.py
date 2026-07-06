from __future__ import annotations

import json, hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, TypedDict, Tuple
from jaxtyping import Float

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config_classes import Batch, LazyLoaded
from evaluation.sequence_metrics import (
    log_expected_likelihood_mc,
    subset_permutation_means_variances,
)


def _stable_hash(obj: Any, algo: str = "sha256") -> str:
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    h = hashlib.new(algo)
    h.update(payload)
    return h.hexdigest()


class ReferenceAugmentedDataLoader:
    """
    Wraps a DataLoader and adds reference-model tensors to each yielded batch.

    Assumes dataloader order is deterministic and matches the cached pass order
    (i.e. shuffle=False, no random sampler).
    """

    def __init__(
        self,
        base: DataLoader[Batch],
        ref_mu: torch.Tensor,  # (M, num_subsets, N)
        ref_var: torch.Tensor,  # (M, num_subsets, N) or whatever you cache
    ):
        self.base = base
        self.ref_mu = ref_mu
        self.ref_var = ref_var

    def __len__(self) -> int:
        return len(self.base)

    def __iter__(self) -> Iterator[Batch]:
        offset = 0
        for batch in self.base:
            bsz = batch["input_ids"].shape[0]
            # Attach CPU tensors; training step can move to GPU as needed
            batch["ref_mu"] = self.ref_mu[offset : offset + bsz]
            batch["ref_var"] = self.ref_var[offset : offset + bsz]
            offset += bsz
            yield batch


@torch.inference_mode()
def get_reference_augmented_dataloader(
    reference_model: LazyLoaded[torch.nn.Module],
    data: LazyLoaded[DataLoader[Batch]],
    *,
    cache_dir: Path = Path(".cache/bd3lm_ref"),
    ignore_cache: bool = False,
    block_size: int = 4,
    mask_token: int = 50257,
    sparse_inference: bool = True,
    cumulative: bool = True,
) -> ReferenceAugmentedDataLoader:
    """
    Returns a dataloader that yields batches with extra keys:
      - "ref_mu"
      - "ref_var"

    Cache key includes:
      - reference_model.fingerprint()
      - data.fingerprint()
      - block_size/mask_token/sparse_inference/cumulative
      - function id
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_payload = {
        "fn": "ref_perm_mean_var_v1",  # <<< base-function id
        "model": reference_model.fingerprint(),
        "data": data.fingerprint(),
        "block_size": int(block_size),
        "mask_token": int(mask_token),
        "sparse_inference": bool(sparse_inference),
        "cumulative": bool(cumulative),
    }
    key = _stable_hash(key_payload)
    path = cache_dir / f"{key}.pt"

    if path.exists() and not ignore_cache:
        obj = torch.load(path, map_location="cpu")
        mu_all = obj["mu"]
        var_all = obj["var"]
        base_loader = data.get()
        return ReferenceAugmentedDataLoader(base_loader, mu_all, var_all)

    # cache miss: run reference model once over full dataset
    mdl = reference_model.get()
    dl = data.get()

    mu_chunks: List[Float[torch.Tensor, "B N"]] = []
    m2_chunks: List[Float[torch.Tensor, "B N"]] = []

    device = next(mdl.parameters()).device
    for batch in tqdm(dl, "Caching Reference Model Outputs"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        # Uses YOUR requested plumbing: this calls _transition_score_mean_var internally
        mu_b, m2_b = subset_permutation_means_variances(
            mdl,
            input_ids,
            attention_mask,
            block_size=block_size,
            mask_token=mask_token,
            sparse_inference=sparse_inference,
            cumulative=cumulative,
        )
        mu_chunks.append(mu_b.detach().to("cpu"))
        m2_chunks.append(m2_b.detach().to("cpu"))

    mu_all = torch.cat(mu_chunks, dim=0)
    m2_all = torch.cat(m2_chunks, dim=0)

    tmp = path.with_suffix(".pt.tmp")
    torch.save({"mu": mu_all, "var": m2_all, "meta": key_payload}, tmp)
    tmp.replace(path)  # atomic

    return ReferenceAugmentedDataLoader(dl, mu_all, m2_all)


@torch.inference_mode()
def get_reference_augmented_dataloader_mc(
    reference_model: LazyLoaded[torch.nn.Module],
    data: LazyLoaded[DataLoader[Batch]],
    *,
    cache_dir: Path = Path(".cache/bd3lm_ref_mc"),
    ignore_cache: bool = False,
    block_size: int = 4,
    mask_token: int = 50257,
    num_samples: int = 8,
    permutations: Optional[torch.Tensor] = None,
    permutation_key: Optional[str] = None,
) -> ReferenceAugmentedDataLoader:
    """
    Returns a dataloader that yields batches with extra keys:
      - "ref_mu"
      - "ref_var"

    Uses Monte Carlo log-expected-likelihood with fixed K.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_payload = {
        "fn": "ref_logL_mc_v1",
        "model": reference_model.fingerprint(),
        "data": data.fingerprint(),
        "block_size": int(block_size),
        "mask_token": int(mask_token),
        "num_samples": int(num_samples),
        "permutation_key": permutation_key,
    }
    key = _stable_hash(key_payload)
    path = cache_dir / f"{key}.pt"

    if path.exists() and not ignore_cache:
        obj = torch.load(path, map_location="cpu")
        mu_all = obj["mu"]
        var_all = obj["var"]
        base_loader = data.get()
        return ReferenceAugmentedDataLoader(base_loader, mu_all, var_all)

    mdl = reference_model.get()
    dl = data.get()

    device = next(mdl.parameters()).device
    permutation_generator = None
    if permutations is None and permutation_key is not None:
        seed = int(hashlib.sha256(permutation_key.encode()).hexdigest()[:8], 16)
        permutation_generator = torch.Generator(device=device)
        permutation_generator.manual_seed(seed)

    mu_chunks: List[Float[torch.Tensor, "B"]] = []
    for batch in tqdm(dl, "Caching Reference MC LogL"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        logL_b = log_expected_likelihood_mc(
            mdl,
            input_ids,
            attention_mask,
            block_size=block_size,
            num_samples=num_samples,
            mask_token=mask_token,
            permutations=permutations,
            permutation_generator=permutation_generator,
        )
        mu_chunks.append(logL_b.detach().to("cpu"))

    mu_all = torch.cat(mu_chunks, dim=0)
    var_all = torch.zeros_like(mu_all)

    tmp = path.with_suffix(".pt.tmp")
    torch.save({"mu": mu_all, "var": var_all, "meta": key_payload}, tmp)
    tmp.replace(path)

    return ReferenceAugmentedDataLoader(dl, mu_all, var_all)
