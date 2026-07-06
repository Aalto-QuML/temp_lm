from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import (
    Callable,
    Generic,
    Optional,
    Literal,
    Any,
    Dict,
    Tuple,
    TypeVar,
    TypedDict,
)
from jaxtyping import Int64, Bool, Float
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM

from utils.block_diffusion_classes import MyopicTemperatureScalingWrapper


from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluation.sequence_metrics import (
    subset_permutation_means_variances,
    log_expected_likelihood_mc,
    all_transition_likelihoods_batched,
)


T = TypeVar("T")


class Batch(TypedDict):
    input_ids: Int64[torch.Tensor, "B N"]
    attention_mask: Bool[torch.Tensor, "B N"]


@dataclass
class LazyLoaded(Generic[T]):
    key: str
    _loader: Callable[[], T]
    _value: Optional[T] = None

    def get(self) -> T:
        if self._value is None:
            self._value = self._loader()
        return self._value

    def fingerprint(self) -> str:
        return self.key

    def __getattr__(self, name: str) -> Any:
        # Optional: lets you use it like the loaded object (loads on first attribute access)
        return getattr(self.get(), name)

    def __iter__(self):
        return iter(self.get())

    def __len__(self):
        return len(self.get())

    def __getitem__(self, idx):
        return self.get()[idx]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    A minimal, cache-friendly model config.

    - `model_id`: HF repo id OR local path
    - `revision`: optional HF commit/tag for reproducibility
    - `max_length`: sets conf.model_length (for bd3lm)
    - wrapper options that change outputs (e.g. temperature) belong here, so they affect hashing.
    """

    model_id: str
    max_length: int

    revision: Optional[str] = None
    trust_remote_code: bool = True

    # runtime-ish but output-affecting choices (so they belong in fingerprint)
    myopic_temperature: Optional[float] = (
        None  # if set, wrap with MyopicTemperatureScalingWrapper
    )

    # not part of model identity, but convenient defaults
    device: Optional[str] = None  # e.g. "cuda"
    dtype: Optional[Literal["fp32", "fp16", "bf16"]] = (
        None  # for autocast policy outside
    )

    def to_key_dict(self) -> Dict[str, Any]:
        # Only include fields that should distinguish cached results.
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "max_length": self.max_length,
            "temperature": f"{self.myopic_temperature if self.myopic_temperature is not None else 1:.6f}",
        }

    def fingerprint(self, algo: str = "sha256") -> str:
        payload = json.dumps(
            self.to_key_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        h = hashlib.new(algo)
        h.update(payload)
        return h.hexdigest()

    def load(self) -> nn.Module:
        conf = AutoConfig.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        # bd3lm uses conf.model_length; harmless for others
        conf.model_length = int(self.max_length)

        # Choose appropriate model loader based on architecture
        # Causal models (GPT2, etc.) use AutoModelForCausalLM
        # Masked models (BERT, RoBERTa, etc.) use AutoModelForMaskedLM
        if conf.model_type in ["gpt2", "gpt2-xl", "gptj", "gpt_neox", "llama", "mistral"]:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                config=conf,
            )
        else:
            # Default to masked LM for models like BERT, RoBERTa, etc.
            model = AutoModelForMaskedLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                config=conf,
            )

        if self.myopic_temperature is not None:
            # assumes you defined this wrapper already
            model = MyopicTemperatureScalingWrapper(
                model, temperature=float(self.myopic_temperature)
            )

        if self.device is not None:
            model = model.to(self.device)

        model.eval()
        return model

    def lazy(self) -> LazyLoaded[nn.Module]:
        return LazyLoaded(key=self.fingerprint(), _loader=self.load)


@dataclass(frozen=True, slots=True)
class OpenWebTextLoaderSpec:
    # What to load
    split: Literal["train"] = "train"

    # Tokenization / batching
    tokenizer_id: str = "gpt2"
    max_length: int = 64
    batch_size: int = 1024
    shuffle: bool = False
    # Deterministic slicing: [start, end)
    # (use None for "from start" / "to end")
    slice_start: Optional[int] = None
    slice_end: Optional[int] = None

    # optional convenience knobs (don’t usually affect results)
    num_workers: int = 0
    pin_memory: bool = True

    def to_key_dict(self) -> Dict[str, Any]:
        # Only things that should distinguish cached results.
        return {
            "dataset": "Skylion007/openwebtext",
            "split": self.split,
            "tokenizer_id": self.tokenizer_id,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "slice_start": self.slice_start,
            "slice_end": self.slice_end,
        }

    def fingerprint(self, algo: str = "sha256") -> str:
        payload = json.dumps(
            self.to_key_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        h = hashlib.new(algo)
        h.update(payload)
        return h.hexdigest()

    def load(self) -> DataLoader[Batch]:
        """
        Returns: dataloader
        """
        ds = load_dataset(
            "Skylion007/openwebtext",
            split=self.split,  # , download_mode="force_redownload"
        )

        if self.slice_start is not None or self.slice_end is not None:
            start = self.slice_start or 0
            end = self.slice_end if self.slice_end is not None else len(ds)
            ds = ds.select(range(start, end))

        tok = AutoTokenizer.from_pretrained(self.tokenizer_id)  # type: ignore
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if tok.mask_token is None:
            tok.add_special_tokens({"mask_token": "[MASK]"})

        def tokenize_fn(batch):  # type: ignore
            return tok(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )  # type: ignore

        ds = ds.map(  # type: ignore
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing openwebtext[{self.split}]",  # type: ignore
        )

        ds = ds.with_format(type="torch", columns=["input_ids", "attention_mask"])  # type: ignore

        dl = DataLoader(  # type: ignore
            ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dl  # , tok  # type: ignore

    def lazy(self) -> LazyLoaded[DataLoader[Batch]]:
        return LazyLoaded(key=self.fingerprint(), _loader=self.load)


def _stable_hash(obj: Any, algo: str = "sha256") -> str:
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    h = hashlib.new(algo)
    h.update(payload)
    return h.hexdigest()


@torch.inference_mode()
def get_full_log_exp_likelihood_cached(
    model: LazyLoaded[torch.nn.Module],  # LazyLoaded[torch.nn.Module]
    data: LazyLoaded[DataLoader[Batch]],  # LazyLoaded[DataLoader[Batch]]
    block_size: int = 4,
    mask_token: int = 50257,
    cache_dir: Path = Path(".cache/bd3lm_eval"),
    ignore_cache: bool = False,
    cache_key_extra: Optional[dict] = None,
) -> Tuple[Float[torch.Tensor, "M"], Float[torch.Tensor, "M"]]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    extra: Dict[str, Any] = cache_key_extra or {}
    cache_payload = {
        "fn": "get_full_log_exp_likelihood_cached_v1",
        "model": model.fingerprint(),
        "data": data.fingerprint(),
        "extra": extra,
    }
    # print("cache_payload", cache_payload)
    cache_key = _stable_hash(cache_payload)
    cache_path = cache_dir / f"{cache_key}.pt"
    if cache_path.exists() and not ignore_cache:
        obj = torch.load(cache_path, map_location="cpu")
        return obj["logL"], obj["var"]
    print(cache_path)
    print(model.fingerprint(), data.fingerprint())
    mdl = model.get()
    dl: DataLoader[Batch] = data.get()

    # # Allow passing eval knobs through cache_key_extra, but keep this minimal.
    # block_size = extra.get("block_size", 4)
    # mask_token = extra.get("mask_token", None)

    device = next(mdl.parameters()).device
    chunks = []
    chunks_var = []

    for batch in tqdm(dl):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True, dtype=torch.bool)

        transition_probs = all_transition_likelihoods_batched(
            mdl,
            input_ids,
            attention_mask,
            block_size=block_size,
            mask_token=mask_token if mask_token is not None else 50257,
        )
        logL_b, var_b = subset_permutation_means_variances(
            transition_probs, block_size=block_size
        )

        chunks.append(logL_b.detach().to("cpu"))
        chunks_var.append(var_b.detach().to("cpu"))

    logL = torch.cat(chunks, dim=0)  # (M,)
    var = torch.cat(chunks_var, dim=0)  # (M,)

    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save({"logL": logL, "var": var, "meta": cache_payload}, tmp_path)
    tmp_path.replace(cache_path)  # atomic on POSIX

    return logL, var


@torch.inference_mode()
def get_log_exp_likelihood_mc_cached(
    model: LazyLoaded[torch.nn.Module],
    data: LazyLoaded[DataLoader[Batch]],
    block_size: int = 4,
    mask_token: int = 50257,
    num_samples: int = 8,
    permutations: Optional[Int64[torch.Tensor, "K B N"]] = None,
    permutation_key: Optional[str] = None,
    cache_dir: Path = Path(".cache/bd3lm_eval_mc"),
    ignore_cache: bool = False,
    cache_key_extra: Optional[dict] = None,
) -> Tuple[Float[torch.Tensor, "M"], Float[torch.Tensor, "M"]]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    extra: Dict[str, Any] = cache_key_extra or {}
    cache_payload = {
        "fn": "get_log_exp_likelihood_mc_cached_v1",
        "model": model.fingerprint(),
        "data": data.fingerprint(),
        "block_size": int(block_size),
        "mask_token": int(mask_token),
        "num_samples": int(num_samples),
        "permutation_key": permutation_key,
        "extra": extra,
    }
    cache_key = _stable_hash(cache_payload)
    cache_path = cache_dir / f"{cache_key}.pt"
    if cache_path.exists() and not ignore_cache:
        obj = torch.load(cache_path, map_location="cpu")
        return obj["logL"], obj["var"]

    mdl = model.get()
    dl: DataLoader[Batch] = data.get()
    device = next(mdl.parameters()).device
    permutation_generator = None
    if permutations is None and permutation_key is not None:
        seed = int(hashlib.sha256(permutation_key.encode()).hexdigest()[:8], 16)
        permutation_generator = torch.Generator(device=device)
        permutation_generator.manual_seed(seed)

    chunks = []
    for batch in tqdm(dl):
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
        chunks.append(logL_b.detach().to("cpu"))

    logL = torch.cat(chunks, dim=0)
    var = torch.zeros_like(logL)

    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save({"logL": logL, "var": var, "meta": cache_payload}, tmp_path)
    tmp_path.replace(cache_path)

    return logL, var
