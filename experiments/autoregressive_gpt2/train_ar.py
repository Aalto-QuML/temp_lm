import argparse
import json
import hashlib
from pathlib import Path
import sys
from typing import Iterator
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    GPT2Model,
    AutoModelForCausalLM,
)
import wandb
from tqdm import tqdm
from jaxtyping import Float

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


from utils.config_classes import OpenWebTextLoaderSpec, Batch, LazyLoaded
from evaluation.sequence_metrics import sequence_likelihood_ratios


def _stable_hash(obj, algo: str = "sha256") -> str:
    """Create stable hash for caching."""
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    h = hashlib.new(algo)
    h.update(payload)
    return h.hexdigest()


class ReferenceAugmentedARDataLoader:
    """
    Wraps a DataLoader and adds AR baseline log-likelihoods to each batch.
    Ensures deterministic batch ordering for alignment with cached logL.
    """

    def __init__(
        self,
        base: DataLoader[Batch],
        ref_logL: Float[torch.Tensor, "M"],  # (M,) - log likelihood per sequence
    ):
        self.base = base
        self.ref_logL = ref_logL

    def __len__(self) -> int:
        return len(self.base)

    def __iter__(self) -> Iterator[Batch]:
        offset = 0
        for batch in self.base:
            bsz = batch["input_ids"].shape[0]
            # Attach CPU tensor; training step can move to GPU as needed
            batch["ref_logL"] = self.ref_logL[offset : offset + bsz]
            offset += bsz
            yield batch


@torch.inference_mode()
def get_reference_augmented_ar_dataloader(
    baseline_model_id: str,
    data: LazyLoaded[DataLoader[Batch]],
    *,
    cache_dir: Path = Path(".cache/baseline_ar_logL"),
    ignore_cache: bool = False,
    device: str = "cuda",
) -> ReferenceAugmentedARDataLoader:
    """
    Returns a dataloader that yields batches with baseline AR log-likelihood.

    Computes logL for the baseline model once over the full dataset and caches it.
    Then wraps the dataloader to include ref_logL in each batch.

    Args:
        baseline_model_id: HuggingFace model ID (e.g., "gpt2")
        data: LazyLoaded DataLoader specification
        cache_dir: Directory to cache results
        ignore_cache: If True, ignore existing cache
        device: Device to use for computation

    Returns:
        ReferenceAugmentedARDataLoader wrapping the base dataloader with ref_logL attached
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache key from model and data fingerprint
    key_payload = {
        "fn": "get_reference_augmented_ar_dataloader_v1",
        "baseline_model_id": baseline_model_id,
        "data": data.fingerprint(),
    }
    key = _stable_hash(key_payload)
    cache_path = cache_dir / f"{key}.pt"

    # Try loading from cache
    if cache_path.exists() and not ignore_cache:
        print(f"Loading cached baseline AR logL from {cache_path}")
        cached_data = torch.load(cache_path, map_location="cpu")
        logL_all = cached_data["logL"]
        # Create a fresh dataloader for training to avoid iterator state issues
        base_loader = data.get()
        return ReferenceAugmentedARDataLoader(base_loader, logL_all)

    # Cache miss: compute baseline logL over full dataset
    print(f"Computing baseline AR logL (will cache to {cache_path})")
    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_model_id)
    baseline_model.to(device)
    baseline_model.eval()

    # Get dataloader for baseline computation
    dl_for_cache = data.get()
    logL_chunks = []
    batch_sizes = []

    for batch in tqdm(dl_for_cache, desc="Computing baseline AR logL"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_sizes.append(len(input_ids))

        logL = compute_sequence_log_likelihood(
            baseline_model, input_ids, attention_mask, device
        )
        logL_chunks.append(logL.detach().to("cpu"))

    logL_all = torch.cat(logL_chunks, dim=0)

    # Save to cache
    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save(
        {"logL": logL_all, "meta": key_payload},
        tmp_path,
    )
    tmp_path.replace(cache_path)  # atomic on POSIX

    print(f"Cached baseline AR logL to {cache_path}")

    # Print diagnostics
    print(
        f"Cache computation: {len(logL_chunks)} batches, batch sizes: {batch_sizes[:5]}... (total {sum(batch_sizes)} sequences)"
    )
    print(f"Total sequences cached: {len(logL_all)}")

    # Now get a FRESH dataloader for training (different instance if possible)
    # Since data.get() is cached, iterate fresh will still use same instance
    # But creating a new ReferenceAugmentedARDataLoader will iterate it fresh
    base_loader = data.get()
    return ReferenceAugmentedARDataLoader(base_loader, logL_all)


def compute_sequence_log_likelihood(model, input_ids, attention_mask, device):
    """
    Compute the log likelihood of sequences given an autoregressive model.

    Args:
        model: The autoregressive language model
        input_ids: Token IDs of shape (batch_size, seq_length)
        attention_mask: Attention mask of shape (batch_size, seq_length)
        device: Device to run on

    Returns:
        log_likelihoods: Log likelihoods for each sequence in batch, shape (batch_size,)
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

            return sequence_log_likelihoods / shifted_attention_mask.sum(
                dim=1
            )  # average logL per token


def train_autoregressive_model(
    model_id: str,
    dataloader_specification: OpenWebTextLoaderSpec,
    val_loader_specification: OpenWebTextLoaderSpec,
    model_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    temperature: float = 1.0,
    num_warmup_steps: int = 500,
    save_path: str = "ar-finetuned",
    device: str = "cuda",
    alpha: float = 1.0,
    baseline_logL_cache_dir: Path = Path(".cache/baseline_ar_logL"),
):
    """
    Train an autoregressive model (e.g., GPT-2) using next-token prediction with ratio loss.
    Combines cross-entropy loss with ratio loss from a fixed GPT2 baseline.

    Args:
        model_id: HuggingFace model ID (e.g., "gpt2")
        dataloader_specification: Specification for the training dataloader
        val_loader_specification: Specification for the validation dataloader
        model_length: Maximum sequence length
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        temperature: Temperature scaling for ratio loss
        num_warmup_steps: Number of warmup steps for scheduler
        save_path: Path to save the fine-tuned model
        device: Device to use ("cuda" or "cpu")
        alpha: Weight for CE loss; (1 - alpha) is weight for ratio loss
        baseline_logL_cache_dir: Directory to cache baseline logL for reuse

    Returns:
        model: The fine-tuned model
    """

    # Initialize W&B
    wandb_name = (
        f"ar-finetune_lr{learning_rate}_bs{batch_size}_temp{temperature}_alpha{alpha}"
    )
    wandb.init(
        project="fine-tune-ar-openwebtext",
        name=wandb_name,
        config={
            "model_id": model_id,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_length": model_length,
            "num_warmup_steps": num_warmup_steps,
            "temperature": temperature,
            "alpha": alpha,
            "save_path": save_path,
        },
    )

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model directly with AutoModelForCausalLM
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)

    # Load dataloaders with baseline logL augmentation
    print(
        f"Computing/loading cached baseline logL for {model_length} sequence length..."
    )
    train_loader = get_reference_augmented_ar_dataloader(
        baseline_model_id="gpt2",
        data=dataloader_specification.lazy(),
        cache_dir=baseline_logL_cache_dir,
        device=device,
        # ignore_cache=True,
    )

    print(
        f"Computing/loading cached validation baseline logL for {model_length} sequence length..."
    )
    val_loader = get_reference_augmented_ar_dataloader(
        baseline_model_id="gpt2",
        data=val_loader_specification.lazy(),
        cache_dir=baseline_logL_cache_dir / "val",
        device=device,
        # ignore_cache=True,
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_loader),
        num_cycles=0.25,
    )

    model.train()
    # model.eval()  # Set to eval mode to disable dropout, but we will still compute gradients for training
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training AR Model"):

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        # Get baseline logL from augmented batch
        baseline_logL = batch["ref_logL"].to(device)

        # Debug: check batch alignment
        if batch_idx < 3:
            print(
                f"Batch {batch_idx}: input_ids shape={input_ids.shape}, baseline_logL shape={baseline_logL.shape}"
            )
            print(f"  input_ids mean token: {input_ids.float().mean().item()}")
            print(f"  baseline_logL mean: {baseline_logL.float().mean().item():.4f}")

        # Shift labels for next-token prediction
        labels = input_ids.clone()
        labels = torch.where(attention_mask == 1, labels, -100)  # Ignore padding tokens

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Single forward pass - compute outputs once
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            ce_loss = outputs.loss
            logits = outputs.logits  # Reuse logits for ratio loss computation

            # Compute model logL from cached logits (no new forward pass)
            log_probs = F.log_softmax(logits, dim=-1)
            shifted_input_ids = input_ids[:, 1:].contiguous()
            shifted_log_probs = log_probs[:, :-1, :]

            batch_size_actual, seq_length = shifted_input_ids.shape
            batch_indices = torch.arange(batch_size_actual, device=device).unsqueeze(1)
            seq_indices = torch.arange(seq_length, device=device).unsqueeze(0)
            token_log_probs = shifted_log_probs[
                batch_indices, seq_indices, shifted_input_ids
            ]

            shifted_attention_mask = attention_mask[:, 1:]
            token_log_probs = token_log_probs * shifted_attention_mask
            model_logL = token_log_probs.sum(dim=1) / shifted_attention_mask.sum(
                dim=1
            )  # shape: (batch_size,)

            # assert torch.allclose(
            #     model_logL,
            #     baseline_logL,
            # ), f"Model logL should be close to baseline at initialization, but got model_logL={model_logL.mean().item():.4f} vs baseline_logL={baseline_logL.mean().item():.4f}"

            # Compute likelihood ratios
            model_ratio = sequence_likelihood_ratios(model_logL, "full")
            baseline_ratio = sequence_likelihood_ratios(baseline_logL, "full")

            # assert torch.allclose(
            #     model_ratio, baseline_ratio, rtol=0.1
            # ), f"Ratios should be close at initialization since model is same as baseline, but the max ratio difference is {(model_ratio - baseline_ratio).abs().max().item():.4f}"

            # MSE loss on ratios
            ratio_loss = F.mse_loss(
                temperature * model_ratio,
                baseline_ratio.to(model_ratio.device),
            )

            # Combined loss: alpha * CE + ratio
            loss = alpha * ce_loss + ratio_loss

            loss.backward()
            # torch.nn.utils.clip_gsrad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Compute training metrics (same format as discrete diffusion)
            train_ratio = (
                (
                    ((baseline_ratio.to(model_ratio.device)) + 0.1)
                    / ((model_ratio) + 0.1)
                )
                .mean()
                .item()
            )
            train_avg_nll = (-model_logL.sum() / len(model_logL)).item()

        wandb.log(
            {
                "train_loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "ratio_loss": ratio_loss.item(),
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "ratio": train_ratio,
                "avg_nll": train_avg_nll,
            }
        )

        # Validation every 1000 steps
        if batch_idx * batch_size % 2048 == 0 and batch_idx > 0:
            model.eval()
            val_model_logLs = []
            val_baseline_logLs = []
            val_input_ids_all = []

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_loader):
                    val_input_ids = val_batch["input_ids"].to(device, non_blocking=True)
                    val_attention_mask = val_batch["attention_mask"].to(
                        device, non_blocking=True
                    )
                    val_labels = val_input_ids.clone()
                    val_labels = torch.where(val_attention_mask == 1, val_labels, -100)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        val_outputs = model(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask,
                            labels=val_labels,
                        )

                        # Compute model logL from logits
                        val_logits = val_outputs.logits
                        val_log_probs = F.log_softmax(val_logits, dim=-1)
                        val_shifted_input_ids = val_input_ids[:, 1:].contiguous()
                        val_shifted_log_probs = val_log_probs[:, :-1, :]

                        val_batch_size, val_seq_length = val_shifted_input_ids.shape
                        val_batch_indices = torch.arange(
                            val_batch_size, device=device
                        ).unsqueeze(1)
                        val_seq_indices = torch.arange(
                            val_seq_length, device=device
                        ).unsqueeze(0)
                        val_token_log_probs = val_shifted_log_probs[
                            val_batch_indices, val_seq_indices, val_shifted_input_ids
                        ]

                        val_shifted_attention_mask = val_attention_mask[:, 1:]
                        val_token_log_probs = (
                            val_token_log_probs * val_shifted_attention_mask
                        )
                        val_model_logL = val_token_log_probs.sum(
                            dim=1
                        ) / val_shifted_attention_mask.sum(
                            dim=1
                        )  # shape: (batch_size,)
                        val_model_logLs.append(val_model_logL.detach())

                        # Get precomputed baseline logL from augmented batch
                        val_baseline_logL = val_batch["ref_logL"].to(device)
                        val_baseline_logLs.append(val_baseline_logL.detach())

                        # DEBUG: Check alignment at first validation batch
                        # if val_batch_idx == 0:
                        #     print(f"\n=== VALIDATION DEBUG (Step {batch_idx}) ===")
                        #     print(
                        #         f"Model logL shape: {val_model_logL.shape}, Baseline logL shape: {val_baseline_logL.shape}"
                        #     )
                        #     print(
                        #         f"Model logL mean: {val_model_logL.float().mean().item():.6f} (std: {val_model_logL.float().std().item():.6f})"
                        #     )
                        #     print(
                        #         f"Baseline logL mean: {val_baseline_logL.float().mean().item():.6f} (std: {val_baseline_logL.float().std().item():.6f})"
                        #     )
                        #     print(
                        #         f"Difference (Model - Baseline) mean: {(val_model_logL - val_baseline_logL).float().mean().item():.6f}"
                        #     )
                        #     print(
                        #         f"Max absolute diff: {(val_model_logL - val_baseline_logL).abs().float().max().item():.6f}"
                        #     )
                        #     print(f"First 5 model logL: {val_model_logL[:5]}")
                        #     print(f"First 5 baseline logL: {val_baseline_logL[:5]}")
                        #     val_input_ids_all.append(val_input_ids.detach())

            val_model_logL_all = torch.cat(val_model_logLs)
            val_baseline_logL_all = torch.cat(val_baseline_logLs)

            # Compute likelihood ratios (same as in training)
            val_model_ratio = sequence_likelihood_ratios(val_model_logL_all, "full")
            val_baseline_ratio = sequence_likelihood_ratios(
                val_baseline_logL_all, "full"
            )

            # DEBUG: Check ratio values
            # print(
            #     f"Model ratio mean: {val_model_ratio.float().mean().item():.8f} (std: {val_model_ratio.float().std().item():.8f})"
            # )
            # print(
            #     f"Baseline ratio mean: {val_baseline_ratio.float().mean().item():.8f} (std: {val_baseline_ratio.float().std().item():.8f})"
            # )
            # print(
            #     f"Model ratio shape: {val_model_ratio.shape}, Baseline ratio shape: {val_baseline_ratio.shape}"
            # )

            # Ratio metric: average of (baseline + eps) / (model + eps) for numerical stability
            val_ratio = ((val_baseline_ratio.to(val_model_ratio.device))) / (
                (val_model_ratio)
            )
            # val_ratio = val_ratio.mean().item()
            print(
                f"Val ratio distribution - median: {val_ratio.median().item():.6f}, mean: {val_ratio.mean().item():.6f}, min: {val_ratio.min().item():.6f}, max: {val_ratio.max().item():.6f}, std: {val_ratio.std().item():.6f}"
            )
            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(10, 6))
            # plt.boxplot(val_ratio.cpu().numpy())
            # plt.ylabel("Validation Ratio")
            # plt.title(f"Validation Ratio Distribution (Step {batch_idx})")
            # plt.grid(True, alpha=0.3, axis="y")
            # plt.show()
            # NLL metric: average negative log likelihood
            val_avg_nll = (-val_model_logL_all.sum() / len(val_model_logL_all)).item()

            wandb.log(
                {
                    "val_ratio": val_ratio.median().item(),
                    "val_avg_nll": val_avg_nll,
                }
            )
            print(
                f"Validation at step {batch_idx}: ratio={val_ratio.mean().item():.4f}, nll={val_avg_nll:.4f}"
            )
            path = f"{save_path}/checkpoint_{batch_idx * batch_size}"
            print(f"\nSaving checkpoint to {path}")
            model.save_pretrained(path)
            model.train()

    # Save the fine-tuned model
    print(f"\nSaving model to {save_path}/model")
    model.save_pretrained(f"{save_path}/model")

    wandb.finish()
    print("\nTraining complete!")

    return model


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train autoregressive transformer model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=300,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--seq_length", type=int, default=128, help="Model sequence length"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./models",
        help="Path to save the fine-tuned model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of samples from dataset"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for ratio loss",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Weight for CE loss",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    model_id = "gpt2"  # or any other autoregressive model

    dataloader_specification = OpenWebTextLoaderSpec(
        max_length=args.seq_length,
        slice_end=args.num_samples,
        batch_size=args.batch_size,
    )

    val_loader_specification = OpenWebTextLoaderSpec(
        max_length=args.seq_length,
        slice_start=args.num_samples,
        slice_end=args.num_samples + 500,
        batch_size=args.batch_size,
    )

    save_path = f"{args.save_path}/ar-finetuned-{model_id.replace('/', '-')}-alpha{args.alpha}-seed{args.seed}-temp{args.temperature}-lr{args.learning_rate}_bs{args.batch_size}_len{args.seq_length}"

    # Train the model
    model = train_autoregressive_model(
        model_id,
        dataloader_specification,
        val_loader_specification,
        model_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        num_warmup_steps=args.num_warmup_steps,
        save_path=save_path,
        alpha=args.alpha,
    )
