import argparse
import json
import hashlib
import shutil
from pathlib import Path
from typing import Iterator, Optional
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
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


def _model_slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


from evaluation.sequence_metrics import sequence_likelihood_ratios
from plotting.evaluate_perplexity_temp_scaling import (
    regress_effective_temperature_scaling,
)
from experiments.reasoning.reasoning_utils import (
    make_math_dataloader,
    get_reference_augmented_ar_dataloader,
    compute_sequence_log_likelihood,
)

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_reasoning_math_dataset(
    model_id: str = "Qwen/Qwen2.5-7B",
    dataset_name: str = "gsm8k",
    max_length: int = 512,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    temperature: float = 1.0,
    use_ratio_loss: bool = True,
    num_warmup_steps: int = 100,
    save_path: str = "phi35-math-finetuned",
    device: str = "cuda",
    alpha: float = 1.0,
    baseline_logL_cache_dir: Path = Path(".cache/baseline_ar_logL"),
    val_fraction: float = 0.1,
    split_seed: int = 42,
):
    model_slug = _model_slug(model_id)
    ratio_mode_slug = "ratio_loss" if use_ratio_loss else "no_ratio_loss"

    # --- W&B ---
    wandb.init(
        project=f"{model_slug}-{dataset_name}-instruct",
        name=f"{model_slug}_{dataset_name}_{ratio_mode_slug}_lr{learning_rate}_bs{batch_size}_alpha{alpha}_temp{temperature}",
        config=dict(
            model_id=model_id,
            model_slug=model_slug,
            dataset_name=dataset_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            temperature=temperature,
            use_ratio_loss=use_ratio_loss,
            alpha=alpha,
            val_fraction=val_fraction,
            split_seed=split_seed,
        ),
    )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    # shuffle=False is critical for the cached baseline logL to stay aligned
    train_dl = make_math_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="train",
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    val_dl = make_math_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="test",  # for MATH, this is our held-out val split
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    # --- Baseline logL (frozen Phi-3.5, answer tokens only) ---
    train_loader = get_reference_augmented_ar_dataloader(
        baseline_model_id=model_id,
        tokenizer=tokenizer,
        dataloader=train_dl,
        dataset_name=dataset_name,
        split_name="train",
        cache_dir=baseline_logL_cache_dir,
        device=device,
    )
    val_loader = get_reference_augmented_ar_dataloader(
        baseline_model_id=model_id,
        tokenizer=tokenizer,
        dataloader=val_dl,
        dataset_name=dataset_name,
        split_name="test",
        cache_dir=baseline_logL_cache_dir / "val",
        device=device,
    )

    # --- Model (trainable) ---
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- Optimiser & scheduler ---
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_loader),
        num_cycles=0.25,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        answer_mask = batch["answer_mask"].to(device)
        baseline_logL = batch["ref_logL"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):

            # Forward pass — labels already have prompt masked to -100
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            ce_loss = outputs.loss
            logits = outputs.logits

            # ---- Ratio loss — restricted to answer tokens ----
            log_probs = F.log_softmax(logits, dim=-1)
            shifted_ids = input_ids[:, 1:].contiguous()
            shifted_log_probs = log_probs[:, :-1, :]

            B, T = shifted_ids.shape
            b_idx = torch.arange(B, device=device).unsqueeze(1)
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            token_log_probs = shifted_log_probs[b_idx, t_idx, shifted_ids]

            ans_mask_shifted = answer_mask[:, 1:].float()
            token_log_probs = token_log_probs * ans_mask_shifted

            denom = ans_mask_shifted.sum(dim=1).clamp(min=1.0)
            model_logL = token_log_probs.sum(dim=1) / denom  # (B,)

            # Likelihood ratios are still logged, but they only affect training when enabled.
            model_ratio = sequence_likelihood_ratios(model_logL, "full")
            baseline_ratio = sequence_likelihood_ratios(baseline_logL, "full")
            ratio_loss = F.mse_loss(
                temperature * model_ratio,
                baseline_ratio.to(model_ratio.device),
            )

            loss = alpha * ce_loss + (ratio_loss if use_ratio_loss else 0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_ratio = (
            ((baseline_ratio.to(model_ratio.device) + 0.1) / (model_ratio + 0.1))
            .mean()
            .item()
        )
        train_avg_nll = (-model_logL.sum() / len(model_logL)).item()

        wandb.log(
            {
                "train_loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "ratio_loss": ratio_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "ratio": train_ratio,
                "avg_nll": train_avg_nll,
            }
        )

        # ---- Validation ----
        if batch_idx % 100 == 0:
            model.eval()
            val_model_logLs, val_baseline_logLs = [], []

            with torch.no_grad():
                for val_batch in val_loader:
                    v_ids = val_batch["input_ids"].to(device)
                    v_amask = val_batch["attention_mask"].to(device)
                    v_ansmask = val_batch["answer_mask"].to(device)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        val_logL = compute_sequence_log_likelihood(
                            model, v_ids, v_amask, device, answer_mask=v_ansmask
                        )
                    val_model_logLs.append(val_logL.detach())
                    val_baseline_logLs.append(val_batch["ref_logL"].to(device).detach())

            val_model_logL_all = torch.cat(val_model_logLs)
            val_baseline_logL_all = torch.cat(val_baseline_logLs)

            val_effective_temp = regress_effective_temperature_scaling(
                val_baseline_logL_all,
                val_model_logL_all,
                pairing_mode="full",
            )

            val_avg_nll = (-val_model_logL_all.sum() / len(val_model_logL_all)).item()

            wandb.log(
                {
                    "val_effective_temperature": float(val_effective_temp),
                    "val_avg_nll": val_avg_nll,
                }
            )
            print(
                f"Step {batch_idx} | val_effective_temp={float(val_effective_temp):.4f} | val_nll={val_avg_nll:.4f}"
            )

            checkpoint_path = Path(save_path) / f"checkpoint_{batch_idx}.pt"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            model.train()

    # -----------------------------------------------------------------------
    # Save model + Phi-specific config files
    # -----------------------------------------------------------------------
    save_path_obj = Path(save_path) / "model"
    save_path_obj.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_path_obj))
    tokenizer.save_pretrained(str(save_path_obj))

    # Copy Phi-specific modeling files to ensure fine-tuned model can be loaded
    # with trust_remote_code=True in the future
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Save the config (includes model architecture info)
    baseline_model.config.save_pretrained(str(save_path_obj))

    # Try to copy the custom modeling file if it exists in the cached model
    try:
        from huggingface_hub import cached_repo_path_and_type

        cache_info = cached_repo_path_and_type(model_id)
        if cache_info[0]:
            source_modeling = Path(cache_info[0]) / "modeling_phi3.py"
            if source_modeling.exists():
                dest_modeling = save_path_obj / "modeling_phi3.py"
                shutil.copy(str(source_modeling), str(dest_modeling))
                print(f"Copied modeling_phi3.py to {dest_modeling}")
    except Exception as e:
        print(
            f"Could not copy modeling file from cache ({e}), it will be downloaded at load time"
        )

    del baseline_model
    torch.cuda.empty_cache()

    wandb.finish()
    print("Training complete!")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Hugging Face model id to train, for example microsoft/Phi-3.5-mini-instruct or Qwen/Qwen2.5-7B.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="math", choices=["gsm8k", "math"]
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--save_path", type=str, default="/m/cs/scratch/temperature_diffusion/"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    ratio_loss_group = parser.add_mutually_exclusive_group()
    ratio_loss_group.add_argument(
        "--use_ratio_loss",
        dest="use_ratio_loss",
        action="store_true",
    )
    ratio_loss_group.add_argument(
        "--no_ratio_loss",
        dest="use_ratio_loss",
        action="store_false",
    )
    parser.set_defaults(use_ratio_loss=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_slug = _model_slug(args.model_id)
    save_path = (
        f"{args.save_path}/{model_slug}-{args.dataset_name}-temperature{args.temperature}"
        f"-alpha{args.alpha}-seed{args.seed}"
        f"-{'ratio_loss' if args.use_ratio_loss else 'no_ratio_loss'}"
    )

    train_reasoning_math_dataset(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        use_ratio_loss=args.use_ratio_loss,
        num_warmup_steps=args.num_warmup_steps,
        save_path=save_path,
        alpha=args.alpha,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
