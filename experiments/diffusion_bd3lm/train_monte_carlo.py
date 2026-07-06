import argparse
import os
import sys
import warnings
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

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
    category=FutureWarning,
)

import torch
import torch._dynamo

torch._dynamo.config.disable = True  # global switch
import wandb
from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup
from utils.cached_reference_dataloader import get_reference_augmented_dataloader_mc
from utils.config_classes import ModelSpec, OpenWebTextLoaderSpec
from evaluation.sequence_metrics import (
    log_expected_likelihood_and_elbo_mc,
    sample_permutations,
    sequence_likelihood_ratios,
)
from plotting.evaluate_perplexity_temp_scaling import (
    regress_effective_temperature_scaling,
    kendall_tau,
)
from experiments.diffusion_bd3lm.val_mc_metrics import run_mc_validation

from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path


os.environ["TOKENIZERS_PARALLELISM"] = "false"
MASK_TOKEN_ID = 50257


def train_bd3lm_model(
    model_specification: ModelSpec,
    baseline_specification: ModelSpec,
    dataloader_specification: OpenWebTextLoaderSpec,
    val_loader_spepecification: OpenWebTextLoaderSpec,
    block_length: int = 4,
    model_length: int = 128,
    num_sequences: int = 4,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    temperature: float = 0.5,
    num_warmup_steps: int = 500,
    save_path: str = "bd3lm-finetuned",
    device: torch.device = "cuda",
    alpha_ao: float = 0.5,
    alpha_ratio: float = 0.5,
    mc_samples: int = 8,
):
    """
    Train a BD3LM model with the monte carlo version of the loss.

    Args:
        model_name: Pretrained model name/path
        tokenizer: Tokenizer for the model
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        block_length: Length of blocks for masking
        model_length: Maximum sequence length
        batch_size: Batch size for processing masked sequences
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        temperature: Temperature parameter for loss calculation
        num_warmup_steps: Number of warmup steps for scheduler
        use_baseline: Whether to use baseline correction
        update_baseline_train: Whether to update baseline during training
        update_baseline_val: Whether to update baseline during validation
        reset_baseline_per_epoch: Whether to reset baseline at start of each epoch
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        save_path: Path to save the fine-tuned model
        device: Device to use (defaults to cuda if available)

    Returns:
        model: The fine-tuned model
    """

    # Initialize W&B
    wandb_name = f"bd3lm-finetune_MC_temp{args.temperature}_K{args.mc_samples}_ratio{args.alpha_ratio}_ao{args.alpha_ao}"
    wandb.init(
        project="fine-tune-bd3lm-openwebtext",
        name=wandb_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "block_length": block_length,
            "model_length": model_length,
            "num_sequences": num_sequences,
            "temperature": temperature,
            "num_warmup_steps": num_warmup_steps,
            "K": mc_samples,
            "delta_ao": alpha_ao,
            "alpha_ratio": alpha_ratio,
            "mc_samples": mc_samples,
            "save_path": save_path,
        },
    )

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load models
    print(f"Loading model: {model_name}")
    model = model_specification.lazy().get()
    model.to(device)



    baseline_loader = get_reference_augmented_dataloader_mc(
        baseline_specification.lazy(),
        dataloader_specification.lazy(),
        block_size=block_length,
        num_samples=mc_samples,
        permutation_key=f"mc-k{mc_samples}-block{block_length}",
        # cache_dir=Path("/m/cs/work/scheufh1/.cache/bd3lm_eval"),
    )

    val_loader = get_reference_augmented_dataloader_mc(
        baseline_specification.lazy(),
        val_loader_spepecification.lazy(),
        block_size=block_length,
        num_samples=mc_samples,
        permutation_key=f"mc-k{mc_samples}-block{block_length}-val",
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(baseline_loader),
        num_cycles=0.25,
    )

    num_subsets = 2 ** (block_length)
    num_blocks = model_length // block_length

    baseline = {
        "count": 0,
        "logL_sum": torch.zeros(num_subsets, num_blocks).to(device),
        "var_sum": torch.zeros(num_subsets, num_blocks).to(device),
    }

    model.train()
    for batch_idx, batch in tqdm(enumerate(baseline_loader), desc="Training"):

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = (
            batch["attention_mask"].to(torch.bool).to(device, non_blocking=True)
        )

        baseline_logL = batch["ref_mu"].detach().to(device)


        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ), torch.amp.autocast("cuda", dtype=torch.bfloat16):

            permutations = sample_permutations(
                mc_samples, input_ids, block_size=block_length
            )

            logL, elbo_mc = log_expected_likelihood_and_elbo_mc(
                model,
                input_ids,
                attention_mask,
                block_size=block_length,
                num_samples=mc_samples,
                mask_token=MASK_TOKEN_ID,
                permutations=permutations,
            )

            reference_ratio = sequence_likelihood_ratios(
                baseline_logL, "full"
            )
            model_ratio = sequence_likelihood_ratios(logL, "full")

            effective_temperature = regress_effective_temperature_scaling(
                baseline_logL,
                logL,
                pairing_mode="full",
            )
            tau = kendall_tau(
                baseline_logL,
                logL,
                pairing_mode="full",
            )


            if alpha_ratio > 0:
                ratio_loss = F.mse_loss(
                    temperature * model_ratio, reference_ratio.to(model_ratio.device)
                )
            else:
                ratio_loss = torch.tensor(0.0).to(device)

            if alpha_ao > 0:
                anyorder_ar_loss = -elbo_mc.mean() / num_sequences
            else:
                anyorder_ar_loss = torch.tensor(0.0).to(device)

            loss = (
                alpha_ratio * ratio_loss
                + alpha_ao * anyorder_ar_loss
            )

            # print("loss done")
            loss.backward()
            # print("backward done")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # print("opt step done")
            lr_scheduler.step()

            optimizer.zero_grad()
            # print("grad zeroed")
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "ratio": (
                        ((reference_ratio.to(model_ratio.device)) + 0.1)
                        / ((model_ratio) + 0.1)
                    ).mean(),
                    "avg_nll": -logL.sum() / len(logL),
                    "ratio_loss": ratio_loss.item(),
                    "ar_loss": anyorder_ar_loss.item(),
                    "effective_temperature": effective_temperature,
                    "kendall_tau": tau,
                }
            )

        if (batch_idx * batch_size) % 2048 == 0 and batch_idx > 0:
            val_metrics = run_mc_validation(
                model,
                val_loader,
                block_size=block_length,
                mc_samples=mc_samples,
                mask_token=MASK_TOKEN_ID,
                device=device,
            )
            wandb.log(val_metrics)
            path = f"{save_path}/checkpoint_{batch_idx * batch_size}"
            print(f"\nSaving checkpoint to {path}")
            model.save_pretrained(path)
            model.train()


    # Save the fine-tuned model
    print(f"\nSaving model to {save_path}/model")
    model.save_pretrained(f"{save_path}/model")

    wandb.finish()
    print("\nTraining complete!")

    return model  # , calculator


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--block_length", type=int, default=4, help="Block length for BD3LM"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for loss calculation",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--seq_length", type=int, default=64, help="Model sequence length"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/m/cs/scratch/temperature_diffusion/models2",
        help="Path to save the fine-tuned model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of samples from dataset"
    )
    parser.add_argument(
        "--alpha_ao", type=float, default=0.1, help="Alpha value for AO loss"
    )
    parser.add_argument(
        "--alpha_ratio", type=float, default=0.0, help="Alpha value for Ratio loss"
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=8,
        help="Number of Monte Carlo permutations for logL",
    )
    # parser.add_argument(
    #     "num_pairs", type=int, default=32, help="Number of pairs for ratio loss"
    # )

    args = parser.parse_args()
    block_length = args.block_length
    batch_size = args.batch_size
    num_sequences = 2 * batch_size // (2**block_length)
    if num_sequences < 2:
        num_sequences = 2
    # num_sequences = batch_size // 2  # Override for now
    # Set random seed
    torch.manual_seed(args.seed)

    sequence_length = args.seq_length
    block_length = args.block_length
    assert block_length in [4, 8, 16], "Block length must be one of [4, 8, 16]"
    model_name = f"kuleshov-group/bd3lm-owt-block_size{block_length}"
    model_specification = baseline_model_specification = ModelSpec(
        model_name, sequence_length, device="cuda"
    )

    dataloader_specification = OpenWebTextLoaderSpec(
        max_length=sequence_length, slice_end=args.num_samples, batch_size=batch_size
    )

    valid_loader_specification = OpenWebTextLoaderSpec(
        max_length=sequence_length,
        slice_start=args.num_samples,
        slice_end=args.num_samples + 500,
        batch_size=16 * batch_size,
    )

    ft_model_name = f"bd3lm-finetuned-openwebtext-MC-blocksize{block_length}_emp{args.temperature}_K{args.mc_samples}_ratio{args.alpha_ratio}_ao{args.alpha_ao}_seed{args.seed}_lr{args.learning_rate}_bs{args.batch_size}_len{args.seq_length}_ws{args.num_warmup_steps}_ns{args.num_samples}"
    save_path = f"{args.save_path}/{ft_model_name}"

    # Train the model
    model = train_bd3lm_model(
        model_specification,
        baseline_model_specification,
        dataloader_specification,
        valid_loader_specification,
        block_length=block_length,
        model_length=sequence_length,
        num_sequences=num_sequences,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        num_warmup_steps=args.num_warmup_steps,
        save_path=save_path,
        alpha_ao=args.alpha_ao,
        alpha_ratio=args.alpha_ratio,
        mc_samples=args.mc_samples,
    )
