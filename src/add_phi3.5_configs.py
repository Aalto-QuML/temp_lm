"""
Utility script to add missing Phi-3.5 configuration and modeling files
to existing fine-tuned models that were saved without them.

Usage:
    python fix_phi_model.py --model_path /path/to/finetuned/model
    python fix_phi_model.py --model_path /path/to/finetuned/model --base_model "microsoft/Phi-3.5-mini-instruct"
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def fix_phi_model(
    model_path: str,
    base_model: str = "microsoft/Phi-3.5-mini-instruct",
):
    """
    Adds missing Phi-3.5 configuration and modeling files to a fine-tuned model.

    Args:
        model_path: Path to the fine-tuned model directory
        base_model: The base Phi model to copy files from (default: Phi-3.5-mini-instruct)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # if (
    #     # not (model_path / "pytorch_model.bin").exists()
    #     not (model_path / "model.safetensors").exists()
    # ):
    #     raise FileNotFoundError(f"No model weights found in {model_path}")

    print(f"Loading base model config from {base_model}...")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

    print(f"Saving config to {model_path}...")
    config.save_pretrained(str(model_path))

    # Try to copy modeling_phi3.py from cache
    try:
        from huggingface_hub import cached_repo_path_and_type

        cache_info = cached_repo_path_and_type(base_model)
        if cache_info[0]:  # if cached locally
            source_dir = Path(cache_info[0])

            # Copy modeling_phi3.py
            source_modeling = source_dir / "modeling_phi3.py"
            if source_modeling.exists():
                dest_modeling = model_path / "modeling_phi3.py"
                shutil.copy(str(source_modeling), str(dest_modeling))
                print(f"✓ Copied modeling_phi3.py")

            # Also try to copy configuration_phi3.py (often not necessary if in config, but doesn't hurt)
            source_config = source_dir / "configuration_phi3.py"
            if source_config.exists():
                dest_config = model_path / "configuration_phi3.py"
                shutil.copy(str(source_config), str(dest_config))
                print(f"✓ Copied configuration_phi3.py")
    except Exception as e:
        print(f"⚠ Could not copy from local cache: {e}")
        print(
            "  The files will be downloaded automatically when loading the model with trust_remote_code=True"
        )

    print(f"\n✓ Model at {model_path} is now ready to use!")
    print(f"  You can load it with:")
    print(
        f"  AutoModelForCausalLM.from_pretrained('{model_path}', trust_remote_code=True)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add missing Phi-3.5 files to fine-tuned models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Base Phi model to copy files from (default: microsoft/Phi-3.5-mini-instruct)",
    )
    args = parser.parse_args()

    fix_phi_model(args.model_path, args.base_model)
