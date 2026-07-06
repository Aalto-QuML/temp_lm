import argparse
import math
import os
from dataclasses import dataclass
from typing import Literal, Optional, List
from pathlib import Path

import sys


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

import torch
import torch._dynamo

torch._dynamo.config.disable = True  # global switch

import utils.config_classes as config_classes
from plotting.evaluation_battery import run_battery_to_csv
from plotting.evaluation_battery_ar import run_ar_battery_to_csv
from utils.config_classes import ModelSpec
from transformers import GPT2LMHeadModel

# 1. Hardcoded list of model folders derived from the commands
# Base path: /scratch/work/scheufh1/models
# Common params: block_length=4, temp=0.1, lr=1e-5, ws=300, ns=30000, bs=16, len=128
# Derived params: pairs=2 (from bs=16, block=4), sh=16 (default), seed=42 (default)
PREFIX = "/m/cs/scratch/temperature_diffusion/models/"

MODEL_FOLDERS = [
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.1_elbo1.0_kl0.0_ratio0.0_huber0.0ao0.2_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.1_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.2_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.1_elbo1.0_kl0.0_ratio0.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.1_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # # ===========================================================================================================================
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp3.0_elbo1.0_kl0.0_ratio0.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp3.0_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # # ============================================================================================================================
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio0.0_huber0.0ao0.0_lhts1.0_sh16_pairs2_seed42_lr5e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.0_lhts0.0_sh16_pairs2_seed42_lr5e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo1.0_kl0.0_ratio0.0_huber0.0ao0.0_lhts0.0_sh16_pairs2_seed42_lr5e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo1.0_kl0.0_ratio0.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr5e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr5e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio0.0_huber0.0ao0.0_lhts1.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.0_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo1.0_kl0.0_ratio0.0_huber0.0ao0.0_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo1.0_kl0.0_ratio0.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # PREFIX
    # + "bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    # "/m/cs/work/scheufh1/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.1_ao0.01_seed42_lr0.0001_bs128_len128_ws300_ns30000",
    # "/m/cs/work/scheufh1/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.8_ao0.0_seed42_lr0.0001_bs128_len128_ws300_ns30000",
    # "/m/cs/work/scheufh1/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.8_ao0.01_seed42_lr0.0001_bs128_len128_ws300_ns30000",
    # "/m/cs/work/scheufh1/models/bd3lm-finetuned-openwebtext-blocksize4_temp3.0_ao0.01_seed42_lr0.0001_bs128_len128_ws300_ns30000",
    # "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.1_ao0.0_seed42_lr1e-05_bs128_len128_ws100_ns30000",
    "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.1_ao0.05_seed42_lr0.0001_bs128_len128_ws100_ns30000",
    "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.1_ao0.1_seed42_lr0.0001_bs128_len128_ws100_ns30000",
    # "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.8_ao0.0_seed42_lr1e-05_bs128_len128_ws100_ns30000",
    # "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.8_ao0.05_seed42_lr0.0001_bs128_len128_ws100_ns30000",
    # "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp3.0_ao0.0_seed42_lr5e-05_bs128_len128_ws100_ns30000",
    "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp3.0_ao0.05_seed42_lr0.0001_bs128_len128_ws100_ns30000",
    "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp3.0_ao0.1_seed42_lr0.0001_bs128_len128_ws100_ns30000",
]

AR_MODEL_FOLDERS = [
    "/m/cs/scratch/temperature_diffusion/models/ar-finetuned-gpt2-alpha0.05-seed42-temp0.1-lr0.0001_bs32_len1024",
    "/m/cs/scratch/temperature_diffusion/models/ar-finetuned-gpt2-alpha0.05-seed42-temp3.0-lr0.0001_bs32_len1024",
    # "/m/cs/scratch/temperature_diffusion/models/ar-finetuned-gpt2-alpha0.0-seed42-temp0.1-lr5e-06_bs32_len1024",
]

AR_08_MODELS = [
    "/m/cs/scratch/temperature_diffusion/models/ar-finetuned-gpt2-alpha0.05-seed42-temp0.8-lr0.0001_bs32_len1024/model",
    "/m/cs/scratch/temperature_diffusion/models/ar-finetuned-gpt2-alpha0.0-seed42-temp0.8-lr1e-05_bs32_len1024/model",
    "/m/cs/scratch/temperature_diffusion/models/ar-finetuned-gpt2-alpha0.0-seed42-temp0.1-lr5e-06_bs32_len1024/model",
    "/m/cs/scratch/temperature_diffusion/models/ar-finetuned-gpt2-alpha0.0-seed42-temp3.0-lr0.0001_bs32_len1024/model",
    "/m/cs/scratch/temperature_diffusion/models/lhts/hf_model",
]

BD_08_MODELS = [
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.0_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300/model",
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.0_lhts0.0_sh16_pairs2_seed42_lr5e-05_bs16_len128_ws300/model",
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300/model",
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr5e-05_bs16_len128_ws300/model",
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.2_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300/model",
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp0.8_K1_ratio1.0_ao0.0_seed42_lr1e-05_bs32_len128_ws100_ns30000/model",
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp0.8_K1_ratio1.0_ao0.2_seed42_lr1e-05_bs32_len128_ws100_ns30000/model",
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp0.8_K4_ratio1.0_ao0.0_seed42_lr1e-05_bs32_len128_ws100_ns30000/model",
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp0.8_K4_ratio1.0_ao0.2_seed42_lr1e-05_bs32_len128_ws100_ns30000/model",
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio0.0_huber0.0ao0.0_lhts1.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300/model",
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.8_elbo0.0_kl0.0_ratio0.0_huber0.0ao0.0_lhts1.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300/model",
]

BD_MODELS_NEW_SCHEDULE = [
    # "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.1_ao0.1_seed42_lr0.0001_bs128_len128_ws100_ns30000_new_schedule",
    # "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp3.0_ao0.2_seed42_lr0.0001_bs128_len128_ws100_ns30000_new_schedule",
    "/m/cs/scratch/temperature_diffusion/models/bd3lm-finetuned-openwebtext-blocksize4_temp0.1_ao0.05_seed42_lr0.0001_bs32_len128_ws100_ns30000_new_schedule",
]

MC_MODELS = [
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp3.0_K4_ratio1.0_ao0.2_seed42_lr1e-05_bs32_len128_ws100_ns30000",
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp3.0_K1_ratio1.0_ao0.2_seed42_lr1e-05_bs32_len128_ws100_ns30000",
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp0.1_K4_ratio1.0_ao0.2_seed42_lr1e-05_bs32_len128_ws100_ns30000",
    "/m/cs/scratch/temperature_diffusion/models2/bd3lm-finetuned-openwebtext-MC-blocksize4_emp0.1_K1_ratio1.0_ao0.2_seed42_lr1e-05_bs32_len128_ws100_ns30000",
]

BD_MODELS = [
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp0.1_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
    "/m/cs/scratch/temperature_diffusion/models/models/bd3lm-finetuned-openwebtext-blocksize4_emp3.0_elbo0.0_kl0.0_ratio1.0_huber0.0ao0.1_lhts0.0_sh16_pairs2_seed42_lr1e-05_bs16_len128_ws300",
]


def load_model_lhts(checkpoint_path, device="cuda:0"):
    """Load model - just the base GPT2 with trained weights"""
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    map_location = {f"cuda:0": device}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Extract just the model weights
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    return model


def get_checkpoints_as_specs(
    model_folders: List[str], seq_length=128, temperature=None
) -> List[ModelSpec]:
    """
    Takes a list of model folder paths, finds all checkpoints, and returns a list of ModelSpecs.
    """
    specs = []

    # Parameters known from the commands that generated these models
    # seq_length = 128
    temperature = None  # IF YOU CHANGE THIS I WILL BE VIOLENT

    for folder_path in model_folders:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Warning: Folder does not exist: {folder}")
            # assert False
            continue

        # Find all checkpoints in the folder
        # Checkpoints are subdirectories starting with "checkpoint_" or named "model"
        for child in folder.iterdir():
            if child.is_dir():
                if child.name == "model" or child.name.startswith("checkpoint_"):
                    spec = ModelSpec(
                        model_id=str(child.absolute()),
                        max_length=seq_length,
                        myopic_temperature=temperature,
                        device="cuda",
                    )
                    specs.append(spec)

    return specs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--model_id", type=int, help="Model identifier", default=0)
    args = parser.parse_args()

    # model_name = "gpt2"
    model_name = "kuleshov-group/bd3lm-owt-block_size4"
    max_length = 128  # 1024
    batch_size = 64  # 64 32
    model_specs = [
        config_classes.ModelSpec(
            model_id=model_name,
            max_length=max_length,
            device="cuda",
            myopic_temperature=float(myopic_temperature),
        )
        # for myopic_temperature in torch.linspace(0.1, 3.1, 31)
        for myopic_temperature in [1.0]  # torch.logspace(math.log10(0.1), 0.5, 30 + 1)
    ]
    base_model_spec = model_specs[0]

    offset = 10**5
    size = 9000

    dataloader_spec = config_classes.OpenWebTextLoaderSpec(
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        slice_start=offset,
        slice_end=offset + size,
    )

    # specs = get_checkpoints_as_specs(BD_MODELS)
    specs = [
        config_classes.ModelSpec(
            model_id=model,
            max_length=max_length,
            device="cuda",
        )
        for model in BD_08_MODELS
    ]
    # specs += [
    #     config_classes.ModelSpec(
    #         model_id=model_name,
    #         max_length=max_length,
    #         device="cuda",
    #         myopic_temperature=myopic_temperature,
    #     )
    #     for myopic_temperature in [
    #         0.1,
    #         0.8,
    #         3.0,
    #     ]  # torch.logspace(math.log10(0.1), 0.5, 30 + 1)
    # ]
    id = args.model_id
    # print(f"Found {len(specs)} checkpoints.")
    # for s in specs:
    # assert False, "run the eval script/function with this spec"

    # Use model_id to select a single checkpoint for parallel execution
    if id >= len(specs):
        print(f"Error: model_id {id} is out of range (only {len(specs)} specs found)")
        exit(1)

    # out = run_battery_to_csv(
    #     model_specs=[specs[id]],  # Evaluate only the selected model
    #     base_model_spec=base_model_spec,
    #     data_spec=dataloader_spec,
    #     out_dir=Path("/m/cs/scratch/temperature_diffusion/results"),
    #     cache_dir=Path("/m/cs/scratch/temperature_diffusion/.cache/bd3lm_eval"),
    #     run_id=id,  # Pass run_id for parallel-safe file naming
    # )
    out = run_battery_to_csv(
        model_specs=[specs[id]],  # Evaluate only the selected model
        base_model_spec=base_model_spec,
        data_spec=dataloader_spec,
        out_dir=Path("/m/cs/scratch/temperature_diffusion/results"),
        cache_dir=Path("/m/cs/scratch/temperature_diffusion/.cache/bd3lm_eval"),
        save_pair_csvs=True,
        pair_csv_dir=Path("/m/cs/scratch/temperature_diffusion/results/pair_csvs"),
        # run_id=id,  # Pass run_id for parallel-safe file naming
    )
    print("Wrote:", out)
