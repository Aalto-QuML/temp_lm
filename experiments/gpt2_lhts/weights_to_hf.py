import torch
from transformers import GPT2LMHeadModel


def save_model_hf(checkpoint_path, output_dir, device="cuda:0"):
    """Convert checkpoint to Hugging Face format"""
    # Load base model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load checkpoint weights
    map_location = {f"cuda:0": device}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Fix: Strip "model." prefix from keys so they match GPT2LMHeadModel structure
    model_state_dict = checkpoint["model"]
    model_state_dict = {
        k.replace("model.", ""): v
        for k, v in model_state_dict.items()
        if not k.startswith("scorer")
    }

    model.load_state_dict(model_state_dict, strict=True)

    # Save in Hugging Face format
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


# Usage:
if __name__ == "__main__":
    checkpoint_path = "/m/cs/scratch/temperature_diffusion/models/lhts/checkpoint.pth"
    output_dir = "/m/cs/scratch/temperature_diffusion/models/lhts/model"

    save_model_hf(checkpoint_path, output_dir)
