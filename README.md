# Temperature Scaling in Discrete Sequence Models

Code for the experiments in **Temperature Scaling in Discrete Sequence (Language) Models**.

This repository contains training, evaluation, and plotting code for sequence-level temperature scaling in autoregressive and discrete diffusion language models. It includes GPT-2/LHTS-style baselines, BD3LM-style diffusion experiments, likelihood-ratio evaluation, variance analysis, and reasoning evaluations.

Paper: https://openreview.net/forum?id=bHIeH7450V

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate diff_env
```

Some reasoning experiments have additional dependencies under:

```bash
experiments/reasoning/power_sampling_reasoning/
```

## Repository Map

- `src/losses.py`: temperature-scaling losses and related objectives.
- `src/evaluation/`: model wrappers and sequence-level likelihood/perplexity metrics.
- `src/utils/`: configuration objects, cached dataloaders, and diffusion helper classes.
- `src/plotting/`: evaluation and plotting scripts for likelihood, variance, and temperature-scaling analyses.
- `experiments/diffusion_bd3lm/`: BD3LM-style diffusion temperature-scaling experiments.
- `experiments/autoregressive_gpt2/`: autoregressive GPT-2 baseline training code.
- `experiments/gpt2_lhts/`: GPT-2 long-horizon temperature-scaling experiments and conversion utilities.
- `experiments/reasoning/`: reasoning-focused temperature-scaling experiments.

## Example Usage

The training scripts expose command-line interfaces and should be launched directly or wrapped in local cluster scripts as needed. For example, BD3LM-style temperature-scaling experiments are under `experiments/diffusion_bd3lm/`, autoregressive baselines under `experiments/autoregressive_gpt2/`, and reasoning experiments under `experiments/reasoning/`.

Merge evaluation CSVs:

```bash
python experiments/merge_eval_csvs.py
```

## Attribution

Parts of this repository build on code from prior open-source research projects:

- `experiments/gpt2_lhts/` contains code adapted from [Long Horizon Temperature Scaling](https://github.com/AndyShih12/LongHorizonTemperatureScaling) by Andy Shih, Dorsa Sadigh, and Stefano Ermon. See `experiments/gpt2_lhts/ATTRIBUTION.md`.
- `experiments/reasoning/power_sampling_reasoning/` contains code adapted from [Reasoning with Sampling](https://github.com/aakaran/reasoning-with-sampling) by Aayush Karan and Yilun Du. See `experiments/reasoning/power_sampling_reasoning/ATTRIBUTION.md`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{scheufele2026temperature,
  title = {Temperature Scaling in Discrete Sequence (Language) Models},
  author = {Scheufele, Hannah and Blohm, Peter and Garg, Vikas},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {306},
  year = {2026},
  address = {Seoul, South Korea},
  url = {https://openreview.net/forum?id=bHIeH7450V}
}
```
