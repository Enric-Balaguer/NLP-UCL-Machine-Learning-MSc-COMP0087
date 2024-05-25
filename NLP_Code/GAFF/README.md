# Testing effects of different active-forgetting timing configurations during pre-training on increased finetuning convergence speeds

This repository implements a method to shape a GPT-NeoX training process by starting training runs to then interrupt them once certain checkpoints are reached. These interruptions allow to modify these checkpoints, representing an approach to curriculum learning or meta learning (e.g. Active Forgetting).

## Setup

Head to [GPT-Neox](https://github.com/EleutherAI/gpt-neox) and install all required dependencies for training.

To then start the Active Forgetting pipeline, set following environment variables:
- REPOSITORY_LOCATION
- TRAINING_CONFIG
- CHECKPOINT_DIR
- AF_INTERVAL
- AF_FIRST_RESET
- AF_LAST_RESET
- WANDB_API_KEY

Under assumption that the training config provided would run through a standard GPT-NeoX training, you can now start the Active Forgetting pipeline via

```bash
python active_forgetting_run.py
```
