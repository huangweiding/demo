# DPO (Direct Preference Optimization) Tutorial

This tutorial demonstrates how to use the Direct Preference Optimization (DPO) algorithm to train language models from preference data. DPO is a stable, performant, and computationally lightweight alternative to RLHF that eliminates the need for sampling from the LM during fine-tuning.

## Overview

DPO is based on the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290) by Rafael Rafailov et al.

### Key Features

- **Stability**: DPO is more stable than traditional RLHF methods
- **Performance**: Achieves comparable or better results than PPO-based RLHF
- **Simplicity**: Eliminates the need for reward model training and complex sampling procedures
- **Efficiency**: Computationally lightweight with simple classification loss

## Tutorial Structure

1. **Basic DPO Training** (`basic_dpo_training.py`): Simple DPO training example
2. **Custom Dataset DPO** (`custom_dataset_dpo.py`): Using custom preference datasets
3. **LoRA DPO Training** (`lora_dpo_training.py`): DPO with LoRA for memory efficiency
4. **DPO with Custom Loss** (`custom_loss_dpo.py`): Different loss functions (sigmoid, hinge, IPO, etc.)
5. **DPO Evaluation** (`dpo_evaluation.py`): Evaluating DPO models
6. **DPO Utils** (`dpo_utils.py`): Utility functions for DPO training

## Quick Start

```bash
# Basic DPO training
python basic_dpo_training.py

# DPO with LoRA
python lora_dpo_training.py

# Custom dataset DPO
python custom_dataset_dpo.py
```

## Requirements

- transformers >= 4.36.0
- trl >= 0.7.0
- datasets
- torch
- accelerate
- peft (for LoRA)

## Data Format

DPO requires preference data in the following format:

```python
{
    "prompt": "Your prompt here",
    "chosen": "Preferred response",
    "rejected": "Less preferred response"
}
```

## Key Parameters

- `beta`: Controls deviation from reference model (default: 0.1)
- `loss_type`: Type of loss function (sigmoid, hinge, ipo, etc.)
- `learning_rate`: Learning rate for optimization
- `max_prompt_length`: Maximum prompt length
- `max_length`: Maximum total sequence length

## References

- [DPO Paper](https://huggingface.co/papers/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [DPO Trainer Documentation](https://huggingface.co/docs/trl/dpo_trainer) 