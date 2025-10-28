import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

# TRL GRPO
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer

from .dataset import MultiTurnJsonlDataset, collate_for_generation
from .prompting import build_prompt
from .reward_fns import RewardConfig, score_response


@dataclass
class ScriptConfig:
    model_name: str
    data_path: str
    output_dir: str
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    num_generations: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    bf16: bool = False
    seed: int = 42


def parse_args() -> ScriptConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--bf16", type=str, default="False")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return ScriptConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        bf16=args.bf16.lower() == "true",
        seed=args.seed,
    )


def build_trl_config(cfg: ScriptConfig) -> TRLGRPOConfig:
    return TRLGRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        bf16=cfg.bf16,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="no",
        remove_unused_columns=False,
    )


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else None,
        device_map="auto",
    )

    dataset = MultiTurnJsonlDataset(cfg.data_path)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_for_generation(batch, tokenizer),
    )

    trl_cfg = build_trl_config(cfg)

    # Define reward function wrapper expected by TRL GRPOTrainer
    reward_cfg = RewardConfig()

    def reward_fn(samples: List[Dict[str, str]]) -> List[float]:
        rewards: List[float] = []
        for s in samples:
            prompt = s["prompt"]
            response = s["response"]
            last_user = s.get("last_user", "")
            rewards.append(score_response(prompt, response, last_user, reward_cfg))
        return rewards

    # Build prompts via format function expected by GRPOTrainer
    def format_batch_prompts(features: List[Dict]) -> List[str]:
        prompts: List[str] = []
        for f in features:
            turns = f["conversation"]
            prompts.append(build_prompt(turns))
        return prompts

    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        args=trl_cfg,
        train_dataset=dataset,
        formatting_func=format_batch_prompts,
        reward_funcs=[reward_fn],
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()


