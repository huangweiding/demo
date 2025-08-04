#!/usr/bin/env python3
"""
DPO Utilities

This module provides utility functions for DPO training that can be reused
across different scripts. It includes functions for dataset preparation,
model loading, configuration, and evaluation.

Based on the TRL implementation patterns from:
- trl/trainer/dpo_trainer.py
- trl/scripts/dpo.py
- trl/trainer/dpo_config.py
"""

import os
import json
import torch
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

def create_preference_dataset_from_pairs(
    prompts: List[str],
    chosen_responses: List[str],
    rejected_responses: List[str]
) -> Dataset:
    """
    Create a preference dataset from lists of prompts, chosen, and rejected responses.
    
    Args:
        prompts: List of prompts
        chosen_responses: List of preferred responses
        rejected_responses: List of less preferred responses
    
    Returns:
        Dataset: HuggingFace dataset in DPO format
    """
    if len(prompts) != len(chosen_responses) or len(prompts) != len(rejected_responses):
        raise ValueError("All input lists must have the same length")
    
    data = []
    for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
        data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    return Dataset.from_list(data)

def load_model_and_tokenizer(
    model_name: str,
    use_lora: bool = False,
    lora_config: Optional[LoraConfig] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer with optional LoRA configuration.
    
    Args:
        model_name: Name or path of the model
        use_lora: Whether to apply LoRA
        lora_config: LoRA configuration (if None, default config will be used)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if requested
    if use_lora:
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def create_dpo_config(
    output_dir: str,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    max_prompt_length: int = 512,
    max_length: int = 1024,
    **kwargs
) -> DPOConfig:
    """
    Create a DPO configuration with common parameters.
    
    Args:
        output_dir: Directory to save the model
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        beta: DPO beta parameter
        loss_type: Type of loss function
        max_prompt_length: Maximum prompt length
        max_length: Maximum total sequence length
        **kwargs: Additional arguments to pass to DPOConfig
    
    Returns:
        DPOConfig: Configured DPO training arguments
    """
    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        # DPO specific parameters
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        loss_type=loss_type,
        # Memory optimization
        gradient_checkpointing=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=50,
        # Logging
        report_to=None,
        **kwargs
    )
    return config

def setup_dpo_trainer(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[DPOConfig] = None,
    peft_config: Optional[LoraConfig] = None
) -> DPOTrainer:
    """
    Set up a DPO trainer with the given components.
    
    Args:
        model: The model to train
        ref_model: The reference model
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config: DPO configuration (optional)
        peft_config: PEFT configuration (optional)
    
    Returns:
        DPOTrainer: Configured DPO trainer
    """
    if config is None:
        config = create_dpo_config("./dpo_model")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    return trainer

def generate_comparison(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7
) -> str:
    """
    Generate a response using the given model and prompt.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: The input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
    
    Returns:
        str: Generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    generation_config = GenerationConfig(
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the output
    response = generated_text[len(prompt):].strip()
    return response

def evaluate_response_length(response: str) -> Dict[str, Any]:
    """
    Evaluate response based on length metrics.
    
    Args:
        response: The response to evaluate
    
    Returns:
        Dict containing length metrics
    """
    words = response.split()
    sentences = response.split('.')
    
    return {
        "char_count": len(response),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "has_content": len(response.strip()) > 0
    }

def save_training_info(
    output_dir: str,
    model_name: str,
    dataset_size: int,
    config: DPOConfig,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save training information to a JSON file.
    
    Args:
        output_dir: Directory to save the info
        model_name: Name of the base model
        dataset_size: Size of the training dataset
        config: DPO configuration used
        additional_info: Additional information to save
    """
    info = {
        "model_name": model_name,
        "dataset_size": dataset_size,
        "training_config": {
            "beta": config.beta,
            "loss_type": config.loss_type,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_train_epochs,
            "batch_size": config.per_device_train_batch_size,
            "max_prompt_length": config.max_prompt_length,
            "max_length": config.max_length,
        },
        "additional_info": additional_info or {}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    info_file = os.path.join(output_dir, "training_info.json")
    
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Training info saved to: {info_file}")

def load_training_info(output_dir: str) -> Dict[str, Any]:
    """
    Load training information from a JSON file.
    
    Args:
        output_dir: Directory containing the info file
    
    Returns:
        Dict containing training information
    """
    info_file = os.path.join(output_dir, "training_info.json")
    
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Training info file not found: {info_file}")
    
    with open(info_file, "r") as f:
        return json.load(f)

def create_sample_preference_data() -> Dataset:
    """
    Create a sample preference dataset for testing and demonstration.
    
    Returns:
        Dataset: Sample preference dataset
    """
    data = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris, which is located in the north-central part of the country.",
            "rejected": "I don't know the capital of France."
        },
        {
            "prompt": "Explain photosynthesis in simple terms.",
            "chosen": "Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create food and oxygen.",
            "rejected": "Photosynthesis is something plants do."
        },
        {
            "prompt": "How do you make a cup of coffee?",
            "chosen": "To make a cup of coffee, you need to boil water, add coffee grounds to a filter, and pour the hot water over the grounds.",
            "rejected": "Just add hot water to instant coffee."
        },
        {
            "prompt": "What are the benefits of exercise?",
            "chosen": "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles, better mood, and increased energy levels.",
            "rejected": "Exercise is good for you."
        },
        {
            "prompt": "Describe the water cycle.",
            "chosen": "The water cycle involves evaporation of water from oceans and lakes, condensation into clouds, precipitation as rain or snow, and collection back into bodies of water.",
            "rejected": "Water goes up and comes down."
        }
    ]
    return Dataset.from_list(data)

def print_model_info(model: AutoModelForCausalLM, model_name: str = "Model"):
    """
    Print information about a model.
    
    Args:
        model: The model to analyze
        model_name: Name to display for the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 {model_name} Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable percentage: {trainable_params/total_params*100:.2f}%")
    
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

def validate_preference_dataset(dataset: Dataset) -> bool:
    """
    Validate that a dataset has the correct format for DPO training.
    
    Args:
        dataset: The dataset to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    required_columns = ["prompt", "chosen", "rejected"]
    
    # Check if all required columns exist
    for col in required_columns:
        if col not in dataset.column_names:
            print(f"❌ Missing required column: {col}")
            return False
    
    # Check if dataset is not empty
    if len(dataset) == 0:
        print("❌ Dataset is empty")
        return False
    
    # Check if all examples have non-empty values
    for i, example in enumerate(dataset):
        for col in required_columns:
            if not example[col] or not str(example[col]).strip():
                print(f"❌ Empty value in column '{col}' at index {i}")
                return False
    
    print("✅ Dataset validation passed")
    return True 