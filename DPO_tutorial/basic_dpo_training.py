#!/usr/bin/env python3
"""
Basic DPO Training Tutorial

This script demonstrates the basic usage of Direct Preference Optimization (DPO)
using the TRL library. It shows how to:
1. Load a pre-trained model and tokenizer
2. Prepare preference data
3. Configure DPO training parameters
4. Train the model using DPO
5. Save the trained model

Based on the TRL implementation from:
- trl/trainer/dpo_trainer.py
- trl/scripts/dpo.py
"""

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig

def create_sample_dataset():
    """
    Create a sample preference dataset for demonstration.
    In practice, you would load your own preference data.
    """
    data = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
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

def main():
    # Configuration
    model_name = "microsoft/DialoGPT-medium"  # Small model for tutorial
    output_dir = "./dpo_trained_model"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Basic DPO Training Tutorial")
    print(f"📦 Model: {model_name}")
    print(f"📁 Output Directory: {output_dir}")
    
    # Step 1: Load model and tokenizer
    print("\n1️⃣ Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 2: Create reference model (copy of the original model)
    print("2️⃣ Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Step 3: Prepare dataset
    print("3️⃣ Preparing preference dataset...")
    dataset = create_sample_dataset()
    print(f"   Dataset size: {len(dataset)} examples")
    
    # Step 4: Configure DPO training arguments
    print("4️⃣ Configuring DPO training arguments...")
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        # DPO specific parameters
        beta=0.1,  # Controls deviation from reference model
        max_prompt_length=512,
        max_length=1024,
        loss_type="sigmoid",  # Default DPO loss
        # Memory optimization
        gradient_checkpointing=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=50,
        # Logging
        report_to=None,  # Disable wandb/comet for this tutorial
    )
    
    # Step 5: Initialize DPO Trainer
    print("5️⃣ Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # Using same dataset for eval in this tutorial
        processing_class=tokenizer,
        data_collator=None,  # Will use default DataCollatorForPreference
    )
    
    # Step 6: Train the model
    print("6️⃣ Starting DPO training...")
    trainer.train()
    
    # Step 7: Save the trained model
    print("7️⃣ Saving trained model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Step 8: Evaluate the model
    print("8️⃣ Evaluating trained model...")
    metrics = trainer.evaluate()
    print(f"   Evaluation metrics: {metrics}")
    
    print(f"\n✅ DPO training completed! Model saved to: {output_dir}")
    print("\n📝 Next steps:")
    print("   - Load the trained model using AutoModelForCausalLM.from_pretrained()")
    print("   - Test the model with prompts to see the improvement")
    print("   - Try different beta values to control the deviation from the reference model")

if __name__ == "__main__":
    main() 