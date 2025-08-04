#!/usr/bin/env python3
"""
Custom Dataset DPO Training Tutorial

This script demonstrates how to use custom preference datasets with DPO training.
It shows different ways to prepare and load preference data for DPO training.

Based on the TRL implementation and dataset examples from:
- trl/trainer/dpo_trainer.py
- examples/datasets/ultrafeedback.py
"""

import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

def create_custom_preference_dataset():
    """
    Create a custom preference dataset for DPO training.
    This function demonstrates different ways to structure preference data.
    """
    
    # Method 1: Simple preference data
    simple_data = [
        {
            "prompt": "Write a short story about a robot learning to paint.",
            "chosen": "In a sunlit studio, Robot-7 carefully dipped its metallic fingers into vibrant paint. At first, its strokes were rigid and mechanical, but with each canvas, it discovered the joy of creation. The robot's paintings evolved from simple geometric shapes to flowing landscapes, capturing the beauty it observed in the world around it.",
            "rejected": "A robot painted a picture."
        },
        {
            "prompt": "Explain quantum computing in simple terms.",
            "chosen": "Quantum computing uses quantum mechanics to process information. Instead of regular bits that are either 0 or 1, quantum computers use qubits that can be 0, 1, or both at the same time. This allows them to solve certain problems much faster than classical computers.",
            "rejected": "Quantum computing is complicated computer stuff."
        },
        {
            "prompt": "What are the environmental benefits of renewable energy?",
            "chosen": "Renewable energy sources like solar, wind, and hydroelectric power produce little to no greenhouse gas emissions, reduce air pollution, conserve water resources, and help combat climate change while providing sustainable energy for future generations.",
            "rejected": "Renewable energy is good for the environment."
        }
    ]
    
    # Method 2: More complex preference data with multiple aspects
    complex_data = [
        {
            "prompt": "Describe the process of making bread from scratch.",
            "chosen": "To make bread from scratch, you'll need flour, water, yeast, salt, and sugar. First, mix warm water with yeast and sugar to activate it. Combine flour and salt, then gradually add the yeast mixture to form a dough. Knead the dough for 10-15 minutes until smooth and elastic. Let it rise in a warm place for 1-2 hours, then shape it and let it rise again. Finally, bake at 375°F for 30-40 minutes until golden brown.",
            "rejected": "Mix flour and water, add yeast, let it rise, and bake it."
        },
        {
            "prompt": "What are the key principles of effective communication?",
            "chosen": "Effective communication involves active listening, clear and concise messaging, empathy, appropriate body language, asking clarifying questions, providing constructive feedback, and adapting your communication style to your audience. It also requires being present, avoiding distractions, and ensuring mutual understanding.",
            "rejected": "Talk clearly and listen to others."
        }
    ]
    
    # Combine all data
    all_data = simple_data + complex_data
    return Dataset.from_list(all_data)

def load_huggingface_dataset():
    """
    Load a preference dataset from Hugging Face Hub.
    This demonstrates how to use existing preference datasets.
    """
    try:
        # Try to load a small preference dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
        
        # Convert to DPO format (this is a simplified conversion)
        dpo_data = []
        for example in dataset:
            if "instruction" in example and "output" in example:
                # Create a simple preference pair
                prompt = example["instruction"]
                chosen = example["output"]
                # Create a less preferred version (simplified)
                rejected = chosen[:len(chosen)//2] if len(chosen) > 10 else "I don't know."
                
                dpo_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        
        return Dataset.from_list(dpo_data)
    except Exception as e:
        print(f"Could not load Hugging Face dataset: {e}")
        print("Falling back to custom dataset...")
        return create_custom_preference_dataset()

def prepare_dataset_for_dpo(dataset, tokenizer, max_length=1024):
    """
    Prepare dataset for DPO training by tokenizing the data.
    This function demonstrates the preprocessing steps needed for DPO.
    """
    
    def tokenize_function(examples):
        # Tokenize prompts
        prompt_tokens = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_length//2,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize chosen responses
        chosen_tokens = tokenizer(
            examples["chosen"],
            truncation=True,
            max_length=max_length//2,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize rejected responses
        rejected_tokens = tokenizer(
            examples["rejected"],
            truncation=True,
            max_length=max_length//2,
            padding=False,
            return_tensors=None,
        )
        
        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset

def main():
    # Configuration
    model_name = "microsoft/DialoGPT-medium"
    output_dir = "./custom_dpo_model"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Custom Dataset DPO Training Tutorial")
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
    
    # Step 2: Create reference model
    print("2️⃣ Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Step 3: Load and prepare dataset
    print("3️⃣ Loading and preparing dataset...")
    
    # Try to load from Hugging Face first, fall back to custom dataset
    dataset = load_huggingface_dataset()
    print(f"   Dataset size: {len(dataset)} examples")
    
    # Show sample data
    print("\n📋 Sample dataset entries:")
    for i, example in enumerate(dataset[:2]):
        print(f"   Example {i+1}:")
        print(f"     Prompt: {example['prompt'][:100]}...")
        print(f"     Chosen: {example['chosen'][:100]}...")
        print(f"     Rejected: {example['rejected'][:100]}...")
        print()
    
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
        beta=0.1,
        max_prompt_length=256,
        max_length=512,
        loss_type="sigmoid",
        # Memory optimization
        gradient_checkpointing=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=50,
        # Logging
        report_to=None,
    )
    
    # Step 5: Initialize DPO Trainer
    print("5️⃣ Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
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
    
    print(f"\n✅ Custom dataset DPO training completed!")
    print(f"📁 Model saved to: {output_dir}")
    
    # Step 9: Save dataset info
    dataset_info = {
        "dataset_size": len(dataset),
        "model_name": model_name,
        "training_args": {
            "beta": training_args.beta,
            "loss_type": training_args.loss_type,
            "learning_rate": training_args.learning_rate,
        }
    }
    
    with open(f"{output_dir}/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n📝 Dataset information saved to dataset_info.json")
    print("\n💡 Tips for custom datasets:")
    print("   - Ensure your preference data is high quality")
    print("   - Balance the length of chosen vs rejected responses")
    print("   - Consider using multiple annotators for preference data")
    print("   - Experiment with different beta values for your specific use case")

if __name__ == "__main__":
    main() 