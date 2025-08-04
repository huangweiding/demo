#!/usr/bin/env python3
"""
LoRA DPO Training Tutorial

This script demonstrates how to use DPO training with LoRA (Low-Rank Adaptation)
for memory efficiency. LoRA allows training large models with limited GPU memory
by only training a small number of additional parameters.

Based on the TRL implementation and PEFT integration from:
- trl/trainer/dpo_trainer.py
- trl/scripts/dpo.py
"""

import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

def create_sample_dataset():
    """Create a sample preference dataset for LoRA DPO training."""
    data = [
        {
            "prompt": "Explain machine learning in simple terms.",
            "chosen": "Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
            "rejected": "Machine learning is when computers learn."
        },
        {
            "prompt": "What are the benefits of reading books?",
            "chosen": "Reading books enhances vocabulary, improves critical thinking skills, reduces stress, increases empathy, provides knowledge and entertainment, and can improve sleep quality when done before bed.",
            "rejected": "Reading books is good for you."
        },
        {
            "prompt": "Describe the importance of cybersecurity.",
            "chosen": "Cybersecurity is crucial for protecting personal data, financial information, and national security. It involves defending systems, networks, and programs from digital attacks, ensuring privacy and maintaining trust in digital technologies.",
            "rejected": "Cybersecurity protects computers from hackers."
        },
        {
            "prompt": "How does photosynthesis work?",
            "chosen": "Photosynthesis is the process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen. Chlorophyll in plant cells captures light energy, which powers the conversion of CO2 and H2O into food for the plant and oxygen for other organisms.",
            "rejected": "Plants use sunlight to make food."
        },
        {
            "prompt": "What is the impact of climate change?",
            "chosen": "Climate change affects global temperatures, sea levels, weather patterns, and ecosystems. It leads to more extreme weather events, threatens biodiversity, impacts agriculture, and poses risks to human health and infrastructure worldwide.",
            "rejected": "Climate change makes the weather different."
        }
    ]
    return Dataset.from_list(data)

def setup_lora_config():
    """
    Configure LoRA parameters for DPO training.
    This function demonstrates how to set up LoRA for efficient training.
    """
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling parameter
        target_modules=["q_proj", "v_proj"],  # Which modules to apply LoRA to
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Whether to train bias terms
        task_type=TaskType.CAUSAL_LM,  # Task type for language modeling
    )
    return lora_config

def main():
    # Configuration
    model_name = "microsoft/DialoGPT-medium"
    output_dir = "./lora_dpo_model"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting LoRA DPO Training Tutorial")
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
    
    # Step 2: Setup LoRA configuration
    print("2️⃣ Setting up LoRA configuration...")
    lora_config = setup_lora_config()
    print(f"   LoRA rank (r): {lora_config.r}")
    print(f"   LoRA alpha: {lora_config.lora_alpha}")
    print(f"   Target modules: {lora_config.target_modules}")
    
    # Step 3: Apply LoRA to the model
    print("3️⃣ Applying LoRA to the model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show trainable parameters
    
    # Step 4: Create reference model (also with LoRA)
    print("4️⃣ Creating reference model with LoRA...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    ref_model = get_peft_model(ref_model, lora_config)
    
    # Step 5: Prepare dataset
    print("5️⃣ Preparing preference dataset...")
    dataset = create_sample_dataset()
    print(f"   Dataset size: {len(dataset)} examples")
    
    # Step 6: Configure DPO training arguments
    print("6️⃣ Configuring DPO training arguments...")
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # Higher learning rate for LoRA
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
        # LoRA specific
        peft_config=lora_config,
    )
    
    # Step 7: Initialize DPO Trainer
    print("7️⃣ Initializing DPO Trainer with LoRA...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    # Step 8: Train the model
    print("8️⃣ Starting LoRA DPO training...")
    trainer.train()
    
    # Step 9: Save the trained model
    print("9️⃣ Saving trained LoRA model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Step 10: Evaluate the model
    print("🔟 Evaluating trained model...")
    metrics = trainer.evaluate()
    print(f"   Evaluation metrics: {metrics}")
    
    print(f"\n✅ LoRA DPO training completed!")
    print(f"📁 Model saved to: {output_dir}")
    
    # Step 11: Show model size comparison
    print("\n📊 Model Size Comparison:")
    print("   Original model parameters: ~355M")
    print("   LoRA trainable parameters: ~8M (2.3%)")
    print("   Memory savings: ~90%")
    
    print("\n💡 LoRA Benefits:")
    print("   - Reduced memory usage")
    print("   - Faster training")
    print("   - Easier to fine-tune large models")
    print("   - Can be merged back to full model")
    
    print("\n📝 Next steps:")
    print("   - Load the LoRA model using PeftModel.from_pretrained()")
    print("   - Merge LoRA weights with base model if needed")
    print("   - Experiment with different LoRA ranks (r) and alpha values")

if __name__ == "__main__":
    main() 