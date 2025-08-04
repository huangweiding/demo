#!/usr/bin/env python3
"""
Custom Loss DPO Training Tutorial

This script demonstrates different loss functions available in DPO training.
The TRL library supports multiple loss types including sigmoid, hinge, IPO, and others.

Based on the DPO loss implementations from:
- trl/trainer/dpo_trainer.py
- trl/trainer/dpo_config.py
"""

import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

def create_sample_dataset():
    """Create a sample preference dataset for testing different loss functions."""
    data = [
        {
            "prompt": "Write a creative story about a time traveler.",
            "chosen": "In the year 2157, Dr. Sarah Chen adjusted the quantum stabilizer on her temporal displacement device. The machine hummed with otherworldly energy as she prepared to step into the unknown. With a deep breath, she activated the sequence, and reality itself seemed to bend around her. When her vision cleared, she found herself standing in the midst of the 1969 Woodstock festival, surrounded by peace signs and flower power. The contrast between her sterile future and this vibrant past was overwhelming, and she realized that sometimes the greatest discoveries aren't about where you go, but what you learn about humanity along the way.",
            "rejected": "A time traveler went back in time and saw old things."
        },
        {
            "prompt": "Explain the concept of quantum entanglement.",
            "chosen": "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently, even when separated by large distances. When particles are entangled, measuring one particle instantly affects the state of the other, regardless of the distance between them. This 'spooky action at a distance,' as Einstein called it, violates classical notions of locality and has been experimentally verified, forming the foundation for quantum computing and quantum cryptography.",
            "rejected": "Quantum entanglement is when particles are connected."
        },
        {
            "prompt": "Describe the process of making a documentary film.",
            "chosen": "Creating a documentary film involves extensive research, planning, and execution. The process begins with choosing a compelling topic and conducting thorough research to understand the subject matter. Next, filmmakers develop a treatment or outline, secure funding, and assemble a crew. Pre-production includes location scouting, casting interviews, and creating shot lists. During production, the team captures interviews, b-roll footage, and ambient sound. Post-production involves editing hours of footage into a coherent narrative, adding music, graphics, and sound effects. The final stages include color grading, sound mixing, and distribution planning.",
            "rejected": "Making a documentary involves filming and editing."
        },
        {
            "prompt": "What are the ethical considerations in artificial intelligence?",
            "chosen": "AI ethics encompasses concerns about bias and fairness in algorithms, privacy and data protection, transparency and explainability of AI decisions, accountability for AI actions, job displacement and economic impact, safety and control of autonomous systems, and the potential for AI to be used maliciously. These considerations require careful balancing of technological advancement with human values, ongoing dialogue between technologists and ethicists, and the development of frameworks to ensure AI benefits society while minimizing harm.",
            "rejected": "AI ethics is about making sure AI is good."
        },
        {
            "prompt": "Explain the importance of biodiversity conservation.",
            "chosen": "Biodiversity conservation is crucial for maintaining ecosystem stability, providing essential ecosystem services like pollination and water purification, supporting food security through genetic diversity in crops, offering potential sources for new medicines and materials, contributing to climate regulation, and preserving cultural and aesthetic values. The loss of biodiversity threatens these benefits and can lead to ecosystem collapse, reduced resilience to environmental changes, and the extinction of species that may hold unknown value for future generations.",
            "rejected": "Biodiversity conservation protects animals and plants."
        }
    ]
    return Dataset.from_list(data)

def train_with_loss_type(loss_type, model_name, output_dir, dataset):
    """
    Train a DPO model with a specific loss type.
    
    Args:
        loss_type: Type of loss function to use
        model_name: Name of the base model
        output_dir: Directory to save the model
        dataset: Training dataset
    """
    
    print(f"\n🔄 Training with {loss_type} loss...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Configure training arguments with specific loss type
    training_args = DPOConfig(
        output_dir=f"{output_dir}_{loss_type}",
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
        logging_dir=f"{output_dir}_{loss_type}/logs",
        remove_unused_columns=False,
        # DPO specific parameters
        beta=0.1,
        max_prompt_length=256,
        max_length=512,
        loss_type=loss_type,  # Use the specified loss type
        # Memory optimization
        gradient_checkpointing=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=50,
        # Logging
        report_to=None,
    )
    
    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(f"{output_dir}_{loss_type}")
    tokenizer.save_pretrained(f"{output_dir}_{loss_type}")
    
    # Evaluate the model
    metrics = trainer.evaluate()
    print(f"   {loss_type} loss - Evaluation metrics: {metrics}")
    
    return metrics

def main():
    # Configuration
    model_name = "microsoft/DialoGPT-medium"
    output_dir = "./custom_loss_dpo_model"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Custom Loss DPO Training Tutorial")
    print(f"📦 Model: {model_name}")
    print(f"📁 Output Directory: {output_dir}")
    
    # Prepare dataset
    print("\n1️⃣ Preparing preference dataset...")
    dataset = create_sample_dataset()
    print(f"   Dataset size: {len(dataset)} examples")
    
    # Define loss types to test
    loss_types = [
        "sigmoid",      # Default DPO loss
        "hinge",        # Hinge loss from SLiC paper
        "ipo",          # IPO loss
        "robust",       # Robust DPO loss
    ]
    
    print("\n2️⃣ Available loss types:")
    for i, loss_type in enumerate(loss_types, 1):
        print(f"   {i}. {loss_type}")
    
    print("\n3️⃣ Training with different loss types...")
    
    results = {}
    
    # Train with each loss type
    for loss_type in loss_types:
        try:
            metrics = train_with_loss_type(loss_type, model_name, output_dir, dataset)
            results[loss_type] = metrics
        except Exception as e:
            print(f"   ❌ Error training with {loss_type} loss: {e}")
            results[loss_type] = {"error": str(e)}
    
    # Print comparison
    print("\n4️⃣ Loss Type Comparison:")
    print("=" * 60)
    print(f"{'Loss Type':<15} {'Train Loss':<15} {'Eval Loss':<15}")
    print("=" * 60)
    
    for loss_type, metrics in results.items():
        if "error" in metrics:
            print(f"{loss_type:<15} {'ERROR':<15} {'ERROR':<15}")
        else:
            train_loss = metrics.get("train_loss", "N/A")
            eval_loss = metrics.get("eval_loss", "N/A")
            print(f"{loss_type:<15} {str(train_loss):<15} {str(eval_loss):<15}")
    
    print("\n💡 Loss Type Explanations:")
    print("   - sigmoid: Standard DPO loss from the original paper")
    print("   - hinge: Hinge loss that focuses on margin between chosen/rejected")
    print("   - ipo: Identity Preference Optimization loss")
    print("   - robust: Robust DPO loss that handles preference noise")
    
    print("\n📝 Key Differences:")
    print("   - sigmoid: Most commonly used, stable training")
    print("   - hinge: May provide better separation between responses")
    print("   - ipo: Alternative formulation with different regularization")
    print("   - robust: Better for noisy preference data")
    
    print(f"\n✅ Custom loss DPO training completed!")
    print(f"📁 Models saved to: {output_dir}_[loss_type]")
    
    print("\n🔍 Next steps:")
    print("   - Compare the quality of responses from different loss types")
    print("   - Experiment with different beta values for each loss type")
    print("   - Try other loss types like 'exo_pair', 'nca_pair', 'bco_pair'")
    print("   - Analyze which loss type works best for your specific use case")

if __name__ == "__main__":
    main() 