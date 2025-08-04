#!/usr/bin/env python3
"""
Example Usage of DPO Utilities

This script demonstrates how to use the utility functions from dpo_utils.py
to create a complete DPO training pipeline with minimal code.
"""

import os
from dpo_utils import (
    create_sample_preference_data,
    load_model_and_tokenizer,
    create_dpo_config,
    setup_dpo_trainer,
    save_training_info,
    validate_preference_dataset,
    print_model_info
)

def main():
    """Demonstrate the usage of DPO utilities."""
    
    print("🚀 DPO Utilities Example Usage")
    print("=" * 50)
    
    # Step 1: Create sample dataset
    print("\n1️⃣ Creating sample preference dataset...")
    dataset = create_sample_preference_data()
    print(f"   Dataset size: {len(dataset)} examples")
    
    # Step 2: Validate dataset
    print("\n2️⃣ Validating dataset...")
    is_valid = validate_preference_dataset(dataset)
    if not is_valid:
        print("❌ Dataset validation failed")
        return
    
    # Step 3: Load model and tokenizer
    print("\n3️⃣ Loading model and tokenizer...")
    model_name = "microsoft/DialoGPT-medium"
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Step 4: Print model information
    print_model_info(model, "Base Model")
    
    # Step 5: Create reference model
    print("\n4️⃣ Creating reference model...")
    ref_model, _ = load_model_and_tokenizer(model_name)
    
    # Step 6: Create DPO configuration
    print("\n5️⃣ Creating DPO configuration...")
    config = create_dpo_config(
        output_dir="./example_dpo_model",
        learning_rate=5e-5,
        num_epochs=1,
        beta=0.1,
        loss_type="sigmoid"
    )
    print(f"   Beta: {config.beta}")
    print(f"   Loss type: {config.loss_type}")
    print(f"   Learning rate: {config.learning_rate}")
    
    # Step 7: Setup DPO trainer
    print("\n6️⃣ Setting up DPO trainer...")
    trainer = setup_dpo_trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
        config=config
    )
    print("   ✅ DPO trainer created successfully")
    
    # Step 8: Save training information
    print("\n7️⃣ Saving training information...")
    save_training_info(
        output_dir="./example_dpo_model",
        model_name=model_name,
        dataset_size=len(dataset),
        config=config,
        additional_info={
            "description": "Example DPO training using utilities",
            "dataset_type": "sample_preference_data"
        }
    )
    
    # Step 9: Optional: Start training (commented out for demo)
    print("\n8️⃣ Training setup complete!")
    print("   To start training, uncomment the following line:")
    print("   trainer.train()")
    
    # Uncomment the line below to actually run training
    # trainer.train()
    
    print("\n✅ Example usage completed!")
    print("\n💡 Key benefits of using utilities:")
    print("   - Reduced boilerplate code")
    print("   - Consistent configuration patterns")
    print("   - Easy dataset validation")
    print("   - Reusable components")

if __name__ == "__main__":
    main() 