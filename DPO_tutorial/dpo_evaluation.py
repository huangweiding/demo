#!/usr/bin/env python3
"""
DPO Evaluation Tutorial

This script demonstrates how to evaluate DPO-trained models and compare them
with reference models. It includes various evaluation metrics and generation
comparisons.

Based on the TRL evaluation methods from:
- trl/trainer/dpo_trainer.py
- trl/scripts/dpo.py
"""

import os
import torch
import json
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from trl import DPOTrainer, DPOConfig

def create_evaluation_dataset():
    """Create a dataset for evaluating DPO models."""
    data = [
        {
            "prompt": "Explain the benefits of renewable energy.",
            "expected_aspects": ["environmental", "economic", "sustainability"]
        },
        {
            "prompt": "Write a short story about friendship.",
            "expected_aspects": ["emotional", "narrative", "positive"]
        },
        {
            "prompt": "Describe the process of making bread.",
            "expected_aspects": ["detailed", "step-by-step", "practical"]
        },
        {
            "prompt": "What are the key principles of effective communication?",
            "expected_aspects": ["comprehensive", "practical", "structured"]
        },
        {
            "prompt": "Explain the concept of machine learning.",
            "expected_aspects": ["clear", "technical", "accessible"]
        }
    ]
    return Dataset.from_list(data)

def load_model_and_tokenizer(model_path: str):
    """Load a trained model and tokenizer."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

def generate_response(model, tokenizer, prompt: str, max_length: int = 200) -> str:
    """Generate a response using the given model and prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    generation_config = GenerationConfig(
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
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

def evaluate_response_quality(response: str, expected_aspects: List[str]) -> Dict[str, Any]:
    """
    Evaluate the quality of a response based on expected aspects.
    This is a simple heuristic evaluation - in practice, you might use more sophisticated metrics.
    """
    evaluation = {
        "length": len(response.split()),
        "has_content": len(response.strip()) > 0,
        "aspects_covered": 0,
        "overall_score": 0
    }
    
    # Simple heuristics for aspect coverage
    response_lower = response.lower()
    
    if "environmental" in expected_aspects and any(word in response_lower for word in ["environment", "pollution", "clean", "green"]):
        evaluation["aspects_covered"] += 1
    
    if "economic" in expected_aspects and any(word in response_lower for word in ["cost", "money", "economy", "financial", "investment"]):
        evaluation["aspects_covered"] += 1
    
    if "detailed" in expected_aspects and len(response.split()) > 50:
        evaluation["aspects_covered"] += 1
    
    if "step-by-step" in expected_aspects and any(word in response_lower for word in ["first", "then", "next", "finally", "step"]):
        evaluation["aspects_covered"] += 1
    
    if "technical" in expected_aspects and any(word in response_lower for word in ["algorithm", "data", "model", "training", "prediction"]):
        evaluation["aspects_covered"] += 1
    
    # Calculate overall score
    evaluation["overall_score"] = evaluation["aspects_covered"] / len(expected_aspects)
    
    return evaluation

def compare_models(base_model_path: str, dpo_model_path: str, eval_dataset: Dataset):
    """Compare responses from base model and DPO-trained model."""
    
    print("🔄 Loading models for comparison...")
    
    # Load base model
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_path)
    if base_model is None:
        print("❌ Could not load base model")
        return
    
    # Load DPO model
    dpo_model, dpo_tokenizer = load_model_and_tokenizer(dpo_model_path)
    if dpo_model is None:
        print("❌ Could not load DPO model")
        return
    
    print("✅ Models loaded successfully")
    
    # Generate and compare responses
    comparison_results = []
    
    for i, example in enumerate(eval_dataset):
        prompt = example["prompt"]
        expected_aspects = example["expected_aspects"]
        
        print(f"\n📝 Example {i+1}: {prompt}")
        
        # Generate responses
        base_response = generate_response(base_model, base_tokenizer, prompt)
        dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt)
        
        # Evaluate responses
        base_eval = evaluate_response_quality(base_response, expected_aspects)
        dpo_eval = evaluate_response_quality(dpo_response, expected_aspects)
        
        # Store results
        result = {
            "prompt": prompt,
            "expected_aspects": expected_aspects,
            "base_response": base_response,
            "dpo_response": dpo_response,
            "base_evaluation": base_eval,
            "dpo_evaluation": dpo_eval,
            "improvement": dpo_eval["overall_score"] - base_eval["overall_score"]
        }
        comparison_results.append(result)
        
        # Print comparison
        print(f"   Base model response: {base_response[:100]}...")
        print(f"   DPO model response: {dpo_response[:100]}...")
        print(f"   Base score: {base_eval['overall_score']:.2f}")
        print(f"   DPO score: {dpo_eval['overall_score']:.2f}")
        print(f"   Improvement: {result['improvement']:.2f}")
    
    return comparison_results

def calculate_metrics(comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall metrics from comparison results."""
    metrics = {
        "total_examples": len(comparison_results),
        "improvements": [],
        "base_scores": [],
        "dpo_scores": [],
        "average_improvement": 0,
        "examples_with_improvement": 0
    }
    
    for result in comparison_results:
        metrics["improvements"].append(result["improvement"])
        metrics["base_scores"].append(result["base_evaluation"]["overall_score"])
        metrics["dpo_scores"].append(result["dpo_evaluation"]["overall_score"])
        
        if result["improvement"] > 0:
            metrics["examples_with_improvement"] += 1
    
    if metrics["improvements"]:
        metrics["average_improvement"] = sum(metrics["improvements"]) / len(metrics["improvements"])
        metrics["average_base_score"] = sum(metrics["base_scores"]) / len(metrics["base_scores"])
        metrics["average_dpo_score"] = sum(metrics["dpo_scores"]) / len(metrics["dpo_scores"])
    
    return metrics

def main():
    # Configuration
    base_model_path = "microsoft/DialoGPT-medium"  # Original model
    dpo_model_path = "./dpo_trained_model"  # DPO-trained model
    output_dir = "./dpo_evaluation_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting DPO Evaluation Tutorial")
    print(f"📦 Base Model: {base_model_path}")
    print(f"📦 DPO Model: {dpo_model_path}")
    print(f"📁 Output Directory: {output_dir}")
    
    # Check if DPO model exists
    if not os.path.exists(dpo_model_path):
        print(f"❌ DPO model not found at {dpo_model_path}")
        print("   Please run basic_dpo_training.py first to train a DPO model")
        return
    
    # Prepare evaluation dataset
    print("\n1️⃣ Preparing evaluation dataset...")
    eval_dataset = create_evaluation_dataset()
    print(f"   Evaluation dataset size: {len(eval_dataset)} examples")
    
    # Compare models
    print("\n2️⃣ Comparing base model vs DPO model...")
    comparison_results = compare_models(base_model_path, dpo_model_path, eval_dataset)
    
    if comparison_results is None:
        print("❌ Comparison failed")
        return
    
    # Calculate metrics
    print("\n3️⃣ Calculating evaluation metrics...")
    metrics = calculate_metrics(comparison_results)
    
    # Print results
    print("\n4️⃣ Evaluation Results:")
    print("=" * 50)
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Average base model score: {metrics.get('average_base_score', 0):.3f}")
    print(f"Average DPO model score: {metrics.get('average_dpo_score', 0):.3f}")
    print(f"Average improvement: {metrics['average_improvement']:.3f}")
    print(f"Examples with improvement: {metrics['examples_with_improvement']}/{metrics['total_examples']}")
    
    # Save detailed results
    print("\n5️⃣ Saving detailed results...")
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "metrics": metrics,
            "detailed_comparisons": comparison_results
        }, f, indent=2)
    
    print(f"   Detailed results saved to: {results_file}")
    
    # Print summary
    print("\n6️⃣ Summary:")
    if metrics['average_improvement'] > 0:
        print(f"   ✅ DPO training improved model performance by {metrics['average_improvement']:.3f} on average")
    else:
        print(f"   ⚠️ DPO training did not show improvement (average change: {metrics['average_improvement']:.3f})")
    
    print(f"   📊 {metrics['examples_with_improvement']}/{metrics['total_examples']} examples showed improvement")
    
    print("\n💡 Evaluation Tips:")
    print("   - Use more sophisticated evaluation metrics for production")
    print("   - Consider human evaluation for subjective tasks")
    print("   - Test on diverse prompts to ensure generalization")
    print("   - Monitor for potential regressions in other capabilities")
    
    print(f"\n✅ DPO evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 