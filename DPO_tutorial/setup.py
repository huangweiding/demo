#!/usr/bin/env python3
"""
DPO Tutorial Setup Script

This script helps users set up their environment for DPO training by:
1. Checking system requirements
2. Installing dependencies
3. Downloading sample models
4. Creating necessary directories
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_gpu():
    """Check GPU availability."""
    print("\n🖥️ Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️ No GPU detected. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed. GPU check skipped.")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Read requirements from requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories for the tutorial."""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "datasets", 
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    return True

def download_sample_model():
    """Download a small sample model for testing."""
    print("\n🤖 Downloading sample model...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "microsoft/DialoGPT-medium"
        print(f"   Downloading {model_name}...")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained("models/sample_model")
        
        # Download model (this will take some time)
        print("   Downloading model weights (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained("models/sample_model")
        
        print("✅ Sample model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to download sample model: {e}")
        print("   You can still use the tutorial with other models")
        return False

def test_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing imports...")
    
    required_packages = [
        "torch",
        "transformers", 
        "trl",
        "datasets",
        "accelerate",
        "peft"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All required packages imported successfully")
        return True

def create_sample_script():
    """Create a sample script to test the setup."""
    print("\n📝 Creating sample test script...")
    
    sample_script = '''#!/usr/bin/env python3
"""
Sample DPO Test Script

This script tests if your DPO setup is working correctly.
"""

from dpo_utils import create_sample_preference_data, validate_preference_dataset

def test_dpo_setup():
    print("🧪 Testing DPO setup...")
    
    # Create sample dataset
    dataset = create_sample_preference_data()
    print(f"✅ Created dataset with {len(dataset)} examples")
    
    # Validate dataset
    is_valid = validate_preference_dataset(dataset)
    if is_valid:
        print("✅ Dataset validation passed")
        print("✅ DPO setup is working correctly!")
    else:
        print("❌ Dataset validation failed")
        print("❌ DPO setup has issues")

if __name__ == "__main__":
    test_dpo_setup()
'''
    
    with open("test_dpo_setup.py", "w") as f:
        f.write(sample_script)
    
    print("✅ Created test_dpo_setup.py")

def main():
    """Main setup function."""
    print("🚀 DPO Tutorial Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        return False
    
    gpu_available = check_gpu()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed at directory creation")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed at import testing")
        return False
    
    # Download sample model (optional)
    download_sample_model()
    
    # Create sample script
    create_sample_script()
    
    # Print success message
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Run: python test_dpo_setup.py")
    print("   2. Run: python basic_dpo_training.py")
    print("   3. Explore other tutorial scripts")
    
    if gpu_available:
        print("\n💡 GPU detected - training will be faster!")
    else:
        print("\n💡 No GPU detected - consider using smaller models for faster training")
    
    print("\n📚 Tutorial files:")
    print("   - basic_dpo_training.py: Basic DPO training")
    print("   - lora_dpo_training.py: DPO with LoRA")
    print("   - custom_loss_dpo.py: Different loss functions")
    print("   - dpo_evaluation.py: Model evaluation")
    print("   - dpo_utils.py: Utility functions")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 