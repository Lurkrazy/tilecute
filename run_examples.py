#!/usr/bin/env python3
"""
Example runner script for tilecute repository.

This script demonstrates how to run different examples and explains what each one does.
Use this to understand the progression from simple to advanced GPU programming concepts.
"""

import os
import sys
import subprocess
import time

def run_example(script_name, description, difficulty="Beginner"):
    """Run a Python example script and measure execution time."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Difficulty: {difficulty}")
    print(f"Description: {description}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"❌ Error: {script_name} not found!")
        return False
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=120)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ Success! Execution time: {end_time - start_time:.2f}s")
            print("\nOutput:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed with exit code: {result.returncode}")
            print("\nError output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout: Script took too long to execute")
        return False
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def check_prerequisites():
    """Check if the required dependencies are installed."""
    print("Checking prerequisites...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import cutlass.cute as cute
        print("✅ cuteDSL installed")
    except ImportError:
        print("❌ cuteDSL not installed")
        return False
    
    try:
        import tilelang
        print("✅ Tilelang installed")
    except (ImportError, OSError) as e:
        print(f"⚠️  Tilelang issue: {e}")
        print("   This may be due to missing CUDA runtime libraries")
        print("   You can still study the code structure and concepts")
        
    return True  # Allow proceeding even with tilelang issues
    
    return True

def main():
    """Main function to run all examples in progression."""
    print("🚀 Tilecute Example Runner")
    print("This script will run the examples in order of complexity\n")
    
    # Check prerequisites first
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install missing dependencies:")
        print("pip install tilelang nvidia-cutlass-dsl")
        return
    
    examples = [
        {
            "script": "element_wise_add_kernel.py",
            "description": "Basic element-wise addition using vectorized operations",
            "difficulty": "Beginner"
        },
        {
            "script": "gemm_f16_nn_cute.py", 
            "description": "F16 GEMM using cuteDSL high-level abstractions",
            "difficulty": "Intermediate"
        },
        {
            "script": "gemm_f16_nn.py",
            "description": "F16 GEMM with manual memory management and optimization",
            "difficulty": "Advanced"
        },
        {
            "script": "matmul_fp8_nt_kernel_cute.py",
            "description": "FP8 matrix multiplication for modern GPU architectures",
            "difficulty": "Expert"
        }
    ]
    
    print(f"\nFound {len(examples)} examples to run:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['script']} ({example['difficulty']})")
    
    # Ask user what to run
    print("\nOptions:")
    print("1. Run all examples")
    print("2. Run specific example")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return
    
    if choice == "1":
        # Run all examples
        success_count = 0
        for example in examples:
            success = run_example(
                example["script"], 
                example["description"], 
                example["difficulty"]
            )
            if success:
                success_count += 1
            
            # Ask to continue after each example
            if example != examples[-1]:  # Not the last example
                try:
                    cont = input("\nPress Enter to continue to next example (or Ctrl+C to exit)...")
                except KeyboardInterrupt:
                    print("\n\nStopping execution...")
                    break
        
        print(f"\n🎯 Summary: {success_count}/{len(examples)} examples ran successfully")
        
    elif choice == "2":
        # Run specific example
        print("\nSelect example to run:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['script']} ({example['difficulty']})")
        
        try:
            example_num = int(input("Enter example number: ")) - 1
            if 0 <= example_num < len(examples):
                example = examples[example_num]
                run_example(
                    example["script"],
                    example["description"], 
                    example["difficulty"]
                )
            else:
                print("Invalid example number!")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or interrupted!")
    
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()