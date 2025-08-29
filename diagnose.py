#!/usr/bin/env python3
"""
Tilecute Environment Diagnostic Script

This script checks your environment and provides recommendations for using tilecute.
Run this before trying the examples to identify potential issues.
"""

import torch
import sys
import platform

def main():
    print("=== Tilecute Environment Diagnostic ===")
    print()
    
    # System Information
    print("🖥️  System Information:")
    print(f"   Operating System: {platform.system()} {platform.release()}")
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    print()
    
    # PyTorch Information
    print("🔥 PyTorch Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"     Total memory: {gpu_props.total_memory / 1e9:.1f} GB")
    else:
        print("   No CUDA GPU detected")
    print()
    
    # Dependencies Check
    print("📦 Dependencies Check:")
    
    # Check cuteDSL
    try:
        import cutlass.cute as cute
        import cutlass
        print("   ✅ cuteDSL (nvidia-cutlass-dsl) available")
        print(f"      CUTLASS version: {getattr(cutlass, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"   ❌ cuteDSL not available: {e}")
    
    # Check tilelang
    try:
        import tilelang
        print("   ✅ tilelang available")
    except (ImportError, OSError) as e:
        print(f"   ⚠️  tilelang issue: {str(e)[:100]}...")
        if "libcuda.so" in str(e):
            print("      This is likely due to missing CUDA runtime libraries")
    
    # Check additional dependencies
    optional_deps = {
        'numpy': 'numpy',
        'tqdm': 'tqdm', 
        'cuda-python': 'cuda.bindings.driver'
    }
    
    for name, import_name in optional_deps.items():
        try:
            __import__(import_name)
            print(f"   ✅ {name} available")
        except ImportError:
            print(f"   ⚠️  {name} not available (optional)")
    
    print()
    
    # Compatibility Analysis
    print("🔍 Compatibility Analysis:")
    
    issues = []
    recommendations = []
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        issues.append("No CUDA GPU detected")
        recommendations.append("Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        recommendations.append("Verify NVIDIA drivers are installed: nvidia-smi")
    
    # Check compute capability
    if torch.cuda.is_available():
        min_cc = min(torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count()))
        if min_cc < 7:
            issues.append(f"GPU compute capability {min_cc}.x < 7.0")
            recommendations.append("Some examples may not work on older GPUs")
            recommendations.append("Try element_wise_add_kernel.py first")
        
        # Check GPU memory
        min_memory = min(torch.cuda.get_device_properties(i).total_memory / 1e9 
                        for i in range(torch.cuda.device_count()))
        if min_memory < 4:
            issues.append(f"Limited GPU memory: {min_memory:.1f} GB")
            recommendations.append("Use smaller matrix sizes in examples")
            recommendations.append("Reduce tile sizes: cta_tiler = (64, 64, 16)")
    
    # Check cuteDSL
    try:
        import cutlass.cute as cute
    except ImportError:
        issues.append("cuteDSL not available")
        recommendations.append("Install cuteDSL: pip install nvidia-cutlass-dsl")
    
    # Display results
    if issues:
        print("   ⚠️  Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ No major issues detected")
    
    print()
    
    if recommendations:
        print("💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    else:
        print("🎉 Your environment looks good for running tilecute examples!")
    
    print()
    
    # Usage suggestions
    print("🚀 Next Steps:")
    
    if torch.cuda.is_available():
        try:
            import cutlass.cute as cute
            print("   1. Run: python element_wise_add_kernel.py")
            if torch.cuda.get_device_capability()[0] >= 7:
                print("   2. Try: python gemm_f16_nn_cute.py")
                if torch.cuda.get_device_capability()[0] >= 8:
                    print("   3. Advanced: python matmul_fp8_nt_kernel_cute.py")
            print("   4. Use: python run_examples.py (interactive runner)")
        except ImportError:
            print("   1. Install cuteDSL first")
            print("   2. Then run the examples")
    else:
        print("   1. Study the code structure and concepts")
        print("   2. Read USAGE_GUIDE.md for theoretical understanding")
        print("   3. Set up CUDA environment for hands-on practice")
    
    print("   📚 Read USAGE_GUIDE.md for detailed documentation")
    print("   🛠️  Check TROUBLESHOOTING.md if you encounter issues")

if __name__ == "__main__":
    main()