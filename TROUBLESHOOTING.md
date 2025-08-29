# Troubleshooting Guide

This guide helps you solve common issues when using the tilecute repository.

## Installation Issues

### Problem: `pip install nvidia-cutlass-dsl` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement nvidia-cutlass-dsl
```

**Solutions:**
1. **Update pip**: `pip install --upgrade pip`
2. **Check Python version**: Requires Python 3.8+
   ```bash
   python --version
   ```
3. **Try with index URL**:
   ```bash
   pip install nvidia-cutlass-dsl --index-url https://pypi.org/simple/
   ```
4. **Install from conda-forge**:
   ```bash
   conda install -c conda-forge nvidia-cutlass-dsl
   ```

### Problem: PyTorch CUDA not available

**Symptoms:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**
1. **Install PyTorch with CUDA**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
2. **Check CUDA installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```
3. **Verify GPU detection**:
   ```python
   import torch
   print(torch.cuda.device_count())
   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
   ```

## Runtime Issues

### Problem: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
1. **Reduce matrix sizes** in examples:
   ```python
   # In gemm_f16_nn_cute.py, change:
   L, M, N, K = 1, 256, 512, 384  # Smaller matrices
   ```

2. **Reduce tile sizes**:
   ```python
   cta_tiler = (64, 64, 16)  # Smaller tiles
   ```

3. **Clear GPU memory**:
   ```python
   torch.cuda.empty_cache()
   ```

4. **Check GPU memory usage**:
   ```bash
   nvidia-smi
   ```

### Problem: "Compute capability not supported"

**Symptoms:**
```
RuntimeError: Your GPU does not support this operation (compute capability < 7.0)
```

**Solutions:**
1. **Check your GPU's compute capability**:
   ```python
   import torch
   if torch.cuda.is_available():
       print(f"Compute capability: {torch.cuda.get_device_capability()}")
   ```

2. **Use appropriate examples**:
   - SM70+: `element_wise_add_kernel.py`, `gemm_f16_nn_cute.py`
   - SM80+: Advanced GEMM examples
   - SM89+: FP8 examples

3. **Modify examples for older GPUs**:
   ```python
   # Change data types for older GPUs
   ab_dtype = cutlass.Float32  # Instead of Float16
   ```

### Problem: Compilation errors

**Symptoms:**
```
cutlass._mlir.CompilationError: Failed to compile kernel
```

**Solutions:**
1. **Check CUDA toolkit version**:
   ```bash
   nvcc --version
   # Should be 11.8 or newer
   ```

2. **Verify imports**:
   ```python
   import cutlass.cute as cute
   import cutlass
   ```

3. **Simplify the kernel**:
   - Start with `element_wise_add_kernel.py`
   - Gradually add complexity

4. **Check for typos** in tensor operations and decorators

## Performance Issues

### Problem: Kernel runs but is very slow

**Diagnosis:**
```python
# Add timing to your kernels
import time
start = time.time()
your_kernel()
torch.cuda.synchronize()
print(f"Execution time: {time.time() - start:.4f}s")
```

**Solutions:**
1. **Optimize memory access patterns**:
   - Ensure coalesced memory access
   - Use appropriate tile sizes

2. **Check occupancy**:
   ```bash
   # Use Nsight Compute for detailed analysis
   ncu --set full python your_script.py
   ```

3. **Tune block sizes**:
   ```python
   # Try different block sizes
   kernel.launch(grid=(4, 8, 1), block=(256, 1, 1))  # vs (128, 1, 1)
   ```

4. **Profile memory usage**:
   ```bash
   # Use Nsight Systems
   nsys profile python your_script.py
   ```

### Problem: Incorrect numerical results

**Symptoms:**
```
AssertionError: Tensor-likes are not close
```

**Solutions:**
1. **Check data types**:
   ```python
   print(f"Input: {a.dtype}, Output: {c.dtype}")
   # Ensure consistent precision
   ```

2. **Adjust tolerance**:
   ```python
   torch.testing.assert_close(gpu_result, cpu_result, atol=1e-2, rtol=1e-2)
   ```

3. **Debug with smaller matrices**:
   ```python
   # Use small matrices for debugging
   M, N, K = 16, 16, 16
   ```

4. **Print intermediate values**:
   ```python
   print(f"A sample: {a[0, :5]}")
   print(f"B sample: {b[0, :5]}")
   print(f"C sample: {c[0, :5]}")
   ```

## Environment Issues

### Problem: ImportError for cutlass modules

**Symptoms:**
```
ImportError: No module named 'cutlass.cute'
```

**Solutions:**
1. **Reinstall with force**:
   ```bash
   pip uninstall nvidia-cutlass-dsl
   pip install --no-cache-dir nvidia-cutlass-dsl
   ```

2. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   ```

3. **Use virtual environment**:
   ```bash
   python -m venv tilecute_env
   source tilecute_env/bin/activate  # On Windows: tilecute_env\Scripts\activate
   pip install tilelang nvidia-cutlass-dsl
   ```

### Problem: "No CUDA-capable device is detected"

**Solutions:**
1. **Check GPU is recognized**:
   ```bash
   lspci | grep -i nvidia
   nvidia-smi
   ```

2. **Install NVIDIA drivers**:
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-driver-535
   
   # Check installation
   nvidia-smi
   ```

3. **Restart after driver installation**

## Development Issues

### Problem: Modifying examples doesn't work

**Common mistakes:**
1. **Incorrect tensor shapes**: Ensure shapes match your operations
2. **Wrong grid/block sizes**: Calculate based on your problem size
3. **Type mismatches**: Ensure consistent data types throughout

**Debugging approach:**
1. **Start simple**: Modify one parameter at a time
2. **Add print statements**: Debug intermediate values
3. **Use smaller problems**: Test with tiny matrices first
4. **Check documentation**: Refer to cuteDSL and CUTLASS docs

### Problem: Creating custom kernels fails

**Best practices:**
1. **Copy existing examples**: Start from working code
2. **Understand memory layouts**: Learn about row-major vs column-major
3. **Learn cuteDSL gradually**: Start with simple operations
4. **Read the templates**: Study `tl_templates/cuda/` for patterns

## Getting Help

### Resources
1. **CUTLASS Documentation**: https://github.com/NVIDIA/cutlass
2. **cuteDSL Guide**: https://nvidia.github.io/cutlass/media/docs/cute/
3. **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. **Repository Issues**: Check existing issues for similar problems

### Reporting Issues
When reporting issues, include:
1. **System info**: GPU model, CUDA version, Python version
2. **Error messages**: Full error traceback
3. **Minimal example**: Simplest code that reproduces the issue
4. **Environment**: Virtual environment, package versions

### Community
- Join NVIDIA Developer forums
- Check Stack Overflow for CUDA/CUTLASS questions
- Read NVIDIA technical blogs for best practices

## Quick Diagnostic Script

Save this as `diagnose.py` to check your environment:

```python
#!/usr/bin/env python3
import torch
import sys

print("=== Tilecute Environment Diagnostic ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No CUDA GPU detected")

try:
    import cutlass.cute as cute
    print("✅ cuteDSL available")
except ImportError as e:
    print(f"❌ cuteDSL not available: {e}")

try:
    import tilelang
    print("✅ tilelang available")
except ImportError as e:
    print(f"❌ tilelang not available: {e}")

print("\nRecommendations:")
if not torch.cuda.is_available():
    print("- Install CUDA-enabled PyTorch")
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
    print("- GPU compute capability < 7.0, some examples may not work")
if torch.cuda.is_available():
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mem_gb < 8:
        print("- GPU has limited memory, use smaller matrix sizes")
```

Run with: `python diagnose.py`