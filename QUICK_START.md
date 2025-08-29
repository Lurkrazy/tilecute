# Quick Start Tutorial

## Getting Started in 5 Minutes

This tutorial will get you running GPU kernels with tilecute quickly.

### Step 1: Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/Lurkrazy/tilecute.git
cd tilecute

# Install dependencies
pip install tilelang nvidia-cutlass-dsl

# Verify your setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Run Your First Kernel (1 minute)

```bash
# Run the simplest example - element-wise addition
python element_wise_add_kernel.py
```

**Expected output:**
```
pass!
```

This kernel adds two 512x1024 float32 matrices and outputs the result as float16.

### Step 3: Understand What Happened (2 minutes)

The kernel you just ran:

1. **Created tensors**: Two input matrices (A, B) and one output matrix (C)
2. **Launched GPU kernel**: 4x4 grid with 128 threads per block
3. **Performed vectorized addition**: Each thread processed 4 elements at once
4. **Verified correctness**: Compared GPU result with CPU reference

**Key code structure:**
```python
@cute.kernel
def elementwise_add_kernel(gA, gB, gC):
    # Get thread coordinates
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    
    # Process 64 iterations, 4 elements each
    for i in cutlass.range_constexpr(64):
        # Calculate memory offsets
        offset = compute_offset(bidx, bidy, tidx, i)
        
        # Create tensor views (4 elements each)
        tA = cute.make_tensor(gA.iterator + offset, (4))
        tB = cute.make_tensor(gB.iterator + offset, (4))
        tC = cute.make_tensor(gC.iterator + offset, (4))
        
        # Vectorized add and store
        tC.store((tA.load() + tB.load()).to(cute.Float16))
```

## Next Steps

### Run Matrix Multiplication (GEMM)

```bash
# Run F16 GEMM with cuteDSL abstractions
python gemm_f16_nn_cute.py
```

This example demonstrates:
- Matrix multiplication using tensor cores
- Advanced memory management
- Performance optimization techniques

**Expected output:**
```
Compiling kernel with cute.compile ...
Compilation time: X.XXXX seconds
max diff tensor(X.XXXX)
FP16 GEMM kernel test passed!
Executing GEMM kernel...
Kernel execution time: X.XXXX ms
```

### Explore Advanced Examples

```bash
# FP8 matrix multiplication (requires modern GPU)
python matmul_fp8_nt_kernel_cute.py
```

## Understanding Performance

The kernels include timing and verification:

```python
# Compilation timing
start_time = time.time()
compiled_kernel = cute.compile(kernel_function, *args)
print(f"Compilation time: {time.time() - start_time:.4f} seconds")

# Execution timing using CUDA events
for _ in range(iterations):
    compiled_kernel(*args)
    
# Correctness verification
torch.testing.assert_close(gpu_result, cpu_reference, atol=1e-3, rtol=1e-3)
```

## Customization Examples

### Change Matrix Sizes

In `gemm_f16_nn_cute.py`, modify:
```python
L, M, N, K = 1, 512, 1024, 768  # Change these values
```

### Change Data Types

```python
ab_dtype = cutlass.Float16    # Input matrices
c_dtype = cutlass.Float16     # Output matrix
acc_dtype = cutlass.Float16   # Accumulator
```

### Adjust Tile Sizes

```python
cta_tiler = (128, 128, 32)  # (M_tile, N_tile, K_tile)
```

## Common Issues and Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce matrix sizes or tile dimensions
```python
L, M, N, K = 1, 256, 512, 384  # Smaller matrices
cta_tiler = (64, 64, 16)       # Smaller tiles
```

### Issue: "No CUDA-capable device"
**Solution**: This repository requires a CUDA-capable GPU. For learning without GPU:
1. Study the code structure and algorithms
2. Understand the cuteDSL concepts
3. Use the examples as templates for when you have GPU access

### Issue: Import errors
**Solution**: Verify installation
```bash
pip install --upgrade tilelang nvidia-cutlass-dsl torch
```

## What Each File Does

| File | Purpose | Difficulty | GPU Requirements |
|------|---------|------------|------------------|
| `element_wise_add_kernel.py` | Basic vectorized operations | Beginner | Any CUDA GPU |
| `gemm_f16_nn_cute.py` | GEMM with cuteDSL | Intermediate | SM70+ |
| `gemm_f16_nn.py` | Manual GEMM implementation | Advanced | SM70+ |
| `matmul_fp8_nt_kernel_cute.py` | FP8 matrix multiply | Advanced | SM89+ |

## Learning Progression

1. **Week 1**: Understand `element_wise_add_kernel.py`
   - Learn basic cuteDSL syntax
   - Understand GPU thread organization
   - Practice modifying simple kernels

2. **Week 2**: Master `gemm_f16_nn_cute.py`
   - Learn matrix multiplication concepts
   - Understand tiling strategies
   - Experiment with different configurations

3. **Week 3+**: Explore advanced examples
   - Study architecture-specific optimizations
   - Learn about memory hierarchy optimization
   - Create your own kernels

## Getting Help

1. **Code Comments**: Each example has detailed comments explaining the concepts
2. **Documentation**: Read `USAGE_GUIDE.md` for comprehensive information
3. **CUTLASS Docs**: Visit [NVIDIA CUTLASS documentation](https://github.com/NVIDIA/cutlass)
4. **Issues**: Check the repository issues for common problems

Happy GPU programming! 🚀