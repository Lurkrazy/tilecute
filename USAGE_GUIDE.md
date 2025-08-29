# Tilecute Usage Guide

## Overview

Tilecute is an experimental backend implementation for Tilelang using cuteDSL (CUTLASS DSL). This repository demonstrates how to implement high-performance GPU kernels using cuteDSL through direct 1:1 mapping from Tilelang semantics.

## What You'll Learn

- How to write GPU kernels using cuteDSL
- Advanced GPU programming techniques (TMA, warp specialization, memory barriers)
- GEMM (General Matrix Multiply) implementations for different GPU architectures
- Element-wise operations with vectorized memory access patterns
- Modern CUDA programming patterns

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support
- Compute capability 7.0+ (for SM70 examples)
- Compute capability 8.0+ (for SM80/Ampere examples) 
- Compute capability 9.0+ (for SM90/Hopper examples)

### Software Requirements
- Python 3.8+
- CUDA Toolkit 11.8+
- PyTorch with CUDA support

## Installation

```bash
# Install required dependencies
pip install tilelang nvidia-cutlass-dsl

# Verify installation
python -c "import cutlass.cute as cute; print('cuteDSL installed successfully')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Repository Structure

```
tilecute/
├── README.md                    # Basic project information
├── USAGE_GUIDE.md              # This comprehensive guide
├── element_wise_add_kernel.py   # Simple element-wise addition example
├── gemm_f16_nn.py              # F16 GEMM with manual memory management
├── gemm_f16_nn_cute.py         # F16 GEMM using cuteDSL abstractions
├── matmul_fp8_nt_kernel.py     # FP8 matrix multiplication example
├── matmul_fp8_nt_kernel_cute.py # FP8 matmul with cuteDSL
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── cpasync.py              # Async copy operations
│   └── mbar.py                 # Memory barrier utilities
└── tl_templates/cuda/          # C++ CUDA template headers
    ├── common.h                # Common CUDA utilities
    ├── gemm_sm70.h            # GEMM for Volta architecture
    ├── gemm_sm80.h            # GEMM for Ampere architecture
    ├── gemm_sm89.h            # GEMM for Ada Lovelace
    ├── gemm_sm90.h            # GEMM for Hopper architecture
    ├── copy_sm90.h            # TMA operations for Hopper
    └── reduce.h               # Reduction operations
```

## Examples and How to Use Them

### 1. Element-wise Addition (`element_wise_add_kernel.py`)

This is the simplest example showing basic cuteDSL usage:

```bash
python element_wise_add_kernel.py
```

**What it demonstrates:**
- Basic cuteDSL kernel structure using `@cute.kernel` decorator
- Thread and block indexing with `cute.arch`
- Vectorized memory access (loading/storing 4 elements at once)
- Type conversions and memory alignment

**Key concepts:**
- `cute.Tensor` for GPU memory management
- `gA.iterator + offset` for pointer arithmetic
- `cute.make_tensor()` for creating tensor views
- `tC.store(tA.load() + tB.load())` for vectorized operations

### 2. F16 GEMM Examples

#### Manual Implementation (`gemm_f16_nn.py`)
Shows low-level GEMM implementation with explicit memory management:

```bash
python gemm_f16_nn.py
```

**What it demonstrates:**
- Manual shared memory management using `cutlass.utils.SmemAllocator`
- Async copy operations with `cp_async_shared_global`
- Pipeline management with `cp_async_commit_group()` and `cp_async_wait_group()`
- Tensor core operations through `cute.gemm()`

#### cuteDSL Implementation (`gemm_f16_nn_cute.py`)
Shows the same GEMM using cuteDSL abstractions:

```bash
python gemm_f16_nn_cute.py
```

**What it demonstrates:**
- High-level cuteDSL abstractions
- Automatic layout management with `cute.local_tile()`
- Simplified copy operations using `cute.copy()`
- Tiled MMA (Matrix Multiply Accumulate) operations

### 3. FP8 Matrix Multiplication

For modern GPU architectures that support FP8:

```bash
python matmul_fp8_nt_kernel_cute.py
```

**What it demonstrates:**
- FP8 data type usage for memory efficiency
- Advanced layout transformations
- Cross-architecture compatibility

## Understanding the Code Structure

### Key cuteDSL Concepts

1. **Kernels**: Functions decorated with `@cute.kernel` that run on GPU
2. **Tensors**: `cute.Tensor` objects that represent GPU memory with layout information
3. **Layouts**: Define how data is organized in memory (row-major, column-major, swizzled)
4. **Tiled Operations**: Break large operations into smaller tiles for efficiency
5. **Copy Operations**: Move data between global memory, shared memory, and registers

### Architecture-Specific Features

- **SM70 (Volta)**: Basic tensor cores
- **SM80 (Ampere)**: Enhanced tensor cores, async copy
- **SM89 (Ada Lovelace)**: FP8 support
- **SM90 (Hopper)**: TMA (Tensor Memory Accelerator), warp specialization

## Performance Considerations

### Memory Hierarchy
1. **Global Memory**: Slow but large, data starts here
2. **Shared Memory**: Fast, shared within thread block
3. **Registers**: Fastest, private to each thread

### Optimization Techniques
1. **Memory Coalescing**: Access memory in patterns that maximize bandwidth
2. **Shared Memory Banking**: Avoid bank conflicts in shared memory access
3. **Pipeline Overlapping**: Overlap computation with memory transfers
4. **Tensor Cores**: Use specialized hardware for matrix operations

## Customizing for Your Use Case

### 1. Modifying Tile Sizes
```python
cta_tiler = (128, 128, 32)  # Change M, N, K tile dimensions
```

### 2. Changing Data Types
```python
ab_dtype = cutlass.Float16  # Input matrices
c_dtype = cutlass.Float16   # Output matrix
acc_dtype = cutlass.Float16 # Accumulator type
```

### 3. Grid and Block Configuration
```python
kernel.launch(grid=(4, 8, 1),    # Grid dimensions
              block=(128, 1, 1),  # Block dimensions
              smem=32768)          # Shared memory size
```

## Debugging and Profiling

### Common Issues
1. **CUDA out of memory**: Reduce tile sizes or batch size
2. **Compilation errors**: Check GPU compute capability compatibility
3. **Numerical errors**: Verify data types and accumulator precision

### Profiling Tools
```bash
# Use NSight Compute for detailed kernel analysis
ncu --set full python your_kernel.py

# Use NSight Systems for timeline analysis
nsys profile python your_kernel.py
```

## Learning Path

1. **Start with**: `element_wise_add_kernel.py` - Learn basic concepts
2. **Move to**: `gemm_f16_nn_cute.py` - Understand tiled operations
3. **Advanced**: Study architecture-specific optimizations in `tl_templates/`
4. **Expert**: Implement your own kernels using the patterns shown

## Contributing

This repository serves as educational material. To contribute:

1. Add new kernel examples with clear documentation
2. Improve existing examples with better comments
3. Add architecture-specific optimizations
4. Create tutorials for specific use cases

## Further Resources

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [cuteDSL Programming Guide](https://nvidia.github.io/cutlass/media/docs/cute/00_quickstart.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Tensor Core Programming](https://docs.nvidia.com/cuda/cublas/index.html#using-tensor-cores)

## Troubleshooting

### Installation Issues
```bash
# If PyTorch CUDA support is missing
pip install torch --index-url https://download.pytorch.org/whl/cu118

# If cutlass-dsl installation fails
pip install --no-cache-dir nvidia-cutlass-dsl
```

### Runtime Issues
```bash
# Check GPU compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"

# Verify CUDA installation
nvcc --version
```

### Performance Issues
- Use NSight Compute to identify bottlenecks
- Check memory access patterns
- Verify occupancy metrics
- Profile different tile sizes