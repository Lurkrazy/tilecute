# Tilecute

Tilecute is an experimental backend implementation for Tilelang using cuteDSL. This project demonstrates the capabilities of cuteDSL through a direct 1:1 mapping approach to Tilelang's semantics.

## 🚀 Quick Start

```bash
# Install dependencies
pip install tilelang nvidia-cutlass-dsl

# Run your first GPU kernel
python element_wise_add_kernel.py
```

👉 **New to GPU programming?** Start with our [**Quick Start Tutorial**](QUICK_START.md)

👉 **Want comprehensive documentation?** Read the [**Usage Guide**](USAGE_GUIDE.md)

## What You'll Learn

- **GPU Kernel Development**: Write high-performance CUDA kernels using cuteDSL
- **Modern GPU Programming**: TMA operations, warp specialization, memory barriers
- **Matrix Operations**: Optimized GEMM implementations for different GPU architectures
- **Memory Management**: Efficient data movement between GPU memory hierarchies

## Featured Examples

| Example | What It Shows | Best For |
|---------|---------------|----------|
| [`element_wise_add_kernel.py`](element_wise_add_kernel.py) | Basic vectorized operations | Learning cuteDSL basics |
| [`gemm_f16_nn_cute.py`](gemm_f16_nn_cute.py) | F16 GEMM with high-level abstractions | Understanding matrix multiplication |
| [`gemm_f16_nn.py`](gemm_f16_nn.py) | Manual GEMM with explicit memory management | Advanced optimization techniques |
| [`matmul_fp8_nt_kernel_cute.py`](matmul_fp8_nt_kernel_cute.py) | FP8 precision for modern GPUs | Cutting-edge features |

## Architecture Support

- ✅ **SM70+ (Volta)**: Basic tensor cores, fundamental operations
- ✅ **SM80+ (Ampere)**: Advanced tensor cores, async copy operations  
- ✅ **SM89+ (Ada Lovelace)**: FP8 data types, enhanced performance
- ✅ **SM90+ (Hopper)**: TMA operations, warp specialization, WGMMA

## Current Implementation Status

- [x] **Elementwise operations**: Vectorized load/store, custom layouts
- [x] **GEMM on Ampere**: LDGSTS, tensor cores, software pipelining
- [x] **GEMM on Hopper**: TMA, warp specialization, memory barriers, WGMMA
- [x] **Multiple precisions**: FP16, FP32, FP8 support
- [x] **Performance optimization**: Memory coalescing, shared memory banking

## Prerequisites

- **Hardware**: NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- **Software**: Python 3.8+, CUDA Toolkit 11.8+
- **Libraries**: PyTorch with CUDA support

## Repository Structure

```
tilecute/
├── 📚 Documentation & Guides
│   ├── README.md              # Project overview & quick start
│   ├── QUICK_START.md         # 5-minute tutorial
│   ├── USAGE_GUIDE.md         # Comprehensive documentation  
│   └── TROUBLESHOOTING.md     # Common issues & solutions
├── 🎯 Examples & Learning
│   ├── element_wise_add_kernel.py      # Basic operations (start here)
│   ├── gemm_f16_nn_cute.py            # GEMM with cuteDSL abstractions
│   ├── gemm_f16_nn.py                 # Manual GEMM optimization
│   ├── matmul_fp8_nt_kernel_cute.py   # FP8 for modern GPUs
│   └── custom_example.py              # Learn customization patterns
├── 🔧 Utilities & Tools
│   ├── diagnose.py            # Environment diagnostic tool
│   ├── run_examples.py        # Interactive example runner
│   └── utils/                 # Helper functions
└── 📦 Templates & Headers
    └── tl_templates/cuda/     # C++ CUDA implementation templates
```

## Getting Started

## Getting Started

### Quick Check
```bash
# First, check your environment
python diagnose.py
```

### Installation
```bash
pip install tilelang nvidia-cutlass-dsl
```

### Run Examples
```bash
# Interactive example runner
python run_examples.py

# Or run individual examples
python element_wise_add_kernel.py
python gemm_f16_nn_cute.py
python custom_example.py  # Learn customization
```

### Learning Path
1. 📖 Read the [Quick Start Tutorial](QUICK_START.md) (5 minutes)
2. 🔬 Run and modify the examples  
3. 📚 Study the [comprehensive guide](USAGE_GUIDE.md)
4. 🏗️ Build your own kernels using [custom_example.py](custom_example.py)
5. 🛠️ Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if needed


