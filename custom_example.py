"""
Custom Example: How to modify tilecute kernels for your own use case

This example shows how to:
1. Start with an existing kernel
2. Modify it for different data types
3. Change matrix dimensions  
4. Adjust performance parameters
5. Add custom functionality

Based on element_wise_add_kernel.py but with extensive customization examples.
"""

import torch
from functools import partial
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ============================================================================
# Configuration Section - Modify these values for your use case
# ============================================================================

# Matrix dimensions - adjust based on your needs and GPU memory
MATRIX_HEIGHT = 256  # Reduce if you get out of memory
MATRIX_WIDTH = 512

# Data types - experiment with different precisions
INPUT_DTYPE = torch.float32   # Input matrices A and B
OUTPUT_DTYPE = torch.float16  # Output matrix C

# Thread block configuration - tune for your GPU
THREADS_PER_BLOCK = 128       # Common values: 64, 128, 256, 512
ELEMENTS_PER_THREAD = 4       # Vectorization factor (2, 4, 8, 16)

# Grid configuration - calculated automatically but can be overridden
GRID_X = 4  # Number of blocks in X dimension
GRID_Y = 4  # Number of blocks in Y dimension

# Performance options
ENABLE_PROFILING = True       # Measure execution time
VERIFY_CORRECTNESS = True     # Compare with CPU reference

# ============================================================================
# Kernel Implementation
# ============================================================================

@cute.kernel
def custom_elementwise_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor, 
    gC: cute.Tensor,
    operation: int = 0  # 0=add, 1=multiply, 2=max, 3=custom
):
    """
    Customizable element-wise operation kernel.
    
    Args:
        gA: Input tensor A
        gB: Input tensor B  
        gC: Output tensor C
        operation: Operation type (0=add, 1=multiply, 2=max, 3=custom)
    """
    # Get thread and block indices
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    # Calculate how many iterations each thread needs to handle
    total_elements = MATRIX_HEIGHT * MATRIX_WIDTH
    elements_per_block = bdimx * ELEMENTS_PER_THREAD
    iterations_per_thread = 64  # This can be calculated dynamically
    
    # Main processing loop
    for i in cutlass.range_constexpr(iterations_per_thread):
        # Calculate memory offset for this thread and iteration
        # This follows the same pattern as the original but is more readable
        offset_A = (bidy * (MATRIX_WIDTH * MATRIX_HEIGHT // GRID_Y) + 
                   i * (MATRIX_WIDTH * GRID_X // iterations_per_thread) + 
                   (tidx >> 6) * MATRIX_WIDTH + 
                   bidx * (MATRIX_WIDTH // GRID_X) + 
                   (tidx & 63) * ELEMENTS_PER_THREAD)
        
        offset_B = offset_A  # Same pattern for B
        offset_C = offset_A  # Same pattern for C

        # Ensure alignment for vectorized access
        offset_A = cute.assume(offset_A, divby=ELEMENTS_PER_THREAD)
        offset_B = cute.assume(offset_B, divby=ELEMENTS_PER_THREAD)  
        offset_C = cute.assume(offset_C, divby=ELEMENTS_PER_THREAD)

        # Create tensor views for vectorized operations
        tA = cute.make_tensor(gA.iterator + offset_A, (ELEMENTS_PER_THREAD,))
        tB = cute.make_tensor(gB.iterator + offset_B, (ELEMENTS_PER_THREAD,))
        tC = cute.make_tensor(gC.iterator + offset_C, (ELEMENTS_PER_THREAD,))

        # Load data from global memory
        data_A = tA.load()
        data_B = tB.load()

        # Perform the selected operation
        if operation == 0:
            # Element-wise addition
            result = data_A + data_B
        elif operation == 1:
            # Element-wise multiplication  
            result = data_A * data_B
        elif operation == 2:
            # Element-wise maximum
            result = cute.maximum(data_A, data_B)
        else:
            # Custom operation: weighted sum with nonlinearity
            # result = tanh(0.5 * A + 0.3 * B)
            result = cute.tanh(0.5 * data_A + 0.3 * data_B)

        # Store result with type conversion
        if OUTPUT_DTYPE == torch.float16:
            tC.store(result.to(cute.Float16))
        elif OUTPUT_DTYPE == torch.float32:
            tC.store(result.to(cute.Float32))
        else:
            tC.store(result)  # Keep original type

# ============================================================================
# Helper Functions
# ============================================================================

def create_test_tensors():
    """Create test tensors with the specified configuration."""
    print(f"Creating tensors: {MATRIX_HEIGHT}x{MATRIX_WIDTH}")
    print(f"Input dtype: {INPUT_DTYPE}, Output dtype: {OUTPUT_DTYPE}")
    
    # Create input tensors
    a = torch.randn(MATRIX_HEIGHT, MATRIX_WIDTH, device="cuda", dtype=INPUT_DTYPE)
    b = torch.randn(MATRIX_HEIGHT, MATRIX_WIDTH, device="cuda", dtype=INPUT_DTYPE)
    c = torch.zeros(MATRIX_HEIGHT, MATRIX_WIDTH, device="cuda", dtype=OUTPUT_DTYPE)

    # Convert to cuteDSL tensors with alignment
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16) 
    c_ = from_dlpack(c, assumed_align=16)

    return a, b, c, a_, b_, c_

def verify_result(a, b, c, operation):
    """Verify the GPU result against CPU reference."""
    if not VERIFY_CORRECTNESS:
        return True
        
    print("Verifying correctness...")
    
    # Compute CPU reference
    if operation == 0:
        cpu_result = (a + b).to(OUTPUT_DTYPE)
    elif operation == 1:
        cpu_result = (a * b).to(OUTPUT_DTYPE)
    elif operation == 2:
        cpu_result = torch.maximum(a, b).to(OUTPUT_DTYPE)
    else:
        cpu_result = torch.tanh(0.5 * a + 0.3 * b).to(OUTPUT_DTYPE)
    
    # Compare results
    try:
        torch.testing.assert_close(c.cpu(), cpu_result.cpu(), atol=1e-3, rtol=1e-3)
        print("✅ Verification passed!")
        return True
    except AssertionError as e:
        print(f"❌ Verification failed: {e}")
        print(f"Max difference: {torch.max(torch.abs(c.cpu() - cpu_result.cpu()))}")
        return False

# ============================================================================
# JIT Compilation and Execution
# ============================================================================

@cute.jit
def run_custom_operation(mA, mB, mC, operation=0):
    """JIT-compiled function to run the custom kernel."""
    kernel = custom_elementwise_kernel(mA, mB, mC, operation)
    kernel.launch(grid=(GRID_X, GRID_Y, 1),
                  block=(THREADS_PER_BLOCK, 1, 1))

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function demonstrating different operations and configurations."""
    print("🔧 Custom Tilecute Example")
    print("=" * 50)
    
    # Print configuration
    print("Configuration:")
    print(f"  Matrix size: {MATRIX_HEIGHT} x {MATRIX_WIDTH}")
    print(f"  Input dtype: {INPUT_DTYPE}")
    print(f"  Output dtype: {OUTPUT_DTYPE}")
    print(f"  Threads per block: {THREADS_PER_BLOCK}")
    print(f"  Elements per thread: {ELEMENTS_PER_THREAD}")
    print(f"  Grid size: {GRID_X} x {GRID_Y}")
    
    # Create test data
    a, b, c, a_, b_, c_ = create_test_tensors()
    
    # Test different operations
    operations = [
        (0, "Addition (A + B)"),
        (1, "Multiplication (A * B)"), 
        (2, "Element-wise Maximum"),
        (3, "Custom (tanh(0.5*A + 0.3*B))")
    ]
    
    for op_code, op_name in operations:
        print(f"\n🧮 Testing: {op_name}")
        
        # Reset output tensor
        c.zero_()
        
        # Compile kernel (done once per operation)
        if ENABLE_PROFILING:
            import time
            start_time = time.time()
            
        compiled_kernel = cute.compile(run_custom_operation, a_, b_, c_, op_code)
        
        if ENABLE_PROFILING:
            compile_time = time.time() - start_time
            print(f"  Compilation time: {compile_time:.4f}s")
        
        # Execute kernel
        if ENABLE_PROFILING:
            start_time = time.time()
            
        compiled_kernel(a_, b_, c_, op_code)
        torch.cuda.synchronize()  # Ensure completion
        
        if ENABLE_PROFILING:
            exec_time = time.time() - start_time
            print(f"  Execution time: {exec_time:.4f}s")
        
        # Verify result
        verify_result(a, b, c, op_code)

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This example requires a CUDA GPU.")
        print("You can still study the code to understand the concepts!")
    else:
        main()