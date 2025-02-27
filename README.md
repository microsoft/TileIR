<div align="center">

# TileIR

</div>

TileIR (**tile-ir**) is a concise domain-specific IR designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), TileIR allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

## Tested Devices
Although TileIR aims to be portable across a range of devices, it has been specifically tested and validated on the following hardware: for NVIDIA GPUs, this includes the H100 (with Auto TMA/WGMMA support), A100, V100, RTX 4090, RTX 3090, and RTX A600; for AMD GPUs, it includes the MI250 (with Auto MatrixCore support) and the MI300X (with Async Copy support).

## OP Implementation Examples
**TileIR** provides the building blocks to implement a wide variety of operators. Some examples include:

- [Matrix Multiplication](./examples/gemm/)
- [Dequantization GEMM](./examples/dequantize_gemm/)
- [Flash Attention](./examples/flash_attention/)
- [Flash Linear Attention](./examples/linear_attention/)

Within the `examples` directory, you will also find additional complex kernels—such as convolutions and forward/backward passes for FlashAttention.

## Installation
### Method 1: Install with Pip

The quickest way to get started is to install the latest release from PyPI:

```bash
pip install tileir
```

Alternatively, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/microsoft/TileIR
```

Or install locally:

```bash
pip install .  # use -e if you want to install in editable mode
```

### Method 2: Build from Source
We currently provide three ways to install **TileIR** from source:
 - [Install from Source (using your own TVM installation)](./docs/Installation.md#install-from-source-with-your-own-tvm-installation)
 - [Install from Source (using the bundled TVM submodule)](./docs/Installation.md#install-from-source-with-our-tvm-submodule)
 - [Install Using the Provided Script](./docs/Installation.md#install-with-provided-script)

## Quick Start

Below is a simple example demonstrating how to write and execute a straightforward GEMM (matrix multiplication) kernel using TileIR, followed by techniques for layout optimizations, pipelining, and L2-cache–friendly swizzling!

### Basic GEMM Example

```python
import tilelang
from tilelang import Profiler
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        # Define a GPU kernel launch configuration:
        #   - Grid dimension: (ceildiv(N, block_N), ceildiv(M, block_M))
        #   - Threads per block: 128
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):

            # Allocate on-chip memory (shared and fragment buffers)
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Initialize the accumulation buffer
            T.clear(C_local)

            # Primary compute loop, with pipelining across chunks of size block_K
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy a tile of A into shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Copy a tile of B into shared memory
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # Perform a tile-level GEMM on the shared buffers into C_local
                T.gemm(A_shared, B_shared, C_local)

            # Write the accumulated result from local memory back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

# 1. Define the kernel (matmul) and compile/lower it into an executable module
func = matmul(1024, 1024, 1024, 128, 128, 32)
rt_mod, params = tileir.lower(func)

# 2. Create a Profiler object for running performance and correctness tests
profiler = Profiler(rt_mod, params, result_idx=[2])

# 3. Test the kernel in Python with PyTorch data
import torch

# Create random input tensors on the GPU
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# Run the kernel through the Profiler
c = profiler(a, b)

# Reference multiplication using PyTorch
ref_c = a @ b

# Validate correctness
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

# 4. Retrieve and inspect the generated CUDA source (optional)
cuda_source = rt_mod.imported_modules[0].get_source()
print("Generated CUDA kernel:\n", cuda_source)
```

### Enhanced Example with Annotations (Layout, L2 Cache Swizzling, Pipelining, etc.)

Below is an example showcasing more advanced features, including custom layout annotations, parallelized copy, and swizzling for improved L2 cache locality. This snippet demonstrates how you can adapt your kernel to maximize performance on complex hardware.

```python
import tileir.language as T
# `make_mma_swizzle_layout` is a Python-defined layout function
# specifically designed for MMA operations (e.g., to avoid bank conflicts).
from tileir.intrinsics import (
    make_mma_swizzle_layout as make_swizzle_layout,
)

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        # Kernel configuration
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Apply layout optimizations or define your own layout
            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Enable swizzle (rasterization) for better L2 cache locality
            T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                T.copy(A[by * block_M, k * block_K], A_shared)

                # Demonstrate parallelized copy from global to shared for B
                for ko, j in T.Parallel(block_K, block_N):
                    B_shared[ko, j] = B[k * block_K + ko, bx * block_N + j]

                # Perform a tile-level GEMM on the shared buffers
                T.gemm(A_shared, B_shared, C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### Explore More with TileIR Beyond GEMM

In addition to GEMM, TileIR provides various examples that highlight its versatility and power:

- [Dequantize GEMM](./examples/dequantize_gemm/): Achieve high-performance dequantization with **fine-grained control over per-thread operations**—adopted by [BitBLAS](https://github.com/microsoft/BitBLAS) to accelerate dequantized GEMM.
- [FlashAttention](./examples/flash_attention/): Showcases cross-operator fusion and includes an auto-tuning example.
- [LinearAttention](./examples/linear_attention/): Features RetNet and Mamba implementations.
- [Convolution](./examples/convolution/): Demonstrations of convolution kernels with IM2Col.

More operators are continuously being added.

---

TileIR is now integrated into the [BitBLAS](https://github.com/microsoft/BitBLAS) project.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's [Trademark & Brand Guidelines](https://www.microsoft.com/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third parties' policies.
