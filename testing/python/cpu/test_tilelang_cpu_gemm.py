# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    num_stages = 0

    @T.prim_func
    def matmul(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_local = T.alloc_local((block_M, block_K), dtype)
            B_local = T.alloc_local((block_K, block_N), dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)

            T.clear(C_local)

            # Apply layout optimizations or define your own layout
            # (Optional).
            # T.annotate_layout(
            #     {
            #         A_local: make_swizzle_layout(A_local),
            #         B_local: make_swizzle_layout(B_local),
            #     }
            # )

            for ko in T.Pipelined(K // block_K, num_stages=num_stages):

                T.copy(A[by * block_M, ko * block_K], A_local)

                # Or Copy with Parallel
                for k, j in T.Parallel(block_K, block_N):
                    B_local[k, j] = B[ko * block_K + k, by * block_N + j]

                for i, j, k in T.grid(block_M, block_N, block_K):
                    C_local[i, j] += A_local[i, k] * B_local[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul


def assert_matmul_codegen(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32):
    func = matmul(M, N, K, block_M, block_N, block_K)

    rt_mod, _ = tilelang.lower(func, target="c")

    code = rt_mod.imported_modules[0].get_source()

    assert code is not None, "Code generation failed"


def test_matmul_codegen():
    assert_matmul_codegen(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)


if __name__ == "__main__":
    tilelang.testing.main()
