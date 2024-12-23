import tilelang
from tilelang import Profiler
import tilelang.language as T
from tilelang.autotuner import *
import itertools
import argparse


def get_configs():
    block_M = [128]
    block_N = [128, 256]
    block_K = [64]
    num_stages = [2]
    threads = [256]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'block_K': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def matmul(M, N, K, tune=False):
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_N, block_K, num_stages, threads):

        @T.prim_func
        def main(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N),
                                                                                     dtype)):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(10)
                T.annotate_layout({C_shared: tilelang.layout.make_swizzled_layout(C_shared)})

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    if tune:

        @autotune(
            configs=get_configs(),
            keys=["block_M", "block_N", "block_K", "num_stages", "threads"],
            warmup=10,
            rep=10)
        @jit(
            out_idx=[2],
            supply_type=tilelang.TensorSupplyType.Integer,
            ref_prog=ref_program,
            profiler="auto")
        def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_N, block_K, num_stages, threads)

        return kernel()
    else:

        def kernel(block_M, block_N, block_K, num_stages, threads):
            return kernel_func(block_M, block_N, block_K, num_stages, threads)

        return kernel


def ref_program(A, B):
    return A @ B


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=8192, help='M')
    parser.add_argument('--n', type=int, default=8192, help='N')
    parser.add_argument('--k', type=int, default=8192, help='K')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    total_flops = 2 * M * N * K

    if (not args.tune):
        program = matmul(
            M, N, K, tune=args.tune)(
                block_M=256, block_N=128, block_K=64, num_stages=4, threads=256)
        mod, params = tilelang.lower(program)
        mod = Profiler(mod, params, [2], tilelang.TensorSupplyType.Normal)
        mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = mod.do_bench(ref_program, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_latency, best_config, ref_latency = matmul(M, N, K, tune=args.tune)
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
