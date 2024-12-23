# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Union, List, Optional, Tuple
from tvm import tir
from tvm.tir import Var
from tvm.script.ir_builder.tir.frame import TIRFrame
from tvm._ffi import register_object
from tilelang import _ffi_api


@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):

    def __enter__(self) -> Union[Var, List[Var]]:  # type: ignore[override]
        # Frames: BlockIdx.x, BlockIdx.y, BlockIdx.z, ThreadIdx.x, ThreadIdx.y, ThreadIdx.z, Root Block
        super().__enter__()
        if len(self.frames) == 5:
            return self.frames[0].iter_var.var
        return [frame.iter_var.var for frame in self.frames[0:-4]]


def Kernel(
    *blocks: List[tir.PrimExpr],
    threads: Union[int, List[int], Tuple] = 128,
    prelude: Optional[str] = None,
):
    """Tools to quickly construct a GPU kernel launch frame.

    Parameters
    ----------
    blocks : List[int]
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    threads : int
        A integer representing blockDim.x
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.
    prelude : str
        The import c code of the kernel,
        will be injected before the generated kernel code.
    layout_annotation: Optional[Map[tir.Buffer, tir.IndexMap]]
        The layout annotation map, used to annotate the layout of the buffers.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    """
    attrs: dict = {}

    if isinstance(threads, int):
        threads = [threads, 1, 1]
    elif isinstance(threads, list):
        threads = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        threads = list(threads) + [1] * (3 - len(threads))
    else:
        raise ValueError("threads must be an integer or a list of integers")

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.KernelLaunch(blocks, threads, attrs)
