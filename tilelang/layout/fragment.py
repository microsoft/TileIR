# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm
from tvm.ir import Range
from tvm.tir import IterVar, Var
from tilelang import _ffi_api
from tilelang.layout import Layout

@tvm._ffi.register_object("tl.Fragment")
class Fragment(Layout):
    # pylint: disable=super-init-not-called
    def __init__(self, shape, forward_thread_fn, replicate=1, forward_fn=None):
        forward_vars = []
        for idx, size in enumerate(shape):
            iv = IterVar(Range(0, size), Var(f"i{idx}", "int32"), 0)
            forward_vars.append(iv)
        vars = [iv.var for iv in forward_vars]

        forward_index = forward_fn(*vars) if forward_fn else None

        if replicate > 1:
            thread_replicate = IterVar(
                Range(0, replicate), Var("rep", "int32"), 0
            )
            forward_thread = forward_thread_fn(*vars, thread_replicate.var)
        else:
            thread_replicate = None
            forward_thread = forward_thread_fn(*vars)
        self.__init_handle_by_constructor__(
            _ffi_api.Fragment,
            forward_vars,
            forward_index,
            forward_thread,
            thread_replicate,
        )

    @property
    def thread(self):
        return _ffi_api.Fragment_thread(self)

    def get_thread_size(self):
        return _ffi_api.Fragment_thread_size(self)

    def repeat(self, repeats, repeat_on_thread: bool = False) -> "Fragment":
        return _ffi_api.Fragment_repeat(self, repeats, repeat_on_thread)

    def condense_rep_var(self) -> "Fragment":
        return _ffi_api.Fragment_condense_rep_var(self)


def make_swizzled_layout(buffer: tvm.tir.Buffer):
    assert len(buffer.shape) == 2
    return _ffi_api.make_swizzled_layout(
        int(buffer.shape[0]),
        int(buffer.shape[1]),
        int(tvm.DataType(buffer.dtype).bits),
    )
