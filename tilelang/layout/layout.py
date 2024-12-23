# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm
from tvm.ir import Node, Range
from tvm.tir import IterVar, Var, PrimExpr
from tilelang import _ffi_api


@tvm._ffi.register_object("tl.Layout")
class Layout(Node):

    def __init__(self, shape, forward_fn):
        forward_vars = []
        for idx, size in enumerate(shape):
            iv = IterVar(Range(0, size), Var(f"i{idx}", "int32"), 0)
            forward_vars.append(iv)
        vars = [iv.var for iv in forward_vars]
        forward_index = forward_fn(*vars)
        if isinstance(forward_index, PrimExpr):
            forward_index = [forward_index]
        self.__init_handle_by_constructor__(_ffi_api.Layout, forward_vars, forward_index)

    @property
    def index(self):
        return _ffi_api.Layout_index(self)

    def get_input_shape(self):
        return _ffi_api.Layout_input_shape(self)

    def get_output_shape(self):
        return _ffi_api.Layout_output_shape(self)

    def inverse(self) -> "Layout":
        return _ffi_api.Layout_inverse(self)
