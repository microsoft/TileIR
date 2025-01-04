# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tvm.tir import Buffer
from typing import List
from functools import reduce
# Scope Checkers for TVM Buffers
# These utility functions check the memory scope of a given TVM buffer.


def is_global(buffer: Buffer) -> bool:
    """
    Check if the buffer is in the global memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in global memory, False otherwise.
    """
    return buffer.scope() == "global"


def is_shared(buffer: Buffer, allow_dynamic: bool = True) -> bool:
    """
    Check if the buffer is in the shared memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in shared memory, False otherwise.
    """
    conditions = [False]
    conditions.append(buffer.scope() == "shared")
    if allow_dynamic:
        conditions.append(is_shared_dynamic(buffer))
    return any(conditions)

def is_shared_dynamic(buffer: Buffer) -> bool:
    """
    Check if the buffer is in the dynamic shared memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in dynamic shared memory, False otherwise.
    """
    return buffer.scope() == "shared.dyn"


def is_local(buffer: Buffer) -> bool:
    """
    Check if the buffer is in the local memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in local memory, False otherwise.
    """
    return buffer.scope() == "local"


def is_fragment(buffer: Buffer) -> bool:
    """
    Check if the buffer is a fragment (e.g., for matrix multiplication operations).

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is a fragment, False otherwise.
    """
    return buffer.scope().startswith("local.fragment")

def array_reduce(array: List[int]) -> int:
    """
    Reduce an array of integers to a single integer.

    Args:
        array (List[int]): The array of integers to reduce.

    Returns:
        int: The reduced integer.
    """
    return reduce(lambda x, y: x * y, array)
