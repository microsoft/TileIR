#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# if dist and build directories exist, remove them
if [ -d dist ]; then
    rm -r dist
fi

if [ -d build ]; then
    rm -r build
fi

python setup.py bdist_wheel

if [ $? -ne 0 ]; then
    echo "Error: Failed to build the wheel."
    exit 1
else
    echo "Wheel built successfully."
fi
