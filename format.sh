#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script runs code formatters and linters on the codebase.
#
# Tools involved:
# - YAPF for Python formatting.
# - Ruff for Python linting.
# - Codespell for spelling checks in Python files.
# - Clang-format for C/C++ formatting.
#
# Usage:
#    # Do your work and commit changes.
#    # Then run:
#    bash format.sh
#
#    # This will:
#    #  - Format Python files that differ from origin/main using YAPF.
#    #  - Check spelling in changed Python files using Codespell.
#    #  - Lint changed Python files using Ruff.
#    #  - Format changed C/C++ files with clang-format.
#
#    # To explicitly format all files or specific files, use:
#    #    bash format.sh --all
#    #    bash format.sh --files path/to/file.py
#
#    # After running, if any files were changed by the formatters, you will need
#    # to review and commit those changes again before pushing.

set -eo pipefail

# Move to script directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')
CODESPELL_VERSION=$(codespell --version)

# Function to check tool versions against requirements-dev.txt
tool_version_check() {
    # params: tool name, installed version, required version
    if [[ $2 != $3 ]]; then
        echo "Wrong $1 version installed: $2 is installed, but $3 is required."
        exit 1
    fi
}

tool_version_check "yapf" $YAPF_VERSION "$(grep yapf requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "ruff" $RUFF_VERSION "$(grep "ruff==" requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "codespell" "$CODESPELL_VERSION" "$(grep codespell requirements-dev.txt | cut -d'=' -f3)"

echo 'tile-lang yapf: Check Start'

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
)

# Format specified Python files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format changed Python files relative to main branch
format_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi
}

# Format all Python files
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .
}

if [[ "$1" == '--files' ]]; then
   format "${@:2}"
elif [[ "$1" == '--all' ]]; then
   format_all
else
   format_changed
fi
echo 'tile-lang yapf: Done'

echo 'tile-lang codespell: Check Start'
# Run codespell on specified files
spell_check() {
    codespell "$@"
}

spell_check_all() {
  codespell --toml pyproject.toml
}

# Check spelling in changed Python files
spell_check_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             codespell
    fi
}

if [[ "$1" == '--files' ]]; then
   spell_check "${@:2}"
elif [[ "$1" == '--all' ]]; then
   spell_check_all
else
   spell_check_changed
fi
echo 'tile-lang codespell: Done'

echo 'tile-lang ruff: Check Start'
# Lint specified Python files
lint() {
    ruff "$@"
}

# Lint changed Python files
lint_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             ruff
    fi
}

if [[ "$1" == '--files' ]]; then
   lint "${@:2}"
elif [[ "$1" == '--all' ]]; then
   lint TileLang tests
else
   lint_changed
fi
echo 'tile-lang ruff: Done'

echo 'tile-lang clang-format: Check Start'
# If clang-format is available, run it; otherwise, skip
if command -v clang-format &>/dev/null; then
    CLANG_FORMAT_VERSION=$(clang-format --version | awk '{print $3}')
    tool_version_check "clang-format" "$CLANG_FORMAT_VERSION" "$(grep clang-format requirements-dev.txt | cut -d'=' -f3)"

    CLANG_FORMAT_FLAGS=("-i")

    # Apply clang-format to specified files
    clang_format() {
        clang-format "${CLANG_FORMAT_FLAGS[@]}" "$@"
    }

    # Format all C/C++ files in the repo
    clang_format_all() {
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -exec clang-format -i {} +
    }

    # Format changed C/C++ files relative to main
    clang_format_changed() {
        if git show-ref --verify --quiet refs/remotes/origin/main; then
            BASE_BRANCH="origin/main"
        else
            BASE_BRANCH="main"
        fi

        MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

        if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' &>/dev/null; then
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' | xargs clang-format -i
        fi
    }

    if [[ "$1" == '--files' ]]; then
       # If --files is given, format only the provided files
       clang_format "${@:2}"
    elif [[ "$1" == '--all' ]]; then
       # If --all is given, format all eligible C/C++ files
       clang_format_all
    else
       # Otherwise, format only changed C/C++ files
       clang_format_changed
    fi
else
    echo "clang-format not found. Skipping C/C++ formatting."
fi
echo 'tile-lang clang-format: Done'

# Check if there are any uncommitted changes after all formatting steps.
# If there are, ask the user to review and stage them.
if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi

echo 'tile-lang: All checks passed'
