#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e
set -o pipefail
export LC_ALL=C.UTF-8

current_dir=$(pwd)
pr_file=$(realpath "${1:-pr_filelist.txt}")
variables_cmake="cmake/variables.cmake"
op_category_list=$(grep -oP 'set\(OP_CATEGORY_LIST\s*\K".*"' "$current_dir/$variables_cmake" | sed 's/"//g')
IFS=' ' read -r -a op_categories <<< "$op_category_list"

SOC_VERSION="ascend910b"
for arg in "$@"; do
    case "$arg" in
        --soc=*) SOC_VERSION="${arg#*=}" ;;
    esac
done

valid_dirs=()
for category in "${op_categories[@]}"
do
    first_level="$current_dir/$category"
    if [ -d "$first_level" ]; then
        for second_level in "$first_level"/*/
        do
            examples_dir="$second_level""examples"
            if [ -d "$examples_dir" ]; then
                if ls "$examples_dir"/test_aclnn_* 1>/dev/null 2>/dev/null; then
                    dir_name=$(basename "$second_level")
                    valid_dirs+=("$dir_name")
                fi
            fi
        done
    fi
done

ops_name=()
mapfile -t lines < <(grep -v "experimental" "${pr_file}")
for file_path in "${lines[@]}"
do
    file_path=$(echo "$file_path" | xargs)
    if [ -z "$file_path" ]; then
        continue
    fi
    if [[ "$file_path" == *.md ]]; then
        continue
    fi
    for dir in "${valid_dirs[@]}"
    do
        if [[ "$file_path" == *"/$dir/"* ]]; then
            if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                ops_name+=("$dir")
            fi
            break
        fi
    done
done

if [ ${#ops_name[@]} -eq 0 ]; then
    echo "no operator with examples need to run"
    exit 0
fi

for name in "${ops_name[@]}"
do
    echo "----------------"
    echo "${name}"
    run_files=(./single/cann-ops-ras-${name}_linux*.run)
    if [ ! -f "${run_files[0]}" ]; then
        run_files=(./single/*.run)
    fi
    if [ ! -f "${run_files[0]}" ]; then
        echo "ERROR: single package not found for ${name} in ./single"
        ls -l ./single/ 2>/dev/null || true
        exit 1
    fi
    chmod +x "${run_files[0]}"
    if ! "${run_files[0]}" --quiet --force; then
        echo "ERROR: install single package failed for ${name}"
        exit 1
    fi
    echo "bash build.sh --run_example $name eager cust --vendor_name=$name --soc=${SOC_VERSION}"
    if ! bash build.sh --run_example $name eager cust --vendor_name=$name --soc=${SOC_VERSION}; then
        echo "${name} example fail"
        exit 1
    fi
done
