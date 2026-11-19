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

if [ -z "${WORKSPACE}" ]; then
    export WORKSPACE=$(pwd)
fi
if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi

current_dir=$(pwd)
pr_file=$(realpath "${1:-pr_filelist.txt}")
variables_cmake="cmake/variables.cmake"
op_category_list=$(grep -oP 'set\(OP_CATEGORY_LIST\s*\K".*"' "$current_dir/$variables_cmake" | sed 's/"//g')
IFS=' ' read -r -a op_categories <<< "$op_category_list"

valid_dirs=()
for category in "${op_categories[@]}"
do
    category_path="$current_dir/$category"
    if [ -d "$category_path" ]; then
        for dir in "$category_path"/*/
        do
            dir_name=$(basename "$dir")
            if [[ "$dir_name" != *"common"* ]]; then
                valid_dirs+=("$dir_name")
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
        if [[ "$file_path" == *"/$dir/"* ]];then
            if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                ops_name+=("$dir")
            fi
            break
        fi
    done
done

mkdir -p ${WORKSPACE}/single

if [ ${#ops_name[@]} -eq 0 ]; then
    echo "no operator need to build single package"
    cd ${WORKSPACE}
    tar -zcf single.tar.gz single/
    exit 0
fi

for name in "${ops_name[@]}"
do
    echo "----------------"
    echo "${name}"
    rm -rf build
    rm -rf build_out
    echo "bash build.sh --pkg --vendor_name=$name --ops=$name --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16"
    if ! bash build.sh --pkg --vendor_name=$name --ops=$name --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16; then
        echo "check pkg fail"
        exit 1
    fi
    run_files=(build_out/cann-ops-ras-${name}_linux*.run)
    if [ ! -f "${run_files[0]}" ]; then
        run_files=(build_out/*.run)
    fi
    if [ ! -f "${run_files[0]}" ]; then
        echo "ERROR: single package not found for ${name} in build_out"
        ls -l build_out/ 2>/dev/null || true
        exit 1
    fi
    cp "${run_files[0]}" ${WORKSPACE}/single/
done
echo "+++++++++++++++++++++++++++=="
cd ${WORKSPACE}
ls
tar -zcf single.tar.gz single/
ls
