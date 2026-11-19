#!/bin/bash

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
set -o pipefail

echo "ge_st_rt2: ${ge_st_rt2}"
echo "GIT_TARGET_BRANCH: ${GIT_TARGET_BRANCH}"
echo "ut_type: ${ut_type}"

########
# Init #
########

function LOG_HEAD() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}"
}

function LOG_DO() {
   local cmd="$*"
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "[Command] ${date_time} ${cmd}"
   ${cmd}
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    if [ "${actual_value}" != "${expect_value}" ]; then
        echo "${assert_msg} is failed."
        exit 1
    else
        echo "${assert_msg} is success."
    fi
}

REPOSITORY_NAME="ops-ras"

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
    echo "Switched to gcc-16"
elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
    echo "Switched to gcc-15"
elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
    echo "gcc-16/15 not available, fell back to gcc-14"
fi
if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi
source /home/jenkins/Ascend/cann/bin/setenv.bash
main(){
    LOG_HEAD "Start run c++ testcase"
    if [ "${ge_st_rt2}X" == "ophostX" ];then
      LOG_DO bash build.sh -u --ophost --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -f "pr_filelist.txt" -j16
      DP_ASSERT_EQUAL "$?" "0" "Run ophost UT testcase"
    elif [ "${ge_st_rt2}X" == "opapiX" ];then
      LOG_DO bash build.sh -u --opapi --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -f "pr_filelist.txt" -j16
      DP_ASSERT_EQUAL "$?" "0" "Run opapi UT testcase"
    elif [ "${ge_st_rt2}X" == "opgraphX" ];then
      if [ "${GIT_TARGET_BRANCH}" = "master" ];then
        LOG_DO bash build.sh -u --opgraph --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -f "pr_filelist.txt" -j16
        DP_ASSERT_EQUAL "$?" "0" "Run opgraph UT testcase"
      else
        echo "not need run opgraph ut"
        exit 0
      fi
    fi
}
main_param=$@
main $main_param
