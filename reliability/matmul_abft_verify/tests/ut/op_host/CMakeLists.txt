# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    # A file-list UT build may define ASCEND_COMPUTE_UNIT as an empty string.
    # add_modules_ut_sources treats every defined value as a SoC name, and an
    # empty expansion makes map_compute_unit/get_target_dir lose an argument.
    if(DEFINED ASCEND_COMPUTE_UNIT AND "${ASCEND_COMPUTE_UNIT}" STREQUAL "")
        unset(ASCEND_COMPUTE_UNIT CACHE)
        unset(ASCEND_COMPUTE_UNIT)
    endif()
    # CANN 9.2 removed gert::InfershapeContextPara, which is still used by the
    # repository-wide executor. This operator uses InferShapeContextFaker
    # directly, so exclude only that obsolete implementation from its target.
    get_target_property(MATMUL_ABFT_VERIFY_INFERSHAPE_COMMON_SRCS
        ${OP_INFERSHAPE_MODULE_NAME}_common_obj SOURCES)
    list(FILTER MATMUL_ABFT_VERIFY_INFERSHAPE_COMMON_SRCS
        EXCLUDE REGEX "infershape_case_executor\\.cpp$")
    set_property(TARGET ${OP_INFERSHAPE_MODULE_NAME}_common_obj PROPERTY SOURCES
        "${MATMUL_ABFT_VERIFY_INFERSHAPE_COMMON_SRCS}")

    # GCC 15 no longer provides std::find transitively to the shared tiling
    # faker. Keep the compatibility workaround local to this operator's UT.
    target_compile_options(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        -include algorithm)

    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(HOSTNAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
