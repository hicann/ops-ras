# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_API_UT)
    # A file-list UT build may leave an empty cache entry after
    # -UASCEND_COMPUTE_UNIT. Restore the project default so both the source
    # collector and the deferred opbuild code generation receive a valid SoC.
    if(DEFINED ASCEND_COMPUTE_UNIT AND "${ASCEND_COMPUTE_UNIT}" STREQUAL "")
        set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "soc that need to be compiled" FORCE)
        set(ASCEND_COMPUTE_UNIT "ascend910b")
    endif()
    # GCC 15 no longer exposes uint32_t transitively through <memory> in the
    # shared op_api UT file I/O helper. Keep the compatibility workaround local
    # to builds that select this operator.
    cmake_language(DEFER DIRECTORY ${PROJECT_SOURCE_DIR} CALL
        target_compile_options op_api_ut_common PRIVATE -include cstdint)

    add_modules_ut_sources(HOSTNAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${OP_API_MODULE_NAME}_cases_obj PRIVATE ${CMAKE_BINARY_DIR}/autogen)
endif()
