# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(ASCEND_OP_NAME "" CACHE STRING "Ascend op names to compile")
set(OP_TYPE "matmul_abft_verify")

# ACLNN code generation is disabled unless this switch is visible from the
# top-level generation stage. Keep the setting local to this operator's build
# configuration instead of changing the repository root CMakeLists.txt.
set(ENABLE_GEN_ACLNN ON CACHE BOOL "Generate ACLNN APIs from operator definitions" FORCE)

if(ASCEND_OP_NAME)
  set(_ASCEND_OP_NAME_TMP "${ASCEND_OP_NAME}")
  separate_arguments(_ASCEND_OP_NAME_TMP)

  list(FIND _ASCEND_OP_NAME_TMP "${OP_TYPE}" _index)
  if(_index EQUAL -1)
    message(STATUS "[${OP_TYPE}] skipped, not in ASCEND_OP_NAME list: ${ASCEND_OP_NAME}")
    return()
  else()
    message(STATUS "[${OP_TYPE}] selected, in ASCEND_OP_NAME list: ${ASCEND_OP_NAME}")
  endif()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
    add_subdirectory(${SUB_DIR})
  endif()
endforeach()

add_ops_compile_options(
    MatmulAbftVerify
    OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)
add_modules_sources(
    HOSTNAME ${OPHOST_NAME}
    MODE PRIVATE
    DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE matmul_abft_verify
    ACLNNTYPE aclnn
)
