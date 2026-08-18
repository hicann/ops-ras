#
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

find_library(OPAPI_MATH_LIBRARY
    NAMES opapi_math
    PATHS ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    NO_DEFAULT_PATH
)
if(NOT OPAPI_MATH_LIBRARY)
    message(FATAL_ERROR
        "The installed CANN Math provider libopapi_math.so was not found under "
        "${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64")
endif()

add_library(opapi_math SHARED IMPORTED GLOBAL)
set_target_properties(opapi_math PROPERTIES
    IMPORTED_LOCATION "${OPAPI_MATH_LIBRARY}"
    IMPORTED_SONAME "libopapi_math.so"
)

if(ENABLE_TEST)
    add_library(ras_math_test_double SHARED ras_math_test_double.cpp)

    target_compile_definitions(ras_math_test_double PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        LOG_CPP
    )

    target_compile_options(ras_math_test_double PRIVATE -Wno-unused-parameter)

    target_include_directories(ras_math_test_double PRIVATE
        ${OPAPI_INCLUDE}
    )
endif()
