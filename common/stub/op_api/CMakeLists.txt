#
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

function(add_ras_math_stub target_name)
    add_library(${target_name} SHARED ras_math_test_double.cpp)

    target_compile_definitions(${target_name} PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        LOG_CPP
    )

    target_compile_options(${target_name} PRIVATE -Wno-unused-parameter)

    target_include_directories(${target_name} PRIVATE
        ${OPAPI_INCLUDE}
    )
endfunction()

# The standalone build needs a link-time provider before the real ops-math
# package is installed. This target is intentionally not installed; at runtime
# libopapi_ras.so resolves these symbols from the ops-math package.
add_ras_math_stub(opapi_math)

if(ENABLE_TEST)
    add_ras_math_stub(ras_math_test_double)
endif()
