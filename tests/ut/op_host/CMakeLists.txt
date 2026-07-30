# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_HOST_UT)
    file(GLOB SUBDIRECTORIES LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUBDIR ${SUBDIRECTORIES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
            add_subdirectory(${SUBDIR})
        endif()
    endforeach()

    ## Build the OpHost UT binary separately from the public build target.  A
    ## custom target is always out of date, so an explicit `build.sh -u
    ## --ophost` invocation executes the test even when the binary itself is
    ## already up to date.
    set(OP_HOST_UT_EXE ${PKG_NAME}_op_host_ut)
    set(OP_HOST_UT_BIN ${OP_HOST_UT_EXE}_bin)
    add_executable(${OP_HOST_UT_BIN}
        test_op_host_main.cpp
        test_host_common.cpp
        ${PROJECT_SOURCE_DIR}/common/src/op_host/hash.cpp
    )

    # Keep runtime resolution explicit in the execution target below.
    set_target_properties(${OP_HOST_UT_BIN} PROPERTIES
        SKIP_BUILD_RPATH TRUE
    )

    target_compile_definitions(${OP_HOST_UT_BIN} PRIVATE LOG_CPP)
    target_compile_options(${OP_HOST_UT_BIN} PUBLIC -fPIE -fno-access-control)
    target_include_directories(${OP_HOST_UT_BIN} PRIVATE
        ${PROJECT_SOURCE_DIR}/common/inc
    )
    target_link_libraries(${OP_HOST_UT_BIN} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        -Wl,--whole-archive
        $<$<TARGET_EXISTS:${OP_TILING_MODULE_NAME}_static_lib>:${OP_TILING_MODULE_NAME}_static_lib>
        $<$<TARGET_EXISTS:${OP_INFERSHAPE_MODULE_NAME}_static_lib>:${OP_INFERSHAPE_MODULE_NAME}_static_lib>
        -Wl,--no-whole-archive
        -Wl,--no-as-needed
        metadef
        -Wl,--as-needed
        error_manager
        exe_graph
        graph_base
        gtest
        graph
        platform
        register
        opp_registry
        $<$<TARGET_EXISTS:opsbase>:opsbase>
        tiling_api
        dlog
        dl
    )

    add_dependencies(${OP_HOST_UT_BIN} json)

    if(ENABLE_UT_EXEC)
        set(ASCEND_HAL_STUB
            ${ASCEND_DIR}/${SYSTEM_PREFIX}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}/libascend_hal.so)
        if(NOT EXISTS ${ASCEND_HAL_STUB})
            message(FATAL_ERROR "OpHost UT requires Ascend HAL stub: ${ASCEND_HAL_STUB}")
        endif()

        if(${ENABLE_ASAN} STREQUAL "TRUE")
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE LIBASAN_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE LIBASAN_RESULT
            )
            if(NOT LIBASAN_RESULT EQUAL 0)
                message(FATAL_ERROR "Compiler does not provide libasan.so")
            endif()
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
                OUTPUT_VARIABLE LIBSTDC_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE LIBSTDC_RESULT
            )
            if(NOT LIBSTDC_RESULT EQUAL 0)
                message(FATAL_ERROR "Compiler does not provide libstdc++.so")
            endif()
            add_custom_target(${OP_HOST_UT_EXE}
                COMMAND ${CMAKE_COMMAND} -E env
                    BUILD_PATH=${BUILD_PATH}
                    LD_PRELOAD=${LIBASAN_PATH}:${LIBSTDC_PATH}:${ASCEND_HAL_STUB}:$ENV{LD_PRELOAD}
                    ASAN_OPTIONS=detect_leaks=0
                    $<TARGET_FILE:${OP_HOST_UT_BIN}>
                DEPENDS ${OP_HOST_UT_BIN}
                COMMENT "Run ops op_host utest with asan"
                USES_TERMINAL
                VERBATIM
            )
        elseif(${ENABLE_VALGRIND} STREQUAL "TRUE")
            add_custom_target(${OP_HOST_UT_EXE}
                COMMAND ${CMAKE_COMMAND} -E env
                    LD_PRELOAD=${ASCEND_HAL_STUB}:$ENV{LD_PRELOAD}
                    valgrind --undef-value-errors=no --leak-check=full $<TARGET_FILE:${OP_HOST_UT_BIN}>
                DEPENDS ${OP_HOST_UT_BIN}
                COMMENT "Run ops op_host utest by valgrind"
                USES_TERMINAL
                VERBATIM
            )
        else()
            add_custom_target(${OP_HOST_UT_EXE}
                COMMAND ${CMAKE_COMMAND} -E env
                    LD_PRELOAD=${ASCEND_HAL_STUB}:$ENV{LD_PRELOAD}
                    $<TARGET_FILE:${OP_HOST_UT_BIN}>
                DEPENDS ${OP_HOST_UT_BIN}
                COMMENT "Run ops op_host utest"
                USES_TERMINAL
                VERBATIM
            )
        endif()
    else()
        add_custom_target(${OP_HOST_UT_EXE}
            DEPENDS ${OP_HOST_UT_BIN}
        )
    endif()
endif()
