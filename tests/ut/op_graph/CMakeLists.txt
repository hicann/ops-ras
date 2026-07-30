# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_GRAPH_UT)
    has_graph_proto_sources(HAS_GRAPH_PROTO_SOURCES)
endif()

if((UT_TEST_ALL OR OP_GRAPH_UT) AND HAS_GRAPH_PROTO_SOURCES)
    merge_graph_headers(TARGET merge_ops_proto_${PKG_NAME} OUT_DIR ${ASCEND_GRAPH_CONF_DST})
    gen_es_ras_lib_ready()

    if(NOT TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})
    endif()

    file(GLOB SUBDIRECTORIES LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUBDIR ${SUBDIRECTORIES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
            add_subdirectory(${SUBDIR})
        endif()
    endforeach()

    set(OP_GRAPH_UT_EXE ${PKG_NAME}_op_graph_ut)
    file(GLOB INFERSHAPE_STUB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/stub/*_infershape.cpp)
    add_executable(${OP_GRAPH_UT_EXE}
        ${INFERSHAPE_STUB_FILES}
        test_op_graph_main.cpp
    )

    target_compile_definitions(${OP_GRAPH_UT_EXE} PRIVATE LOG_CPP)
    target_compile_options(${OP_GRAPH_UT_EXE} PUBLIC -fPIE -fno-access-control)
    target_include_directories(${OP_GRAPH_UT_EXE} PRIVATE
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/pkg_inc
    )
    add_dependencies(${OP_GRAPH_UT_EXE} json build_es_ras)
    target_link_directories(${OP_GRAPH_UT_EXE} PRIVATE
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
        ${CMAKE_BINARY_DIR}/es_packages/lib64
    )
    target_link_libraries(${OP_GRAPH_UT_EXE} PRIVATE
        es_ras
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        -Wl,--allow-multiple-definition
        -Wl,--whole-archive
        $<$<TARGET_EXISTS:${OP_GRAPH_MODULE_NAME}_static_lib>:${OP_GRAPH_MODULE_NAME}_static_lib>
        -Wl,--no-whole-archive
        -Wl,--no-as-needed
        metadef
        -Wl,--as-needed
        error_manager
        unified_dlog
        exe_graph
        graph_base
        gtest
        graph
        platform
        ge_common
        register
        opp_registry
        rt2_registry
        runtime
        c_sec
        mmpa
        dlog
        dl
        ascend_protobuf
    )

    if(ENABLE_UT_EXEC)
        add_custom_command(
            TARGET ${OP_GRAPH_UT_EXE} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env
                LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/es_packages/lib64:${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64:$ENV{LD_LIBRARY_PATH}
                ./${OP_GRAPH_UT_EXE}
            COMMENT "Run ops op_graph utest"
        )
    endif()
elseif(UT_TEST_ALL OR OP_GRAPH_UT)
    message(STATUS "No RAS op_graph proto sources found; op_graph UT is skipped.")
    add_custom_target(${PKG_NAME}_op_graph_ut)
endif()
