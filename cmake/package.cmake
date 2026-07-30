# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### CPACK to package run #####

function(pack_custom)
  add_cann_third_party(makeself-fetch)

  npu_op_package(${PACK_CUSTOM_NAME}
    TYPE RUN
    CONFIG
      ENABLE_SOURCE_PACKAGE True
      ENABLE_BINARY_PACKAGE True
      INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/
      VENDOR_NAME "${VENDOR_PACKAGE_NAME}"
      ENABLE_DEFAULT_PACKAGE_NAME_RULE False
  )
  set(op_package_list)
  if(ENABLE_ASC_BUILD)
    add_custom_kernel_library(ascendc_kernels)
    list(APPEND op_package_list ${ascendc_kernels})
  endif()
  if(TARGET ${OPHOST_NAME}_opapi_obj OR TARGET opbuild_gen_aclnn_all)
    list(APPEND op_package_list cust_opapi)
  endif()
  if(TARGET cust_proto)
    list(APPEND op_package_list cust_proto)
  endif()
  if(TARGET ${OPHOST_NAME}_tiling_obj)
    list(APPEND op_package_list cust_opmaster)
  endif()

  list(LENGTH op_package_list OP_PACKAGE_LENGTH)
  if(OP_PACKAGE_LENGTH GREATER 0)
    npu_op_package_add(${PACK_CUSTOM_NAME}
      LIBRARY
        ${op_package_list}
    )
  endif()

  set(custom_proto_target "${PACK_CUSTOM_NAME}_ascendc_cust_op_proto")
  if(TARGET ${custom_proto_target})
    set_target_properties(${custom_proto_target} PROPERTIES
      INSTALL_RPATH "\$ORIGIN/../../../es/lib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    )
  endif()
endfunction()

macro(INSTALL_SCRIPTS src_path)
  install(DIRECTORY ${src_path}/
      DESTINATION share/info/ops_ras/script
      FILE_PERMISSIONS
          OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
          GROUP_READ GROUP_EXECUTE
          WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
          OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
          GROUP_READ GROUP_EXECUTE
          WORLD_READ WORLD_EXECUTE
      REGEX "(setenv|prereq_check)\\.(bash|fish|csh)" EXCLUDE
  )
endmacro()

function(pack_built_in)
  # 打印路径
  message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "CANN_CMAKE_DIR = ${CANN_CMAKE_DIR}")
  message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

  set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/ops_ras/scripts)
  INSTALL_SCRIPTS(${script_prefix})

  set(SCRIPTS_FILES
      ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
      ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
      ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
  )

  install(FILES ${SCRIPTS_FILES}
      DESTINATION share/info/ops_ras/script
  )
  set(COMMON_FILES
      ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
      ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
      ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
  )

  set(PACKAGE_FILES
      ${COMMON_FILES}
      ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
  )
  set(CONF_FILES
      ${CANN_CMAKE_DIR}/scripts/package/cfg/path.cfg
  )
  install(FILES ${CMAKE_BINARY_DIR}/version.ops-ras.info
      DESTINATION share/info/ops_ras
      RENAME version.info
  )
  install(FILES ${CONF_FILES}
      DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION share/info/ops_ras/script
  )

  string(FIND "${ASCEND_COMPUTE_UNIT}" ";" SEMICOLON_INDEX)
  if (SEMICOLON_INDEX GREATER -1)
      # 截取分号前的字串
      math(EXPR SUBSTRING_LENGTH "${SEMICOLON_INDEX}")
      string(SUBSTRING "${ASCEND_COMPUTE_UNIT}" 0 "${SUBSTRING_LENGTH}" compute_unit)
  else()
      # 没有分号取全部内容
      set(compute_unit "${ASCEND_COMPUTE_UNIT}")
  endif()

  message(STATUS "current compute_unit is: ${compute_unit}")
  set(script_with_soc_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/ops_ras/scripts)
  INSTALL_SCRIPTS(${script_with_soc_prefix})

  # 打包可选的 cann_ops_ras whl；PR8 未落地时安全跳过。
  set(WHL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/torch_extension/dist")
  file(GLOB WHL_FILES "${WHL_SOURCE_DIR}/cann_ops_ras-*.whl")

  if(WHL_FILES)
      install(FILES ${WHL_FILES}
          DESTINATION python/site-packages
      )
      message(STATUS "Including whl package: ${WHL_FILES}")
  else()
      message(STATUS "Optional cann_ops_ras whl not found in ${WHL_SOURCE_DIR}, skipping")
  endif()

  include(${CMAKE_SOURCE_DIR}/cmake/runtimeKB.cmake)
  set_cann_cpack_config(ops-ras ENABLE_DEVICE ${ENABLE_DEVICE} COMPUTE_UNIT ${ASCEND_COMPUTE_UNIT} SHARE_INFO_NAME ops_ras)
endfunction()
