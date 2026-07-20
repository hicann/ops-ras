/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_obfuscation_setup.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "obfuscation_setup.h"
#include "opdev/op_executor.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
 
// 根据API定义，需要列出所能支持的所有dtype
static bool CheckNotNull(const aclTensor* fd)
{
  OP_CHECK_NULL(fd, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* fd)
{
  OP_CHECK_DTYPE_NOT_MATCH(fd, op::DataType::DT_INT32, return false);
  return true;
}

static bool CheckShape(const aclTensor* fd)
{
  OP_CHECK(fd->GetViewShape().GetDimNum() == 1,
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                  "Shape of fd tensor should be [1], but current is %s.",
                  op::ToString(fd->GetViewShape()).GetString()),
          return false);
  return true;
}

static aclnnStatus CheckParams(const int32_t fdToClose, const int32_t dataType,
                               const int32_t hiddenSize, const int32_t tpRank,
                               const int32_t modelObfSeedId, const int32_t dataObfSeedId,
                               const int32_t cmd, const int32_t threadNum, const aclTensor* fd)
{
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(fd), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入输出数据类型
  CHECK_RET(CheckDtypeValid(fd), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输出输出shape
  CHECK_RET(CheckShape(fd), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnObfuscationSetupGetWorkspaceSize(int32_t fdToClose, int32_t dataType, 
                                                  int32_t hiddenSize, int32_t tpRank,
                                                  int32_t modelObfSeedId, int32_t dataObfSeedId, 
                                                  int32_t cmd, int32_t threadNum, aclTensor* fd, 
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
  L2_DFX_PHASE_1(aclnnObfuscationSetup, DFX_IN(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId, dataObfSeedId, cmd, threadNum), DFX_OUT(fd));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, fd);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  auto fdToCloseTensor = uniqueExecutor.get() -> ConvertToTensor(&fdToClose, 1, op::DataType::DT_INT32);
  auto dataTypeTensor = uniqueExecutor.get() -> ConvertToTensor(&dataType, 1, op::DataType::DT_INT32);
  auto hiddenSizeTensor = uniqueExecutor.get() -> ConvertToTensor(&hiddenSize, 1, op::DataType::DT_INT32);
  auto tpRankTensor = uniqueExecutor.get() -> ConvertToTensor(&tpRank, 1, op::DataType::DT_INT32);
  auto modelObfSeedIdTensor = uniqueExecutor.get() -> ConvertToTensor(&modelObfSeedId, 1, op::DataType::DT_INT32);
  auto dataObfSeedIdTensor = uniqueExecutor.get() -> ConvertToTensor(&dataObfSeedId, 1, op::DataType::DT_INT32);
  float obfCoefficient = 1.0;
  auto fdOut = l0op::ObfuscationSetup(fdToCloseTensor, dataTypeTensor, hiddenSizeTensor, tpRankTensor, 
                                      modelObfSeedIdTensor,dataObfSeedIdTensor, cmd, threadNum, obfCoefficient, uniqueExecutor.get());

  // 固定写法，将计算结果转换成输出fd的数据类型
  auto castFd = l0op::Cast(fdOut, fd->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castFd != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出y上，y可能是非连续的tensor
  auto viewCopyFd = l0op::ViewCopy(castFd, fd, uniqueExecutor.get());
  CHECK_RET(viewCopyFd != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnObfuscationSetup(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnObfuscationSetup);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif