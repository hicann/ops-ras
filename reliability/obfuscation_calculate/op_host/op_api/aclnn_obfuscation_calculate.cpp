/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_obfuscation_calculate.h"

#include "obfuscation_calculate.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_executor.h"
 
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_BF16
};

static bool CheckNotNull(const aclTensor* x, const aclTensor* y)
{
  OP_CHECK_NULL(x, return false);
  OP_CHECK_NULL(y, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
  size_t xDimNum = x->GetViewShape().GetDimNum();
  size_t yDimNum = y->GetViewShape().GetDimNum();
  OP_CHECK(xDimNum == yDimNum,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of x tensor should be same with y tensor, but (got x: %s, y: %s)", 
            op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString()), return false);
  return true;
}

static aclnnStatus CheckParams(const int32_t fd, const aclTensor* x, const int32_t param, 
                              const int32_t cmd, const aclTensor* y)
{
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(x, y), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入输出数据类型
  CHECK_RET(CheckDtypeValid(x, y), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输出输出shape
  CHECK_RET(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnObfuscationCalculateGetWorkspaceSize(int32_t fd, const aclTensor* x, int32_t param,
                                                      int32_t cmd, aclTensor* y, 
                                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
  L2_DFX_PHASE_1(aclnnObfuscationCalculate, DFX_IN(fd, x, param, cmd), DFX_OUT(y));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(fd, x, param, cmd, y);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (x->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入x转换成连续的tensor
  auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
  CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行ObfuscationCalculate计算
  auto fdTensor = uniqueExecutor.get() -> ConvertToTensor(&fd, 1, op::DataType::DT_INT32);
  auto paramTensor = uniqueExecutor.get() -> ConvertToTensor(&param, 1, op::DataType::DT_INT32);
  float obfCoefficient = 1.0;

  auto yOut = l0op::ObfuscationCalculate(fdTensor, x, paramTensor, cmd, obfCoefficient, uniqueExecutor.get());

  // 固定写法，将计算结果转换成输出y的数据类型
  auto castY = l0op::Cast(yOut, y->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castY != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出y上，y可能是非连续的tensor
  auto viewCopyY = l0op::ViewCopy(castY, y, uniqueExecutor.get());
  CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnObfuscationCalculate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnObfuscationCalculate);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif