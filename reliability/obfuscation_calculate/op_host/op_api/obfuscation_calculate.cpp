/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "obfuscation_calculate.h"

#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/aicpu/aicpu_task.h"

#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(ObfuscationCalculate);

// AICPU算子kernel
static const aclTensor *ObfuscationCalculateAiCpu(const aclTensor* fd, const aclTensor* x, const aclTensor* param,
                                                  int32_t cmd, float obfCoefficient, 
                                                  aclTensor* y,aclOpExecutor* executor) {
  L0_DFX(ObfuscationCalculateAiCpu, fd, x, param, cmd, obfCoefficient, y);
  static internal::AicpuTaskSpace space("ObfuscationCalculate");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ObfuscationCalculate, OP_ATTR_NAMES({"cmd", "obf_coefficient"}), 
                                        OP_INPUT(fd, x, param), OP_OUTPUT(y), OP_ATTR(cmd, obfCoefficient));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ObfuscationCalculateAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
  }

  return y;
}

const aclTensor *ObfuscationCalculate(const aclTensor* fd, const aclTensor* x, const aclTensor* param,
                                      int32_t cmd, float obfCoefficient, aclOpExecutor* executor) {
  L0_DFX(ObfuscationCalculate, fd, x, param);
  auto y = executor->AllocTensor(x ->GetViewShape(), x->GetDataType());
  return ObfuscationCalculateAiCpu(fd, x, param, cmd, obfCoefficient, y, executor);
}

}  // namespace l0op