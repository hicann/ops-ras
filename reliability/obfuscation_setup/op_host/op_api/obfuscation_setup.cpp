/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "obfuscation_setup.h"

#include "opdev/op_log.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/aicpu/aicpu_task.h"

#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op
{

OP_TYPE_REGISTER(ObfuscationSetup);

// AICPU算子kernel
static const aclTensor *ObfuscationSetupAiCpu(const aclTensor* fdToClose, const aclTensor* dataType, 
                                              const aclTensor* hiddenSize, const aclTensor* tpRank, 
                                              const aclTensor* modelObfSeedId, const aclTensor* dataObfSeedId, 
                                              int32_t cmd, int32_t threadNum, float obfCoefficient,
                                              aclTensor* fd, aclOpExecutor* executor) {
    L0_DFX(ObfuscationSetupAiCpu, fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,
           dataObfSeedId, cmd, threadNum, obfCoefficient, fd);
    static internal::AicpuTaskSpace space("ObfuscationSetup");
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICPU(ObfuscationSetup, OP_ATTR_NAMES({"cmd", "thread_num", "obf_coefficient"}),
                                   OP_INPUT(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId, dataObfSeedId),
                                   OP_OUTPUT(fd), OP_ATTR(cmd, threadNum, obfCoefficient));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ObfuscationSetupAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
    }
    return fd;
}

const aclTensor *ObfuscationSetup(const aclTensor* fdToClose, const aclTensor* dataType,
                                  const aclTensor* hiddenSize, const aclTensor* tpRank, 
                                  const aclTensor* modelObfSeedId, const aclTensor* dataObfSeedId,
                                  int32_t cmd, int32_t threadNum, float obfCoefficient, aclOpExecutor* executor)
{
    L0_DFX(ObfuscationSetup, fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId, dataObfSeedId);

    auto fd = executor->AllocTensor({1}, op::DataType::DT_INT32);
    return ObfuscationSetupAiCpu(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId, 
                                dataObfSeedId, cmd, threadNum, obfCoefficient, fd, executor);
}

}  // namespace l0op