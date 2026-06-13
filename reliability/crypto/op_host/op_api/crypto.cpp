/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "crypto.h"

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

OP_TYPE_REGISTER(Crypto);

static const aclTensor *CryptoAiCpu(
    const aclTensor* key,
    const aclTensor* inputText,
    aclTensor* outputText,
    const aclTensor* iv,
    const aclTensor* opConfig,
    aclTensor* tag,
    aclTensor* aad,
    aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(CryptoAiCpu, key, inputText, outputText, iv, opConfig, tag, aad, out);
    static internal::AicpuTaskSpace space("Crypto");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Crypto,
        OP_ATTR_NAMES({}),
        OP_INPUT(key, inputText, outputText, iv, opConfig, tag, aad),
        OP_OUTPUT(out),
        OP_ATTR());
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CryptoAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
    }

    return out;
}

const aclTensor *Crypto(
    const aclTensor* key,
    const aclTensor* inputText,
    aclTensor* outputText,
    const aclTensor* iv,
    const aclTensor* opConfig,
    aclTensor* tag,
    aclTensor* aad,
    aclOpExecutor* executor)
{
    L0_DFX(Crypto, key, inputText, outputText, iv, opConfig, tag, aad);
    auto out = executor->AllocTensor({1}, op::DataType::DT_INT32);
    return CryptoAiCpu(key, inputText, outputText, iv, opConfig, tag, aad, out, executor);
}

}  // namespace l0op
