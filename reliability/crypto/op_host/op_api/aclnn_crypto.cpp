/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_crypto.h"

#include "crypto.h"
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

static const std::initializer_list<op::DataType> DATA_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,   op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> PARAM_UINT8_DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> PARAM_UINT32_DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT32};

static bool CheckNotNull(const aclTensor* key, const aclTensor* opConfig, aclTensor* out)
{
    OP_CHECK_NULL(key, return false);
    OP_CHECK_NULL(opConfig, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValidAllowNull(const aclTensor* tensor, const std::initializer_list<op::DataType> supportList)
{
    if (tensor == nullptr) {
        return true;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(tensor, supportList, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* key, const aclTensor* inputText, aclTensor* outputText,
                            const aclTensor* iv, const aclTensor* opConfig, aclTensor* tag, aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(key, PARAM_UINT8_DTYPE_SUPPORT_LIST, return false);
    if (!CheckDtypeValidAllowNull(inputText, DATA_DTYPE_SUPPORT_LIST)) {
        return false;
    }

    if (!CheckDtypeValidAllowNull(outputText, DATA_DTYPE_SUPPORT_LIST)) {
        return false;
    }

    if (!CheckDtypeValidAllowNull(iv, PARAM_UINT8_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(opConfig, PARAM_UINT32_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, PARAM_UINT32_DTYPE_SUPPORT_LIST, return false);
    if (!CheckDtypeValidAllowNull(tag, PARAM_UINT8_DTYPE_SUPPORT_LIST)) {
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* inputText, const aclTensor* outputText)
{
    if (inputText == nullptr && outputText == nullptr) {
        return true;
    }
    if (inputText == nullptr || outputText == nullptr) {
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(inputText, outputText, return false);
    return true;
}

static bool MakeContiguous(const aclTensor* tensor, const char* tensorName, aclOpExecutor* executor)
{
    if (tensor == nullptr || tensor->IsEmpty()) {
        return true;
    }
    auto contiguous = l0op::Contiguous(tensor, executor);
    OP_CHECK(contiguous != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AclnnCrypto %s contiguous is nullptr", tensorName),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* key, const aclTensor* inputText, aclTensor* outputText,
                               const aclTensor* iv, const aclTensor* opConfig, aclTensor* tag, aclTensor* aad,
                               aclTensor* out)
{
    CHECK_RET(CheckNotNull(key, opConfig, out), ACLNN_ERR_PARAM_NULLPTR);
    // aad 当前仅支持 NULL
    if (aad != NULL) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(
        CheckDtypeValid(key, inputText, outputText, iv, opConfig, tag, out),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(inputText, outputText), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCryptoGetWorkspaceSize(
    const aclTensor* key, const aclTensor* inputText, aclTensor* outputText, const aclTensor* iv,
    const aclTensor* opConfig, aclTensor* tag, aclTensor* aad, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnCrypto, DFX_IN(key, inputText, outputText, iv, opConfig, tag, aad), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(key, inputText, outputText, iv, opConfig, tag, aad, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(MakeContiguous(inputText, "inputTextContiguous", uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(outputText, "outputTextContiguous", uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(key, "key", uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(iv, "iv", uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(tag, "tag", uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(opConfig, "opConfig", uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(aad, "aad", uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    auto outOut = l0op::Crypto(key, inputText, outputText, iv, opConfig, tag, aad, uniqueExecutor.get());
    auto castOut = l0op::Cast(outOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyOut = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCrypto(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCrypto);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
