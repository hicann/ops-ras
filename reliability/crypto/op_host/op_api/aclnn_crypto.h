/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_CRYPTO_H_
#define OP_API_INC_CRYPTO_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCryptoworkspace
 * @domain aclnn_ops_infer
 *
 * AES-GCM
 *
 * @param [in] key: npu device aclTensor shape(keyLen, )INT8 ND
 * @param [in] inputText: npu device aclTensor  shape(inputLen, )INT8 ND
 * @param [in] outputText: npu device aclTensor buffershape(inputLen, )INT8 ND
 * @param [in] iv: npu device aclTensor IV INT8 ND
 * @param [in] opConfig: npu deviceconst aclTensor* opConfig/
 * @param [in] tag: npu device aclTensor GCM Tag INT8 ND
 * @param [in] aad: npu device aclTensor AAD Additional Authenticated Data INT8 ND
 * @param [out] out: npu device aclTensor INT8 ND
 * @param [out] workspaceSize: npu device workspace
 * @param [out] executor: op
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnCryptoGetWorkspaceSize(
    const aclTensor* key, const aclTensor* inputText, aclTensor* outputText, const aclTensor* iv,
    const aclTensor* opConfig, aclTensor* tag, aclTensor* aad, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnCrypto
 *
 * @param [in] workspace: npu device workspace
 * @param [in] workspaceSize: npu device workspace aclnnCryptoGetWorkspaceSize
 * @param [in] executor: op
 * @param [in] stream: acl stream
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnCrypto(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_CRYPTO_H_