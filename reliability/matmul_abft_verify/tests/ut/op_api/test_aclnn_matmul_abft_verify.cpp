/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "aclnn_matmul_abft_verify.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
constexpr int64_t kM = 64;
constexpr int64_t kN = 128;
constexpr int64_t kK = 128;
constexpr int64_t kCompRowElements = 8;
constexpr double kEMax = 0.001;

TensorDesc ADesc()
{
    return TensorDesc({kM, kK}, ACL_FLOAT16, ACL_FORMAT_ND);
}

TensorDesc BDesc()
{
    return TensorDesc({kK, kN}, ACL_FLOAT16, ACL_FORMAT_ND);
}

TensorDesc CDesc()
{
    return TensorDesc({kM, kN}, ACL_FLOAT, ACL_FORMAT_ND);
}

TensorDesc ChecksumWeightDesc()
{
    return TensorDesc({kN}, ACL_FLOAT16, ACL_FORMAT_ND);
}

TensorDesc OutputDesc()
{
    return TensorDesc({kCompRowElements}, ACL_UINT8, ACL_FORMAT_ND);
}
} // namespace

TEST(MatmulAbftVerifyOpApi, rejects_null_a)
{
    auto ut = OP_API_UT(aclnnMatmulAbftVerify,
                        INPUT(static_cast<aclTensor *>(nullptr), BDesc(), CDesc(), ChecksumWeightDesc(), kEMax),
                        OUTPUT(OutputDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(MatmulAbftVerifyOpApi, rejects_null_b)
{
    auto ut = OP_API_UT(aclnnMatmulAbftVerify,
                        INPUT(ADesc(), static_cast<aclTensor *>(nullptr), CDesc(), ChecksumWeightDesc(), kEMax),
                        OUTPUT(OutputDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(MatmulAbftVerifyOpApi, rejects_null_c)
{
    auto ut = OP_API_UT(aclnnMatmulAbftVerify,
                        INPUT(ADesc(), BDesc(), static_cast<aclTensor *>(nullptr), ChecksumWeightDesc(), kEMax),
                        OUTPUT(OutputDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(MatmulAbftVerifyOpApi, rejects_null_checksum_weight)
{
    auto ut = OP_API_UT(aclnnMatmulAbftVerify,
                        INPUT(ADesc(), BDesc(), CDesc(), static_cast<aclTensor *>(nullptr), kEMax),
                        OUTPUT(OutputDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(MatmulAbftVerifyOpApi, rejects_null_output)
{
    auto ut = OP_API_UT(aclnnMatmulAbftVerify,
                        INPUT(ADesc(), BDesc(), CDesc(), ChecksumWeightDesc(), kEMax),
                        OUTPUT(static_cast<aclTensor *>(nullptr)));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}
