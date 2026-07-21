/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "aclnn_crypto.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_crypto_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "crypto_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "crypto_test TearDown" << endl;
    }

    TensorDesc GetKeyTensorDesc() const
    {
        return TensorDesc({16}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetKeyTensorDesc(aclDataType dataType) const
    {
        return TensorDesc({16}, dataType, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetInputTensorDesc() const
    {
        return TensorDesc({32}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetInputTensorDesc(aclDataType dataType) const
    {
        return TensorDesc({32}, dataType, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetInputTensorDesc(aclDataType dataType, const std::vector<int64_t>& shape) const
    {
        return TensorDesc(shape, dataType, ACL_FORMAT_ND)
            .ValueRange(0.0, 255.0);
    }

    TensorDesc GetOutputTensorDesc() const
    {
        return TensorDesc({32}, ACL_UINT8, ACL_FORMAT_ND);
    }

    TensorDesc GetOutputTensorDesc(aclDataType dataType) const
    {
        return TensorDesc({32}, dataType, ACL_FORMAT_ND);
    }

    TensorDesc GetOutputTensorDesc(aclDataType dataType, const std::vector<int64_t>& shape) const
    {
        return TensorDesc(shape, dataType, ACL_FORMAT_ND)
            .ValueRange(0.0, 255.0);
    }

    TensorDesc GetIvTensorDesc() const
    {
        return TensorDesc({12}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetIvTensorDesc(aclDataType dataType) const
    {
        return TensorDesc({12}, dataType, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetOpConfig() const
    {
        return TensorDesc({16}, ACL_UINT32, ACL_FORMAT_ND);
    }

    TensorDesc GetOpConfig(aclDataType dataType) const
    {
        return TensorDesc({16}, dataType, ACL_FORMAT_ND);
    }

    TensorDesc GetTagTensorDesc() const
    {
        return TensorDesc({16}, ACL_UINT8, ACL_FORMAT_ND);
    }

    TensorDesc GetTagTensorDesc(aclDataType dataType) const
    {
        return TensorDesc({16}, dataType, ACL_FORMAT_ND);
    }

    TensorDesc GetAadTensorDesc() const
    {
        return TensorDesc({16}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetAadTensorDesc(aclDataType dataType) const
    {
        return TensorDesc({16}, dataType, ACL_FORMAT_ND).ValueRange(0.0, 255.0);
    }

    TensorDesc GetYTensorDesc() const
    {
        return TensorDesc({1}, ACL_UINT32, ACL_FORMAT_ND);
    }

    TensorDesc GetYTensorDesc(aclDataType dataType) const
    {
        return TensorDesc({1}, dataType, ACL_FORMAT_ND);
    }
};

TEST_F(l2_crypto_test, case_success)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              GetOpConfig(), GetTagTensorDesc(), nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_crypto_test, case_nullptr_key)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT((aclTensor *)nullptr, GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              GetOpConfig(), GetTagTensorDesc(), nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_crypto_test, case_nullptr_op_config)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              (aclTensor *)nullptr, GetTagTensorDesc(), nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_crypto_test, case_nullptr_y)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              GetOpConfig(), GetTagTensorDesc(), nullptr),
                        OUTPUT((aclTensor *)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_crypto_test, case_key_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(ACL_INT8), GetInputTensorDesc(), GetOutputTensorDesc(),
                              GetIvTensorDesc(), GetOpConfig(), GetTagTensorDesc(),
                              nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_input_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(ACL_BOOL), GetOutputTensorDesc(),
                              GetIvTensorDesc(), GetOpConfig(), GetTagTensorDesc(),
                              nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_output_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(ACL_BOOL),
                              GetIvTensorDesc(), GetOpConfig(), GetTagTensorDesc(),
                              nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_iv_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(),
                              GetIvTensorDesc(ACL_INT8), GetOpConfig(), GetTagTensorDesc(),
                              nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_op_config_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              GetOpConfig(ACL_INT32), GetTagTensorDesc(), nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_tag_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              GetOpConfig(), GetTagTensorDesc(ACL_INT8), nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_aad_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              GetOpConfig(), GetTagTensorDesc(), GetAadTensorDesc(ACL_INT8)),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_y_dtype_invalid)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(), GetOutputTensorDesc(), GetIvTensorDesc(),
                              GetOpConfig(), GetTagTensorDesc(), nullptr),
                        OUTPUT(GetYTensorDesc(ACL_INT32)));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_crypto_test, case_supported_data_dtypes)
{
    const std::vector<aclDataType> dataTypes = {
        ACL_UINT8, ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_DOUBLE};
    for (auto dtype : dataTypes) {
        auto ut = OP_API_UT(aclnnCrypto,
                            INPUT(GetKeyTensorDesc(), GetInputTensorDesc(dtype), GetOutputTensorDesc(dtype),
                                  GetIvTensorDesc(), GetOpConfig(), GetTagTensorDesc(), GetAadTensorDesc()),
                            OUTPUT(GetYTensorDesc()));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_crypto_test, case_shape_dim_num_mismatch)
{
    auto ut = OP_API_UT(aclnnCrypto,
                        INPUT(GetKeyTensorDesc(), GetInputTensorDesc(ACL_UINT8, {2, 16}),
                              GetOutputTensorDesc(ACL_UINT8, {32}), GetIvTensorDesc(),
                              GetOpConfig(), GetTagTensorDesc(), nullptr),
                        OUTPUT(GetYTensorDesc()));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
