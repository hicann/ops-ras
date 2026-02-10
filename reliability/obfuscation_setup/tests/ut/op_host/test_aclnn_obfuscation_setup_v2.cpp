/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "aclnn_obfuscation_setup_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

 using namespace std;
 
class l2_obfuscation_setup_v2_test : public testing::Test {
  protected:
    static void SetUpTestCase() {
      cout << "obfuscation_setup_v2_test SetUp" << endl; 
    }

    static void TearDownTestCase() {
      cout << "obfuscation_setup_v2_test TearDown" << endl; 
    }
};

// test success
TEST_F(l2_obfuscation_setup_v2_test, case_success) {
  int32_t fdToClose = 35;
  int32_t dataType = 1;
  int32_t hiddenSize = 4096;
  int32_t tpRank = 0;
  int32_t modelObfSeedId = 123456789;
  int32_t dataObfSeedId = 987654321;
  int32_t cmd = 1;
  int32_t threadNum = 4;
  float obfCoefficient = 0.5;
  auto fd_tensor_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnObfuscationSetupV2, 
                      INPUT(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, obfCoefficient), 
                      OUTPUT(fd_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// *** nullptr test ***
// test nullptr fd
TEST_F(l2_obfuscation_setup_v2_test, case_anullptr_y) {
  int32_t fdToClose = 40;
  int32_t dataType = 0;
  int32_t hiddenSize = 2048;
  int32_t tpRank = 0;
  int32_t modelObfSeedId = 987654321;
  int32_t dataObfSeedId = 123456789;
  int32_t cmd = 1;
  int32_t threadNum = 2;
  float obfCoefficient = 0.5;

  auto ut = OP_API_UT(aclnnObfuscationSetupV2, 
                      INPUT(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, obfCoefficient), 
                      OUTPUT((aclTensor*)nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// *** shape test ***
// test shape fd != 1
TEST_F(l2_obfuscation_setup_v2_test, case_shape_x_equal_y) {
  int32_t fdToClose = 40;
  int32_t dataType = 0;
  int32_t hiddenSize = 2048;
  int32_t tpRank = 0;
  int32_t modelObfSeedId = 987654321;
  int32_t dataObfSeedId = 123456789;
  int32_t cmd = 1;
  int32_t threadNum = 2;
  float obfCoefficient = 0.5;

  auto fd_tensor_desc = TensorDesc({2,3}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnObfuscationSetupV2, 
                      INPUT(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, obfCoefficient), 
                      OUTPUT(fd_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}