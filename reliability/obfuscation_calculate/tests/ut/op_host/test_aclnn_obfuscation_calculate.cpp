/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "aclnn_obfuscation_calculate.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
 
class l2_obfuscation_calculate_test : public testing::Test {
  protected:
    static void SetUpTestCase() {
        cout << "obfuscation_calculate_test SetUp" << endl; 
    }

    static void TearDownTestCase() {
      cout << "obfuscation_calculate_test TearDown" << endl; }
};

// test success
TEST_F(l2_obfuscation_calculate_test, case_float_success) {
  int32_t fd = 35;
  int32_t param = 4;
  int32_t cmd = 1;
  auto x_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto y_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnObfuscationCalculate, INPUT(fd, x_tensor_desc, param, cmd), 
                      OUTPUT(y_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// *** nullptr test ***
// test nullptr x
TEST_F(l2_obfuscation_calculate_test, case_anullptr_x) {
  int32_t fd = 35;
  int32_t param = 4;
  int32_t cmd = 1;
  auto y_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnObfuscationCalculate, INPUT(fd, (aclTensor*)nullptr, param, cmd), 
                      OUTPUT(y_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr y
TEST_F(l2_obfuscation_calculate_test, case_anullptr_y) {
  auto x_tensor_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  int32_t fd = 35;
  int32_t param = 4;
  int32_t cmd = 1;

  auto ut = OP_API_UT(aclnnObfuscationCalculate, INPUT(fd, x_tensor_desc, param, cmd), OUTPUT((aclTensor*)nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test datatype x != y 
TEST_F(l2_obfuscation_calculate_test, case_datatype_x_equal_y) {
  auto x_tensor_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  int32_t fd = 35;
  int32_t param = 4;
  int32_t cmd = 1;

  auto y_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnObfuscationCalculate, INPUT(fd, x_tensor_desc, param, cmd), OUTPUT(y_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test shape x != y 
TEST_F(l2_obfuscation_calculate_test, case_shape_x_equal_y) {
  auto x_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  int32_t fd = 35;
  int32_t param = 4;
  int32_t cmd = 1;

  auto y_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnObfuscationCalculate, INPUT(fd, x_tensor_desc, param, cmd), OUTPUT(y_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}