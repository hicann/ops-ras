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
#include "infer_shape_context_faker.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"

class CryptoInferShapeTest : public testing::Test {
protected:
    void RunInferShapeTest(gert::Shape &inputShape)
    {
        auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Crypto")->infer_shape;
        ASSERT_NE(inferShapeFunc, nullptr);

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(7, 1)
                          .IrInstanceNum({7, 1})
                          .InputShapes({&inputShape})
                          .NodeInputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_UINT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto* context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
        const gert::Shape* actualShape = context->GetOutputShape(0);
        ASSERT_NE(actualShape, nullptr);
        EXPECT_EQ(actualShape->GetDimNum(), inputShape.GetDimNum());
        for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
            EXPECT_EQ(actualShape->GetDim(i), inputShape.GetDim(i));
        }
    }
};

TEST_F(CryptoInferShapeTest, crypto_infershape_copies_input_shape)
{
    gert::Shape inputShape = {2, 3, 4};
    RunInferShapeTest(inputShape);
}

TEST_F(CryptoInferShapeTest, crypto_infershape_supports_empty_shape)
{
    gert::Shape inputShape = {};
    RunInferShapeTest(inputShape);
}
