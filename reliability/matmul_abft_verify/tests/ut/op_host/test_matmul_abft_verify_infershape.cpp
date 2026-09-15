/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_context_faker.h"

namespace {
constexpr int64_t L1_TILE_N = 256;
constexpr int64_t COMP_PACK = 8;
constexpr char OP_TYPE[] = "MatmulAbftVerify";

ge::graphStatus RunInferShape(const std::vector<std::vector<int64_t>>& inputDims,
                              const std::vector<ge::DataType>& inputTypes,
                              std::vector<int64_t>* outputDims)
{
    std::vector<gert::Shape> inputShapes;
    inputShapes.reserve(inputDims.size());
    for (const auto& dims : inputDims) {
        inputShapes.emplace_back();
        for (const int64_t dim : dims) {
            inputShapes.back().AppendDim(dim);
        }
    }
    std::vector<gert::Shape*> inputShapePtrs;
    inputShapePtrs.reserve(inputShapes.size());
    for (auto& shape : inputShapes) {
        inputShapePtrs.push_back(&shape);
    }

    auto faker = gert::InferShapeContextFaker();
    faker.SetOpType(OP_TYPE).NodeIoNum(inputShapes.size(), 1).InputShapes(inputShapePtrs);
    for (size_t i = 0; i < inputTypes.size(); ++i) {
        faker.NodeInputTd(i, inputTypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    faker.NodeOutputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND);

    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto* opImpl = registry->GetOpImpl(OP_TYPE);
    if (context == nullptr || opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::graphStatus ret = opImpl->infer_shape(context);
    if (ret == ge::GRAPH_SUCCESS && outputDims != nullptr) {
        const gert::Shape* outputShape = context->GetOutputShape(0);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        outputDims->clear();
        for (size_t i = 0; i < outputShape->GetDimNum(); ++i) {
            outputDims->push_back(outputShape->GetDim(i));
        }
    }
    return ret;
}

void ExpectSuccessfulInfer(int64_t m, int64_t n, int64_t k, ge::DataType precisionType)
{
    const std::vector<std::vector<int64_t>> inputs = {{m, k}, {k, n}, {m, n}, {n}};
    const std::vector<ge::DataType> types = {precisionType, precisionType, ge::DT_FLOAT, precisionType};
    std::vector<int64_t> outputDims;
    ASSERT_EQ(RunInferShape(inputs, types, &outputDims), ge::GRAPH_SUCCESS);

    const int64_t splitN = (n + L1_TILE_N - 1) / L1_TILE_N;
    const int64_t compRowLength = ((m + COMP_PACK - 1) / COMP_PACK) * splitN;
    EXPECT_EQ(outputDims, std::vector<int64_t>({compRowLength}));
}
} // namespace

TEST(MatmulAbftVerifyInfershape, infer_shape_fp16_single_n_tile)
{
    ExpectSuccessfulInfer(64, 128, 128, ge::DT_FLOAT16);
}

TEST(MatmulAbftVerifyInfershape, infer_shape_bf16_multi_n_tile)
{
    ExpectSuccessfulInfer(129, 513, 512, ge::DT_BF16);
}

TEST(MatmulAbftVerifyInfershape, infer_shape_fp32_large_n_tile)
{
    ExpectSuccessfulInfer(1024, 4097, 1024, ge::DT_FLOAT);
}

TEST(MatmulAbftVerifyInfershape, infer_shape_fails_when_k_mismatch)
{
    const std::vector<std::vector<int64_t>> inputs = {{64, 128}, {256, 128}, {64, 128}, {128}};
    const std::vector<ge::DataType> types(4, ge::DT_FLOAT16);
    EXPECT_EQ(RunInferShape(inputs, types, nullptr), ge::GRAPH_FAILED);
}

TEST(MatmulAbftVerifyInfershape, infer_shape_fails_when_a_is_not_rank2)
{
    const std::vector<std::vector<int64_t>> inputs = {{64, 16, 8}, {128, 128}, {64, 128}, {128}};
    const std::vector<ge::DataType> types(4, ge::DT_FLOAT16);
    EXPECT_EQ(RunInferShape(inputs, types, nullptr), ge::GRAPH_FAILED);
}
