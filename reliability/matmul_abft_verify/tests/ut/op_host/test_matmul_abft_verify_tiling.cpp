/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "../../../op_kernel/matmul_abft_verify_tiling_data.h"
#include "../../../op_kernel/matmul_abft_verify_tiling_key.h"
#include "any_value.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t COMP_PACK = 8;
constexpr size_t SYSTEM_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr size_t WORKSPACE_ALIGNMENT = 32;

using TilingTensorDesc = gert::TilingContextPara::TensorDescription;
using TilingAttr = gert::TilingContextPara::OpAttr;

struct MatmulAbftVerifyCase {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    ge::DataType precisionType;
    float eMax;
};

struct MatmulAbftVerifyCompileInfo {};

std::vector<TilingTensorDesc> MakeInputDescs(const MatmulAbftVerifyCase& param)
{
    return {
        {{{param.m, param.k}, {param.m, param.k}}, param.precisionType, ge::FORMAT_ND},
        {{{param.k, param.n}, {param.k, param.n}}, param.precisionType, ge::FORMAT_ND},
        {{{param.m, param.n}, {param.m, param.n}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{param.n}, {param.n}}, param.precisionType, ge::FORMAT_ND},
    };
}

std::vector<TilingTensorDesc> MakeOutputDescs(const MatmulAbftVerifyCase& param)
{
    const uint32_t splitN = (param.n + L1_TILE_N - 1) / L1_TILE_N;
    return {
        {{{((param.m + COMP_PACK - 1) / COMP_PACK) * splitN},
          {((param.m + COMP_PACK - 1) / COMP_PACK) * splitN}}, ge::DT_UINT8, ge::FORMAT_ND},
    };
}

std::vector<TilingAttr> MakeAttrs(float eMax)
{
    return {{"e_max", Ops::Ras::AnyValue::CreateFrom<float>(eMax)}};
}

uint32_t CalcSplitReduceM(uint32_t splitN, int64_t reduceCores)
{
    uint32_t splitReduceM = (splitN + static_cast<uint32_t>(reduceCores) - 1) / static_cast<uint32_t>(reduceCores);
    splitReduceM = std::min<uint32_t>(splitReduceM, 8);
    return std::max<uint32_t>(splitReduceM, 2);
}

uint32_t CalcSplitReduceN(uint32_t k)
{
    return ((k + 255) / 256) < 2 ? (k + 1) / 2 : 256;
}

void AddTensor(size_t elements, size_t elementBytes, size_t& bytes)
{
    bytes = (bytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
    bytes += elements * elementBytes;
}

size_t CalcWorkspaceSize(const MatmulAbftVerifyCase& param, uint32_t splitN)
{
    const size_t rowSplitElements = static_cast<size_t>(param.m) * splitN;
    const size_t bStatElements = ((static_cast<size_t>(splitN) + 7) / 8) * 8 + 8;
    const size_t beElements = static_cast<size_t>(param.k) * splitN;
    const size_t precisionBytes = param.precisionType == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const size_t ceBytes = param.precisionType == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    size_t bytes = 0;
    for (uint32_t i = 0; i < 3; ++i) AddTensor(rowSplitElements, sizeof(float), bytes);
    for (uint32_t i = 0; i < 3; ++i) AddTensor(bStatElements, sizeof(float), bytes);
    AddTensor(beElements, precisionBytes, bytes);
    AddTensor(beElements, ceBytes, bytes);
    for (uint32_t i = 0; i < 2; ++i) AddTensor(beElements, sizeof(float), bytes);
    for (uint32_t i = 0; i < 3; ++i) AddTensor(param.m, sizeof(float), bytes);
    AddTensor(static_cast<size_t>(param.m) * (splitN + 1), sizeof(float), bytes);
    return SYSTEM_WORKSPACE_SIZE + bytes;
}

void ExpectCoord(const MatmulAbftVerifyGemmCoordTilingData& actual, uint32_t m, uint32_t n, uint32_t k)
{
    EXPECT_EQ(actual.m, m);
    EXPECT_EQ(actual.n, n);
    EXPECT_EQ(actual.k, k);
}

void ExpectCoord(const MatmulAbftVerifyGemvCoordTilingData& actual, uint32_t m, uint32_t n)
{
    EXPECT_EQ(actual.m, m);
    EXPECT_EQ(actual.n, n);
}

void ExpectTilingFields(const MatmulAbftVerifyCase& param)
{
    TilingInfo info;
    auto inputs = MakeInputDescs(param);
    auto outputs = MakeOutputDescs(param);
    auto attrs = MakeAttrs(param.eMax);
    MatmulAbftVerifyCompileInfo compileInfo = {};
    gert::TilingContextPara contextPara("MatmulAbftVerify", inputs, outputs, attrs, &compileInfo);
    ASSERT_TRUE(ExecuteTiling(contextPara, info));
    ASSERT_GE(info.tilingDataSize, sizeof(MatmulAbftVerifyTilingData));
    ASSERT_EQ(info.tilingKey, GET_TPL_TILING_KEY(MATMUL_ABFT_VERIFY_TPL_SCH_MODE_BF16));
    ASSERT_EQ(info.blockNum, 64);

    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    const uint32_t splitN = (param.n + L1_TILE_N - 1) / L1_TILE_N;
    EXPECT_EQ(info.workspaceSizes[0], CalcWorkspaceSize(param, splitN));

    MatmulAbftVerifyTilingData tiling = {};
    std::copy_n(reinterpret_cast<const unsigned char *>(info.tilingData.get()), sizeof(MatmulAbftVerifyTilingData),
        reinterpret_cast<unsigned char *>(&tiling));
    const auto& compute = tiling.compute;
    ExpectCoord(compute.problemGemmShape, param.m, param.n, param.k);
    ExpectCoord(compute.problemGemmShapeFirst, param.m, splitN, param.k);
    ExpectCoord(compute.problemGemmShapeRemain, param.m, param.n, param.k);
    ExpectCoord(compute.problemShape, param.m, param.n);
    ExpectCoord(compute.problemShapeCol, param.n, param.m);
    ExpectCoord(compute.problemCompShape, 1, param.m + param.n);
    ExpectCoord(compute.problemSliceShape, splitN, param.m);

    EXPECT_EQ(compute.totalInputElements, param.m + param.n);
    EXPECT_EQ(compute.rowOutputElements, (param.m + COMP_PACK - 1) / COMP_PACK);
    EXPECT_EQ(compute.colOutputElements, (param.n + COMP_PACK - 1) / COMP_PACK);
    EXPECT_EQ(compute.totalOutputElements, compute.rowOutputElements + compute.colOutputElements);
    EXPECT_EQ(compute.xLen, std::max(param.m, param.n));
    EXPECT_EQ(compute.layoutThreLen, param.m);
    EXPECT_EQ(compute.splitNNum, splitN);
    EXPECT_EQ(compute.splitReduceM, CalcSplitReduceM(splitN, 8));
    EXPECT_EQ(compute.splitReduceN, CalcSplitReduceN(param.k));
    EXPECT_EQ(compute.splitKNum, 1U);
    EXPECT_EQ(compute.outputThre, 1);
    EXPECT_EQ(compute.outputCE, 1);
    EXPECT_EQ(compute.outputWorkspace, 0);
    EXPECT_NEAR(compute.eMax, param.eMax * 6.0f * std::sqrt(static_cast<float>(param.k) / 1024.0f), 1e-6f);
}
} // namespace

class MatmulAbftVerifyTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAbftVerifyTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAbftVerifyTiling TearDown" << std::endl;
    }
};

TEST_F(MatmulAbftVerifyTiling, tiling_fp16_small_problem)
{
    ExpectTilingFields({64, 128, 128, ge::DT_FLOAT16, 0.001f});
}

TEST_F(MatmulAbftVerifyTiling, tiling_bf16_non_aligned_n)
{
    ExpectTilingFields({129, 513, 512, ge::DT_BF16, 0.002f});
}

TEST_F(MatmulAbftVerifyTiling, tiling_fp32_large_n)
{
    ExpectTilingFields({1024, 4097, 1024, ge::DT_FLOAT, 0.001f});
}

TEST_F(MatmulAbftVerifyTiling, tiling_fails_when_k_mismatch)
{
    MatmulAbftVerifyCase param = {64, 128, 128, ge::DT_FLOAT16, 0.001f};
    auto inputs = MakeInputDescs(param);
    auto outputs = MakeOutputDescs(param);
    auto attrs = MakeAttrs(param.eMax);
    inputs[1] = {{{256, param.n}, {256, param.n}}, param.precisionType, ge::FORMAT_ND};
    MatmulAbftVerifyCompileInfo compileInfo = {};
    gert::TilingContextPara contextPara("MatmulAbftVerify", inputs, outputs, attrs, &compileInfo);
    ExecuteTestCase(contextPara, ge::GRAPH_FAILED);
}
