/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_abft_verify_tiling.cpp
 * \brief Host-side tiling for MatmulAbftVerify.
 */
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/matmul_abft_verify_tiling_data.h"
#include "../op_kernel/matmul_abft_verify_tiling_key.h"

namespace optiling {
namespace {
constexpr size_t SYSTEM_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t WORKSPACE_ALIGNMENT = 32;
constexpr uint64_t FLOAT_BYTES = sizeof(float);

bool AddWorkspaceTensor(uint64_t elements, uint64_t elementBytes, uint64_t& workspaceBytes)
{
    if (workspaceBytes > std::numeric_limits<uint64_t>::max() - (WORKSPACE_ALIGNMENT - 1)) {
        return false;
    }
    workspaceBytes = (workspaceBytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
    if (elements != 0 && elementBytes > std::numeric_limits<uint64_t>::max() / elements) {
        return false;
    }
    const uint64_t tensorBytes = elements * elementBytes;
    if (workspaceBytes > std::numeric_limits<uint64_t>::max() - tensorBytes) {
        return false;
    }
    workspaceBytes += tensorBytes;
    return true;
}

uint32_t FloatBits(float value)
{
    uint32_t bits;
    std::copy_n(reinterpret_cast<const unsigned char *>(&value), sizeof(value),
        reinterpret_cast<unsigned char *>(&bits));
    return bits;
}

uint16_t FloatToBf16Bits(float value)
{
    const uint32_t bits = FloatBits(value);
    return static_cast<uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

uint16_t FloatToFp16Bits(float value)
{
    const uint32_t bits = FloatBits(value);
    const int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent < -24) {
        return 0;
    }
    if (exponent < -14) {
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(-exponent - 1);
        const uint32_t rounded = mantissa + ((1U << (shift - 1U)) - 1U) + ((mantissa >> shift) & 1U);
        return static_cast<uint16_t>(rounded >> shift);
    }
    uint32_t halfExponent = static_cast<uint32_t>(exponent + 15);
    mantissa += 0xFFFU + ((mantissa >> 13U) & 1U);
    if ((mantissa & 0x800000U) != 0U) {
        ++halfExponent;
        mantissa = 0;
    }
    return static_cast<uint16_t>((halfExponent << 10U) | (mantissa >> 13U));
}
}

template <typename Policy>
MatmulAbftVerifyComputeTilingData BuildComputeTiling(uint32_t m, uint32_t n, uint32_t k,
    float eMax, uint32_t reduceCores, uint32_t splitKs)
{
    // DIVISOR_GUARDED[585]: keep the reduction divisor non-zero for all caller inputs.
    const uint32_t safeReduceCores = reduceCores == 0U ? 1U : reduceCores;
    const uint32_t splitNNum = (n + Policy::L1TileShape_N - 1) / Policy::L1TileShape_N;
    const uint32_t firstBlockN = splitNNum;
    const uint32_t remainBlockN = n;

    const uint32_t totalInputElements = m + n;
    const uint32_t totalOutputElements = (m + 7) / 8 + (n + 7) / 8;
    const uint32_t rowOutputElements = (m + 7) / 8;
    const uint32_t colOutputElements = (n + 7) / 8;
    const uint32_t xLen = m > n ? m : n;

    uint32_t splitReduceM = (splitNNum + safeReduceCores - 1U) / safeReduceCores;
    splitReduceM = splitReduceM >= Policy::UBTileShapeforBRed_M ? Policy::UBTileShapeforBRed_M : splitReduceM;
    splitReduceM = splitReduceM < 2 ? 2 : splitReduceM;

    const uint32_t splitReduceNNum = (k + Policy::UBTileShapeforBRed_N - 1) / Policy::UBTileShapeforBRed_N;
    uint32_t splitReduceN = Policy::UBTileShapeforBRed_N;
    if (splitReduceNNum < 2) {
        splitReduceN = (k + 1) / 2;
    }

    const uint32_t nRemainSplit = n % Policy::L1TileShape_N;
    const float commonSize = static_cast<float>(k) * Policy::L1TileShape_N;
    const float remainSize = static_cast<float>(k) * nRemainSplit;

    const float commonStdFactor = std::sqrt(2.0f * std::log(commonSize));
    float remainStdFactor = commonStdFactor;
    float remainKnRatio = commonSize;
    float remainKnSqrtRatio = std::sqrt(commonSize);
    float remainKSqrtNRatio = std::sqrt(static_cast<float>(k)) * Policy::L1TileShape_N;
    if (nRemainSplit > 0) {
        remainStdFactor = std::sqrt(2.0f * std::log(remainSize));
        remainKnRatio = remainSize;
        remainKnSqrtRatio = std::sqrt(static_cast<float>(nRemainSplit));
        remainKSqrtNRatio = std::sqrt(static_cast<float>(k)) * nRemainSplit;
    }

    MatmulAbftVerifyComputeTilingData tiling{};
    tiling.problemGemmShape = {m, n, k};
    tiling.problemGemmShapeFirst = {m, firstBlockN, k};
    tiling.problemGemmShapeRemain = {m, remainBlockN, k};
    tiling.problemShape = {m, n};
    tiling.problemShapeCol = {n, m};
    tiling.problemCompShape = {1, totalInputElements};
    tiling.problemSliceShape = {splitNNum, m};
    tiling.totalInputElements = totalInputElements;
    tiling.totalOutputElements = totalOutputElements;
    tiling.rowOutputElements = rowOutputElements;
    tiling.colOutputElements = colOutputElements;
    tiling.xLen = xLen;
    tiling.layoutThreLen = m;
    tiling.roundingAlpha = 1.0f;
    tiling.eMax = eMax * 6 * std::sqrt(static_cast<float>(k) / 1024.0f);
    tiling.stdEstARowRatio = 1.0f / std::sqrt(2.0f * std::log(static_cast<float>(k)));
    tiling.aRowScaleRatio = 1.0f / static_cast<float>(k);

    tiling.stdEstRatios[0] = commonStdFactor == 0.0f ? 0.0f : 1.0f / commonStdFactor;
    tiling.stdEstRatios[1] = remainStdFactor == 0.0f ? 0.0f : 1.0f / remainStdFactor;
    tiling.knRatios[0] = commonSize;
    tiling.knRatios[1] = remainKnRatio;
    tiling.knScaleRatios[0] = 1.0f / commonSize;
    tiling.knScaleRatios[1] = 1.0f / remainKnRatio;
    tiling.knSqrtRatios[0] = std::sqrt(commonSize);
    tiling.knSqrtRatios[1] = remainKnSqrtRatio;
    tiling.kSqrtNRatios[0] = std::sqrt(static_cast<float>(k)) * Policy::L1TileShape_N;
    tiling.kSqrtNRatios[1] = remainKSqrtNRatio;

    tiling.nScaleRatios[0] = 1.0f / Policy::L1TileShape_N;
    tiling.nRatios[0] = static_cast<float>(Policy::L1TileShape_N);
    tiling.nSqrtRatios[0] = std::sqrt(static_cast<float>(Policy::L1TileShape_N));
    tiling.nSquareRatios[0] = static_cast<float>(Policy::L1TileShape_N * Policy::L1TileShape_N);
    const uint32_t remainN = nRemainSplit > 0 ? nRemainSplit : Policy::L1TileShape_N;
    tiling.nScaleRatios[1] = 1.0f / remainN;
    tiling.nRatios[1] = static_cast<float>(remainN);
    tiling.nSqrtRatios[1] = std::sqrt(static_cast<float>(remainN));
    tiling.nSquareRatios[1] = static_cast<float>(remainN * remainN);

    tiling.splitNNum = splitNNum;
    tiling.splitReduceM = splitReduceM;
    tiling.splitReduceN = splitReduceN;
    tiling.splitKNum = splitKs;
    tiling.outputThre = 1;
    tiling.outputCE = 1;
    tiling.outputWorkspace = 0;
    return tiling;
}


static ge::graphStatus MatmulAbftVerifyTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bShape);

    const gert::Shape& aStorageShape = aShape->GetStorageShape();
    const gert::Shape& bStorageShape = bShape->GetStorageShape();
    OP_CHECK_IF(aStorageShape.GetDimNum() != 2 || bStorageShape.GetDimNum() != 2,
        OP_LOGE(context, "MatmulAbftVerify expects rank-2 A and B"), return ge::GRAPH_FAILED);

    const int64_t m = aStorageShape.GetDim(0);
    const int64_t k = aStorageShape.GetDim(1);
    const int64_t bK = bStorageShape.GetDim(0);
    const int64_t n = bStorageShape.GetDim(1);
    OP_CHECK_IF(m <= 0 || n <= 0 || k <= 0 || k != bK,
        OP_LOGE(context, "invalid MatmulAbftVerify problem shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(m > UINT32_MAX || n > UINT32_MAX || k > UINT32_MAX,
        OP_LOGE(context, "MatmulAbftVerify shape exceeds uint32 range"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    uint32_t idx = 0;
    const float* eMaxAttr = attrs->GetAttrPointer<float>(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context, eMaxAttr);

    const float eMax = *eMaxAttr;
    const uint32_t splitKs = 1;
    const auto* aDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, aDesc);
    const ge::DataType aDtype = aDesc->GetDataType();

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0,
        OP_LOGE(context, "invalid AIC core count"), return ge::GRAPH_FAILED);
    const uint32_t reduceCores = std::min(aicNum, 8U);

    MatmulAbftVerifyTilingData* tiling = context->GetTilingData<MatmulAbftVerifyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (aDtype == ge::DT_FLOAT16) {
        tiling->compute = BuildComputeTiling<MatmulAbftVerifyFp16TilingPolicy>(
            static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            eMax, reduceCores, splitKs);
    } else if (aDtype == ge::DT_FLOAT) {
        tiling->compute = BuildComputeTiling<MatmulAbftVerifyFp32TilingPolicy>(
            static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            eMax, reduceCores, splitKs);
    } else {
        tiling->compute = BuildComputeTiling<MatmulAbftVerifyBf16TilingPolicy>(
            static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            eMax, reduceCores, splitKs);
    }
    const float aMeanFactor = 1.0f / static_cast<float>(k);
    if (aDtype == ge::DT_FLOAT16) {
        tiling->compute.aMeanFactorBits = FloatToFp16Bits(aMeanFactor);
    } else if (aDtype == ge::DT_FLOAT) {
        tiling->compute.aMeanFactorBits = FloatBits(aMeanFactor);
    } else {
        tiling->compute.aMeanFactorBits = FloatToBf16Bits(aMeanFactor);
    }

    const uint64_t splitN = tiling->compute.splitNNum;
    const uint64_t rowSplitElements = static_cast<uint64_t>(m) * splitN;
    const uint64_t bStatElements = ((splitN + 7) / 8) * 8 + 8;
    const uint64_t beElements = static_cast<uint64_t>(k) * splitN;
    const uint64_t precisionBytes = aDtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t ceBytes = aDtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);

    uint64_t userWorkspaceBytes = 0;
    bool workspaceValid = true;
    // The order is part of the workspace ABI and must match matmul_abft_verify.cpp.
    for (uint32_t i = 0; i < 3; ++i) { // z_row, d_row, threshold
        workspaceValid &= AddWorkspaceTensor(rowSplitElements, FLOAT_BYTES, userWorkspaceBytes);
    }
    for (uint32_t i = 0; i < 3; ++i) { // b_mean_abs, b_mean_square, b_var
        workspaceValid &= AddWorkspaceTensor(bStatElements, FLOAT_BYTES, userWorkspaceBytes);
    }
    workspaceValid &= AddWorkspaceTensor(beElements, precisionBytes, userWorkspaceBytes); // be
    workspaceValid &= AddWorkspaceTensor(beElements, ceBytes, userWorkspaceBytes); // be_for_aiv
    for (uint32_t i = 0; i < 2; ++i) { // b_max_slice, b_min_slice
        workspaceValid &= AddWorkspaceTensor(beElements, FLOAT_BYTES, userWorkspaceBytes);
    }
    for (uint32_t i = 0; i < 3; ++i) { // a_max, a_mean, a_min
        workspaceValid &= AddWorkspaceTensor(m, FLOAT_BYTES, userWorkspaceBytes);
    }
    const uint64_t internalWorkspaceElements =
        static_cast<uint64_t>(m) * (splitN + 1) * tiling->compute.splitKNum;
    workspaceValid &= AddWorkspaceTensor(internalWorkspaceElements, FLOAT_BYTES, userWorkspaceBytes);

    OP_CHECK_IF(!workspaceValid || userWorkspaceBytes >
            std::numeric_limits<size_t>::max() - SYSTEM_WORKSPACE_SIZE,
        OP_LOGE(context, "MatmulAbftVerify workspace size overflow"), return ge::GRAPH_FAILED);
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = SYSTEM_WORKSPACE_SIZE + static_cast<size_t>(userWorkspaceBytes);

    context->SetBlockDim(aicNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(MATMUL_ABFT_VERIFY_TPL_SCH_MODE_BF16));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulAbftVerify).Tiling(MatmulAbftVerifyTilingFunc);
} // namespace optiling
