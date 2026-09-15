/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP_ROBUST
#define FTSELF_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP_ROBUST

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../helper/type_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

FTSELF_DECLARE_TILE_THRESHOLD_ATLAS_SPECIALIZATION(
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST)

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;

    FTSELF_DEVICE
    TileThreCalc() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> threWorkspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY nRatioFactor, ElementY nSqrtRatioFactor,
        ElementY nSquareRatioFactor, ElementY bSliceMeanAbs,
        ElementY bSliceMeanSquare, ElementY bSliceVar,
        ElementY bSliceVarSquare, ElementY eMax)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t workspaceStride = mActual;
        if constexpr (std::is_same_v<ElementX, float> && std::is_same_v<ElementY, float>) {
            workspaceStride = FTSelf::helper::CeilDiv(mActual, ELE_NUM_PER_C0) * ELE_NUM_PER_C0;
        }

        FTSelf::Gemv::helper::ZeroTensor(threWorkspace, static_cast<int32_t>(workspaceStride) * 5);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::MulAddDst<ElementY, ElementX>(
            threWorkspace[2 * workspaceStride], srcStdTensor, srcStdTensor, mActual);
        AscendC::Abs(srcMeanTensor, srcMeanTensor, static_cast<int32_t>(mActual));
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::MulAddDst<ElementY, ElementX>(
            threWorkspace[3 * workspaceStride], srcMeanTensor, srcMeanTensor, mActual);
        FTSelf::Gemv::helper::VectorBarrier();

        if constexpr (!std::is_same_v<ElementX, ElementY>) {
            AscendC::Cast<ElementY, ElementX>(
                threWorkspace[4 * workspaceStride], srcMeanTensor,
                AscendC::RoundMode::CAST_NONE, mActual);
        }

        ElementY meanAbsScale;
        ElementY meanSquareScale;
        ElementY stdSquareScale;
        ElementY stdScale;
        if constexpr (std::is_same_v<ElementY, half>) {
            meanAbsScale = static_cast<ElementY>(static_cast<float>(nRatioFactor) *
                static_cast<float>(bSliceMeanAbs));
            meanSquareScale = static_cast<ElementY>(16.0f * static_cast<float>(nRatioFactor) *
                static_cast<float>(bSliceVarSquare));
            stdSquareScale = static_cast<ElementY>(16.0f * static_cast<float>(nSquareRatioFactor) *
                static_cast<float>(bSliceMeanSquare));
            stdScale = static_cast<ElementY>(static_cast<float>(nSqrtRatioFactor) *
                static_cast<float>(bSliceVar));
        } else {
            meanAbsScale = nRatioFactor * bSliceMeanAbs;
            meanSquareScale = 16.0f * nRatioFactor * bSliceVarSquare;
            stdSquareScale = 16.0f * nSquareRatioFactor * bSliceMeanSquare;
            stdScale = nSqrtRatioFactor * bSliceVar;
        }

        FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
        if constexpr (std::is_same_v<ElementX, ElementY>) {
            AscendC::Muls<ElementY, false>(
                threWorkspace[4 * workspaceStride], srcMeanTensor, meanAbsScale,
                AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        } else {
            AscendC::Muls<ElementY, false>(
                threWorkspace[4 * workspaceStride], threWorkspace[4 * workspaceStride], meanAbsScale,
                AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        }
        AscendC::Muls<ElementY, false>(
            threWorkspace[3 * workspaceStride], threWorkspace[3 * workspaceStride], meanSquareScale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        AscendC::Muls<ElementY, false>(
            threWorkspace[2 * workspaceStride], threWorkspace[2 * workspaceStride], stdSquareScale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        AscendC::Muls<ElementY, false>(
            threWorkspace[workspaceStride], srcStdTensor, stdScale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
        FTSelf::Gemv::helper::VectorBarrier();

        AscendC::Add<ElementY>(threWorkspace, threWorkspace[3 * workspaceStride],
            threWorkspace[2 * workspaceStride], mActual);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Abs(threWorkspace, threWorkspace, mActual);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Sqrt(threWorkspace, threWorkspace, mActual);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Add<ElementY>(threWorkspace, threWorkspace[4 * workspaceStride],
            threWorkspace, mActual);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Add<ElementY>(threWorkspace, threWorkspace[workspaceStride],
            threWorkspace, mActual);
        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
        AscendC::Muls<ElementY, false>(dstTensor, threWorkspace, eMax,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
        if constexpr (std::is_same_v<ElementX, float> && std::is_same_v<ElementY, float>) {
            FTSelf::Gemv::helper::VectorBarrier();
        }
    }
};

}

#endif
