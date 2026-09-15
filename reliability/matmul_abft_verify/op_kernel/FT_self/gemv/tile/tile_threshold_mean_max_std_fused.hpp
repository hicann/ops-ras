/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP
#define FTSELF_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../helper/type_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

FTSELF_DECLARE_TILE_THRESHOLD_ATLAS_SPECIALIZATION(
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR)

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;

    FTSELF_DEVICE
    TileThreCalc() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> threWorkspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY knRatioFactor, ElementY knSqrtRatioFactor,
        ElementY kSqrtNRatioFactor, ElementY bSliceMeanAbs,
        ElementY bSliceStd, ElementY eMax)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t mRound = FTSelf::helper::GetShape(layoutDst, 0);
        uint32_t elemRepeatSize = ELE_NUM_PER_C0 * 8;
        uint64_t addMask = (mActual < elemRepeatSize) ? mActual : elemRepeatSize;

        FTSelf::Gemv::helper::ZeroTensor(threWorkspace, static_cast<int32_t>(mActual) * 5);
        if constexpr (std::is_same_v<ElementX, ElementY>) {
            AscendC::Abs(threWorkspace[4 * mActual], srcMeanTensor, static_cast<int32_t>(mActual));
        } else {
            AscendC::Abs(srcMeanTensor, srcMeanTensor, static_cast<int32_t>(mActual));
            FTSelf::Gemv::helper::VectorBarrier();
            AscendC::Cast<ElementY, ElementX>(threWorkspace[4 * mActual], srcMeanTensor,
                AscendC::RoundMode::CAST_NONE, mActual);
        }
        FTSelf::Gemv::helper::VectorBarrier();

        ElementY meanScale;
        ElementY meanStdScale;
        ElementY stdMeanScale;
        ElementY stdScale;
        if constexpr (std::is_same_v<ElementY, half>) {
            meanScale = static_cast<ElementY>(static_cast<float>(knRatioFactor) *
                static_cast<float>(bSliceMeanAbs));
            meanStdScale = static_cast<ElementY>(4.0f * static_cast<float>(knSqrtRatioFactor) *
                static_cast<float>(bSliceStd));
            stdMeanScale = static_cast<ElementY>(4.0f * static_cast<float>(kSqrtNRatioFactor) *
                static_cast<float>(bSliceMeanAbs));
            stdScale = static_cast<ElementY>(4.0f * static_cast<float>(knSqrtRatioFactor) *
                static_cast<float>(bSliceStd));
        } else {
            meanScale = knRatioFactor * bSliceMeanAbs;
            meanStdScale = 4 * knSqrtRatioFactor * bSliceStd;
            stdMeanScale = 4 * kSqrtNRatioFactor * bSliceMeanAbs;
            stdScale = 4 * knSqrtRatioFactor * bSliceStd;
        }

        FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
        AscendC::Muls<ElementY, false>(threWorkspace, threWorkspace[4 * mActual], meanScale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        AscendC::Muls<ElementY, false>(threWorkspace[mActual], threWorkspace[4 * mActual], meanStdScale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        AscendC::Muls<ElementY, false>(threWorkspace[2 * mActual], srcStdTensor, stdMeanScale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        AscendC::Muls<ElementY, false>(threWorkspace[3 * mActual], srcStdTensor, stdScale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
        FTSelf::Gemv::helper::VectorBarrier();

        auto params = FTSelf::Gemv::MakeBinaryRepeatParams(1, 1, 1, 8, 8, 8);
        uint32_t repeatTimes = FTSelf::helper::CeilDiv(mRound, elemRepeatSize);
        AscendC::Add<ElementY, true>(dstTensor, threWorkspace, threWorkspace[mActual],
            addMask, repeatTimes, params);
        AscendC::Add<ElementY, true>(threWorkspace[2 * mActual], threWorkspace[3 * mActual],
            threWorkspace[2 * mActual], addMask, repeatTimes, params);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Add<ElementY, true>(dstTensor, threWorkspace[2 * mActual], dstTensor,
            addMask, repeatTimes, params);
        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
        AscendC::Muls<ElementY, false>(dstTensor, dstTensor, eMax,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
    }
};

}

#endif
