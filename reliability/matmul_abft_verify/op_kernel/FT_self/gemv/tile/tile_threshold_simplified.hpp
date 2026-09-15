/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP_SAMPLIFIED
#define FTSELF_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP_SAMPLIFIED

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../helper/type_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

template <class ElementA, class ElementX, class ElementY>
struct TileThreCalc<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED,
                FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;
    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using LayoutDst = FTSelf::layout::VectorLayout;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);

    FTSELF_DEVICE
    TileThreCalc() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> threWorkspace,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc,
        ElementY nSqrtRatioFactor,
        ElementY bSliceVar,
        ElementY eMax)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);

        if constexpr (std::is_same_v<ElementY, half>) {
            FTSelf::Gemv::helper::ZeroTensor(threWorkspace, static_cast<int32_t>(mActual) * 2);
        } else {
            AscendC::Duplicate<ElementY>(threWorkspace, static_cast<ElementY>(0.0),
                static_cast<int32_t>(mActual) * 2);
        }
        FTSelf::Gemv::helper::VectorBarrier();

        ElementY scale;
        if constexpr (std::is_same_v<ElementY, half>) {
            scale = static_cast<ElementY>(static_cast<float>(nSqrtRatioFactor) *
                static_cast<float>(bSliceVar));
        } else {
            scale = nSqrtRatioFactor * bSliceVar;
        }

        FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
        AscendC::Muls<ElementY, false>(
            threWorkspace[mActual], srcStdTensor, scale,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
        AscendC::Muls<ElementY, false>(
            dstTensor, threWorkspace[mActual], eMax,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
        FTSelf::Gemv::helper::VectorBarrier();
    }
};

}

#endif
