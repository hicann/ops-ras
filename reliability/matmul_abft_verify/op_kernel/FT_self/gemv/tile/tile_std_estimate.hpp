/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_STD_ESTIMATE_HPP
#define FTSELF_GEMV_TILE_TILE_STD_ESTIMATE_HPP

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../helper/type_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

template <class ArchTag, class XType, class YType, class BiasType = void>
struct TileStdEst
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileStdEst, can not find the specialization.");
};

template <class ArchTag, class XType, class YType, class BiasType = void>
struct TileStdEstRobust
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileStdEstRobust, can not find the specialization.");
};

template <class ArchTag, class XType, class YType, class BiasType = void>
struct TileStdEstSimplified
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileStdEstSimplified, can not find the specialization.");
};

template <class ElementX, class ElementY>
struct TileStdEst<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>, void>
{
    using ElementAccumulator = typename FTSelf::helper::ElementAccumulatorSelector<
        ElementX, ElementY>::ElementAccumulator;
    using LayoutDst = FTSelf::layout::VectorLayout;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);

    FTSELF_DEVICE
    TileStdEst() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY stdScaleFactor)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);
        if constexpr (std::is_same_v<ElementX, ElementY>) {
            AscendC::Sub(dstTensor, srcMaxTensor, srcMeanTensor, mActual);
        } else {
            AscendC::Sub(srcMaxTensor, srcMaxTensor, srcMeanTensor, mActual);
            FTSelf::Gemv::helper::VectorBarrier();
            AscendC::Cast<ElementY, ElementX>(dstTensor, srcMaxTensor,
                AscendC::RoundMode::CAST_NONE, mActual);
        }
        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
        AscendC::Muls<ElementY, false>(dstTensor, dstTensor, stdScaleFactor,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
        FTSelf::Gemv::helper::VectorBarrier();
    }
};

template <class ElementX, class ElementY>
struct TileStdEstRobust<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>, void>
{
    using ElementAccumulator = typename FTSelf::helper::ElementAccumulatorSelector<
        ElementX, ElementX>::ElementAccumulator;
    using LayoutDst = FTSelf::layout::VectorLayout;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);

    FTSELF_DEVICE
    TileStdEstRobust() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementY> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcMaxTensor,
        AscendC::LocalTensor<ElementY> srcMinTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);
        AscendC::Sub(srcMaxTensor, srcMaxTensor, srcMeanTensor, mActual);
        AscendC::Sub(srcMinTensor, srcMeanTensor, srcMinTensor, mActual);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Mul(srcMaxTensor, srcMaxTensor, srcMinTensor, mActual);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Abs(srcMaxTensor, srcMaxTensor, mActual);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Sqrt(dstTensor, srcMaxTensor, mActual);
        FTSelf::Gemv::helper::VectorBarrier();
    }
};

template <class ElementX, class ElementY>
struct TileStdEstSimplified<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>, void>
{
    using AccumulatorSelectorY = std::conditional_t<
        std::is_same_v<ElementX, half> && std::is_same_v<ElementY, float>, ElementX, ElementY>;
    using ElementAccumulator = typename FTSelf::helper::ElementAccumulatorSelector<
        ElementX, AccumulatorSelectorY>::ElementAccumulator;
    using LayoutDst = FTSelf::layout::VectorLayout;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementX);
    static constexpr uint32_t OUT_ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);

    FTSELF_DEVICE
    TileStdEstSimplified() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMaxTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY stdScaleFactor)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);
        if constexpr (std::is_same_v<ElementX, ElementY>) {
            FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
            AscendC::Muls<ElementY, false>(dstTensor, srcMaxTensor, stdScaleFactor,
                AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
            FTSelf::Gemv::helper::ResetCounterMask();
        } else {
            AscendC::Cast<ElementY, ElementX>(dstTensor, srcMaxTensor,
                AscendC::RoundMode::CAST_NONE, mActual);
            FTSelf::Gemv::helper::VectorBarrier();
            FTSelf::Gemv::helper::SetCounterMask<ElementY>(mActual);
            AscendC::Muls<ElementY, false>(dstTensor, dstTensor, stdScaleFactor,
                AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
            FTSelf::Gemv::helper::ResetCounterMask();
        }
        FTSelf::Gemv::helper::VectorBarrier();
    }
};

}

#endif
