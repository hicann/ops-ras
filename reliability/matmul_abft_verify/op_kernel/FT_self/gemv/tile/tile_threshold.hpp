/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_THRESHOLD_HPP
#define FTSELF_GEMV_TILE_TILE_THRESHOLD_HPP

#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

template <class ElementX, class ElementY>
struct TileThreCalcBase
{
    using ElementAccumulator = typename FTSelf::helper::ElementAccumulatorSelector<
        ElementX, ElementY>::ElementAccumulator;
    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using LayoutDst = FTSelf::layout::VectorLayout;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);
};

template <
    /// Tag indicating architecture
    class ArchTag,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    class AType,
    class XType,
    class YType,
    class BiasType = void
>
struct TileThreCalc
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>, "Unsupported TileThreCalc, can not find the specialization.");
};

#define FTSELF_DECLARE_TILE_THRESHOLD_ATLAS_SPECIALIZATION(algoType_) \
template <class ElementA, class ElementX, class ElementY> \
struct TileThreCalc<FTSelf::Gemv::Arch::AtlasA2, algoType_, \
    FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>, \
    FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>, \
    FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>, void> \
    : TileThreCalcBase<ElementX, ElementY> \
{ \
    using Base = TileThreCalcBase<ElementX, ElementY>; \
    using typename Base::ElementAccumulator; \
    using typename Base::LayoutDst; \
    using typename Base::LayoutSrc; \
    using Base::ELE_NUM_PER_C0; \
    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

}

#include "../../gemv/tile/tile_threshold_compute.hpp"
#include "../../gemv/tile/tile_threshold_mean_max_std_fused.hpp"
#include "../../gemv/tile/tile_threshold_robust.hpp"
#include "../../gemv/tile/tile_threshold_simplified.hpp"
#endif
