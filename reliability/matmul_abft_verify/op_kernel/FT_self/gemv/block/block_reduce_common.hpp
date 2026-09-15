/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_REDUCE_COMMON_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_REDUCE_COMMON_HPP

#include <cstdint>

#include "../../gemv/helper.hpp"
#include "../../gemv/tile/tile_fault_sum.hpp"
#include "../../gemv/tile/tile_vmad.hpp"
#include "../../gemv/tile/tile_vmuls.hpp"

namespace FTSelf::Gemv::Block::detail {

struct BlockReduceGeometry {
    uint32_t tileM;
    uint32_t tileN;
    uint32_t blockM;
    uint32_t blockN;
};

struct BlockReduceSetup {
    BlockReduceGeometry geometry;
    uint32_t strideACol;
    uint32_t strideARow;
    uint32_t strideOut;
};

struct BlockReduceDimensions {
    uint32_t m_actual_total;
    uint32_t n_actual_total;
    uint32_t x_actual_total;
    uint32_t y_actual_total;
    uint32_t TileMRound;
    uint32_t TileNRound;
    uint32_t BlockMRound;
    uint32_t BlockNRound;
};

template <class AType, class BType, class XType, class YType>
struct BlockReduceTypes {
    using ElementA = typename AType::Element;
    using LayoutA = typename AType::Layout;
    using ElementB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using ElementX = typename XType::Element;
    using LayoutX = typename XType::Layout;
    using ElementY = typename YType::Element;
    using LayoutY = typename YType::Layout;
};

template <class UBTileShape, class UBBlockShape, class AlignHelper>
FTSELF_DEVICE constexpr BlockReduceGeometry MakeBlockReduceGeometry()
{
    return {
        FTSelf::helper::RoundUp(UBTileShape::M, AlignHelper::ALIGN),
        FTSelf::helper::RoundUp(UBTileShape::N, AlignHelper::ALIGN),
        FTSelf::helper::RoundUp(UBBlockShape::M, AlignHelper::ALIGN),
        FTSelf::helper::RoundUp(UBBlockShape::N, AlignHelper::ALIGN)};
}

template <class UBTileShape, class UBBlockShape, class AlignHelper, class Layout>
FTSELF_DEVICE constexpr BlockReduceSetup MakeBlockReduceSetup(Layout const &layout)
{
    auto geometry = MakeBlockReduceGeometry<UBTileShape, UBBlockShape, AlignHelper>();
    return {geometry,
        static_cast<uint32_t>(layout.stride(1) * geometry.tileN),
        static_cast<uint32_t>(layout.stride(0) * geometry.tileM),
        static_cast<uint32_t>(layout.shape(0))};
}

template <uint32_t Stages, class StageEvents>
FTSELF_DEVICE void WaitForBlockReduceEvents(StageEvents const (&stageEvents)[Stages])
{
    for (uint32_t i = 0; i < Stages; ++i) {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].input);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForA);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForB);
    }
}

template <class StageEvent, class Tensor>
FTSELF_DEVICE void PrepareBlockReduceOutput(
    StageEvent const &stageEvent, Tensor minTensor, Tensor maxTensor, uint32_t elementCount)
{
    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvent.outputForB);
    FTSelf::Gemv::helper::InitializeExtrema(minTensor, maxTensor, elementCount);
    FTSelf::Gemv::helper::VectorBarrier();
}

template <class TileFaultSum, class OutputTensor, class InputTensor, class WorkspaceTensor, class Layout>
FTSELF_DEVICE void ReduceBlockTile(TileFaultSum &tileFaultSum, OutputTensor minTensor, OutputTensor maxTensor,
    InputTensor inputTensor, WorkspaceTensor workspaceTensor, Layout const &computeLayout, Layout const &actualLayout)
{
    tileFaultSum(minTensor, maxTensor, inputTensor, workspaceTensor, computeLayout, actualLayout);
}

} // namespace FTSelf::Gemv::Block::detail

#endif // FTSELF_GEMV_BLOCK_BLOCK_REDUCE_COMMON_HPP
