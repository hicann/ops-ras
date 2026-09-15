/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_GEMV_BE_COMMON_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_GEMV_BE_COMMON_HPP
#include "../../gemv/helper.hpp"
namespace FTSelf::Gemv::Block::detail {

template <class L1TileShape, class L0TileShape, class ArchTag, class ElementA,
    class ElementX, class ElementAccumulator, uint32_t Stages>
struct BeStorageSizes {
    static constexpr uint32_t L1B_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementA);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0C_TILE_NUM = (L1TileShape::M + L0TileShape::M - 1) / L0TileShape::M;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / Stages;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / Stages;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / Stages;
    static constexpr uint32_t MAX_L1TILE_SIZE = (L1TileShape::M > L1TileShape::N) ? L1TileShape::M : L1TileShape::N;
    static constexpr uint32_t MAX_L0TILE_SIZE = (L0TileShape::M > L0TileShape::N) ? L0TileShape::M : L0TileShape::N;
    static constexpr uint32_t L1A_SIZE = 16 * L1TileShape::N * sizeof(ElementX);
    static constexpr uint32_t L0A_TILE_SIZE = L1TileShape::M * L0TileShape::N * sizeof(ElementX);
    static constexpr uint32_t L0C_TILE_SIZE = L1TileShape::M * L0TileShape::M * sizeof(ElementAccumulator);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementA);
};

struct BeStageIndices {
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};
};

template <class ElementX, class ElementA, class ElementAccumulator, uint32_t Stages>
struct BeTensorStorage {
    AscendC::LocalTensor<ElementX> l1ATensorList[Stages];
    AscendC::LocalTensor<ElementA> l1BTensorList[Stages];
    AscendC::LocalTensor<ElementX> l0ATensorList[Stages];
    AscendC::LocalTensor<ElementA> l0BTensorList[Stages];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[Stages];
};

template <class CopyA, class CopyB, class TensorA, class TensorB, class GmA, class GmX,
    class LayoutX, class LayoutA, class LayoutXInL1, class LayoutAInL1>
FTSELF_DEVICE void LoadBeTile(CopyA &copyA, CopyB &copyB, TensorA tensorA, TensorB tensorB,
    GmX gmX, GmA gmA, LayoutX const &layoutX, LayoutA const &layoutA,
    LayoutXInL1 const &layoutXInL1, LayoutAInL1 const &layoutAInL1,
    uint32_t nRound, uint32_t mActual, int32_t eventA, int32_t eventB)
{
    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE1_MTE2>(eventA);
    auto layoutTileX = layoutX.GetTileLayout(MakeCoord(nRound));
    copyA(tensorA, gmX, layoutXInL1, layoutTileX);
    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE1>(eventA);

    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE1_MTE2>(eventB);
    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mActual, nRound));
    copyB(tensorB, gmA, layoutAInL1, layoutTileA);
    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE1>(eventB);
}
template <uint32_t Stages, class EventArray>
FTSELF_DEVICE void InitializeBeStageEvents(EventArray &l1A, EventArray &l1B, EventArray &l0A, EventArray &l0B, EventArray &l0C)
{
    for (uint32_t i = 0; i < Stages; ++i) {
        l1A[i] = i; l1B[i] = i + Stages; l0A[i] = i; l0B[i] = i + Stages; l0C[i] = i + Stages * 2;
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE1_MTE2>(l1A[i]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE1_MTE2>(l1B[i]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::M_MTE1>(l0A[i]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::M_MTE1>(l0B[i]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::FIX_M>(l0C[i]);
    }
}
template <uint32_t Stages, class EventArray>
FTSELF_DEVICE void WaitBeStageEvents(EventArray const &l1A, EventArray const &l1B, EventArray const &l0A, EventArray const &l0B, EventArray const &l0C)
{
    for (uint32_t i = 0; i < Stages; ++i) {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE1_MTE2>(l1A[i]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE1_MTE2>(l1B[i]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::M_MTE1>(l0A[i]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::M_MTE1>(l0B[i]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::FIX_M>(l0C[i]);
    }
}

template <uint32_t Stages, class StageEvents>
FTSELF_DEVICE void InitializeBeStageEvents(StageEvents (&events)[Stages])
{
    for (uint32_t i = 0; i < Stages; ++i) {
        events[i].l1A = i; events[i].l1B = i + Stages;
        events[i].l0A = i; events[i].l0B = i + Stages; events[i].l0C = i + Stages * 2;
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE1_MTE2>(events[i].l1A);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE1_MTE2>(events[i].l1B);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::M_MTE1>(events[i].l0A);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::M_MTE1>(events[i].l0B);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::FIX_M>(events[i].l0C);
    }
}

template <uint32_t Stages, class StageEvents>
FTSELF_DEVICE void WaitBeStageEvents(StageEvents const (&events)[Stages])
{
    for (uint32_t i = 0; i < Stages; ++i) {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE1_MTE2>(events[i].l1A);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE1_MTE2>(events[i].l1B);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::M_MTE1>(events[i].l0A);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::M_MTE1>(events[i].l0B);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::FIX_M>(events[i].l0C);
    }
}
}
#endif
