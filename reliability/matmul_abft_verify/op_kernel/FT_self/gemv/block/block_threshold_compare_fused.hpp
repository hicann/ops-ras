/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_THRESHOLD_COMPARE_FUSED_AIV_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_THRESHOLD_COMPARE_FUSED_AIV_HPP

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "../../gemv/tile/tile_fault_compare.hpp"

namespace FTSelf::Gemv::Block {

struct ThresholdStageEvents {
    int32_t inputA;
    int32_t inputX;
    int32_t output;
};

#define FTSELF_THRESHOLD_TYPE_PARAMS \
    class AType_, class XType_, class YType_, class ZType_, class BiasType_, \
    class TileCopy_, class TileThreCalc_, class TileVmuls_

#define FTSELF_THRESHOLD_TYPE_ARGS \
    AType_, XType_, YType_, ZType_, BiasType_, TileCopy_, TileThreCalc_, TileVmuls_

#define FTSELF_THRESHOLD_COMMON_TYPES \
    using DispatchPolicy = FTSelf::Gemv::GemvAtlasA2; \
    using ArchTag = typename DispatchPolicy::ArchTag; \
    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE; \
    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM; \
    using FT_AIV_PIPE_FUSE_TYPE = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE; \
    using ElementA = typename AType_::Element; \
    using LayoutA = typename AType_::Layout; \
    using ElementX = typename XType_::Element; \
    using LayoutX = typename XType_::Layout; \
    using ElementY = typename YType_::Element; \
    using LayoutY = typename YType_::Layout; \
    using ElementZ = typename ZType_::Element; \
    using LayoutZ = typename ZType_::Layout; \
    using TileThreCalc = TileThreCalc_; \
    using TileVmuls = TileVmuls_; \
    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb; \
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm; \
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb; \
    using VecCopyGmToUbInX = typename TileCopy_::VecCopyGmToUbInX; \
    using VecCopyGmToUbInY = typename TileCopy_::VecCopyGmToUbInY; \
    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ; \
    using ElementAccumulator = typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator; \
    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>; \
    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord; \
    using ElementXoR = uint16_t; \
    using ElementComp = int32_t; \
    using ElementSub = ElementY; \
    using ElementWork = ElementY; \
    using TileCompare = FTSelf::Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag, ZType_, YType_, YType_>; \
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES; \
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024; \
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB

#define FTSELF_THRESHOLD_COMMON_LIFECYCLE(CLASS_NAME) \
    FTSELF_DEVICE CLASS_NAME() {} \
    FTSELF_DEVICE CLASS_NAME(FTSelf::Arch::Resource<FTSelf::Arch::AtlasA2> &resource, uint32_t start = 0) \
    { Initialize(resource, start); } \
    FTSELF_DEVICE CLASS_NAME(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t start = 0) \
    { Initialize(resource, start); } \
    FTSELF_DEVICE ~CLASS_NAME() \
    { \
        for (uint32_t i = 0; i < STAGES; ++i) { \
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputA); \
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputX); \
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].output); \
        } \
    } \
    template <class Resource> \
    FTSELF_DEVICE void Initialize(Resource &resource, uint32_t start) \
    { \
        uint32_t yOffset = start + Abuf_SIZE_; \
        uint32_t workOffset = yOffset + Ybuf_SIZE_; \
        uint32_t inXOffset = workOffset + workspace_SIZE_; \
        uint32_t inCOffset = inXOffset + InXbuf_SIZE; \
        uint32_t zOffset = inCOffset + InCbuf_SIZE; \
        for (uint32_t i = 0; i < STAGES; ++i) { \
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(start + i * (Abuf_SIZE_ / 2)); \
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(yOffset + i * (Ybuf_SIZE_ / 2)); \
            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(workOffset + i * (workspace_SIZE_ / 2)); \
            UbInXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(inXOffset + i * (InXbuf_SIZE / 2)); \
            UbInCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(inCOffset + i * (InCbuf_SIZE / 2)); \
            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(zOffset + i * (Zbuf_SIZE / 2)); \
            stageEvents[i].inputA = i; \
            stageEvents[i].inputX = i + STAGES; \
            stageEvents[i].output = i; \
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputA); \
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputX); \
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].output); \
        } \
    }

#define FTSELF_THRESHOLD_COMPUTE_TILE_TAIL \
    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].inputA); \
    } \
    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputA); \
    auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound)); \
    auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n)); \
    tileThreCalc(OutYTile, UbATensorList[UbInListId], UbWTensorList[UbInListId], \
        layoutComputeInUb, layoutTileCompute, dst_offset_ratio); \
    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputA); \
    UbInListId = UbInListIdNext; \
    } \
    FTSelf::Gemv::helper::VectorBarrier()

#define FTSELF_THRESHOLD_COMMON_STORAGE \
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES]; \
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES]; \
    AscendC::LocalTensor<ElementX> UbWTensorList[STAGES]; \
    AscendC::LocalTensor<ElementY> UbInXTensorList[STAGES]; \
    AscendC::LocalTensor<ElementY> UbInCTensorList[STAGES]; \
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES]; \
    ThresholdStageEvents stageEvents[STAGES]; \
    uint32_t UbOutListId{0}; \
    uint32_t UbInListId{0}; \
    FTSelf::Gemv::TileActualShape tileActual; \
    uint32_t out_z_actual, m_actual_total, n_actual_total

#define FTSELF_THRESHOLD_TILE_OPERATORS \
    uint32_t dst_offset_ratio; \
    TileThreCalc tileThreCalc; \
    TileVmuls tileVmuls; \
    MatrixCopyGmToUb matrixCopyGmToUb; \
    VecCopyGmToUb vecCopyGmToUb; \
    VecCopyUbToGm vecCopyUbToGm; \
    VecCopyGmToUbInX vecCopyGmToUbInX; \
    VecCopyGmToUbInY vecCopyGmToUbInY; \
    VecCopyUbToGmZ vecCopyUbToGmZ; \
    TileCompare tileCompare


// class TileVmuls_

template <
    class UBTileShape_,
    class UBTileShapeTotal_,
    FTSELF_THRESHOLD_TYPE_PARAMS
>
struct BlockThresholdCalcFused <
    FTSelf::Gemv::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED,
    FTSelf::Gemv::helper::FT_ENC_TYPE::RCE,
    FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    UBTileShapeTotal_,
    FTSELF_THRESHOLD_TYPE_ARGS
> {
public:
    FTSELF_THRESHOLD_COMMON_TYPES;
    using UBTileShape = UBTileShape_;
    using UBTileShapeTotal = UBTileShapeTotal_;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 12 * 1024;
    static constexpr uint32_t Zbuf_SIZE = 4 *1024;
    static constexpr uint32_t InXbuf_SIZE = 8 * 1024;
    static constexpr uint32_t InCbuf_SIZE = 8 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static constexpr uint32_t ELE_NUM_PER_REPEAT = FTSelf::Gemv::BYTE_PER_C0 * 8 / sizeof(ElementY);
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;


    FTSELF_THRESHOLD_COMMON_LIFECYCLE(BlockThresholdCalcFused)

    // AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,

    FTSELF_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmInX, LayoutY const &layoutInX,
        AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementZ> const &gmOutZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &actualShapeTotal,
        ElementY alpha,bool outputThre)
    {
        TileMRound = FTSelf::helper::RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        TileMTotalRound = FTSelf::helper::RoundUp(UBTileShapeTotal::M, UBAlignHelper::ALIGN);
        TileNTotalRound = FTSelf::helper::RoundUp(UBTileShapeTotal::N, UBAlignHelper::ALIGN);

        strideA = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;
        tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        tileActual.SetN((actualShape.n() < TileNRound) ? actualShape.n() : TileNRound);

        m_actual_total = (actualShapeTotal.m() < TileMTotalRound) ? actualShapeTotal.m() : TileMTotalRound;
        n_actual_total = (actualShapeTotal.n() < TileNTotalRound) ? actualShapeTotal.n() : TileNTotalRound;

        uint32_t MLoop = FTSelf::helper::CeilDiv(m_actual_total, TileMRound);

        dst_offset_ratio = MLoop;

        out_z_actual = (m_actual_total + 8 - 1)/ 8;
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].output);

        FTSelf::Gemv::helper::ZeroTensor(UbYTensorList[UbOutListId], m_actual_total);

        FTSelf::Gemv::helper::VectorBarrier();

        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);
        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);

        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputX);
        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputX);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputA);
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputA);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].inputX);
        vecCopyGmToUbInX(UbInXTensorList[UbOutListId], gmInX, m_actual_total);
        vecCopyGmToUbInY(UbInCTensorList[UbOutListId], gmInC, m_actual_total);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].inputX);

        // main loop
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);
        for(uint32_t mLoopIdx = 0; mLoopIdx < MLoop; mLoopIdx++) {
            tileActual.SetM((mLoopIdx == MLoop - 1) ? (actualShapeTotal.m() - mLoopIdx * TileMRound) : TileMRound);
            auto OutYTile = UbYTensorList[UbOutListId][mLoopIdx * TileMRound];
            for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
                tileActual.SetN((LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound);
                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (LoopIdx < Nloop - 1) {
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t m_actual_next = tileActual.m;
                    uint32_t n_actual_next =
                        (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].inputA);
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[mLoopIdx * strideARow + LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].inputA);
                }else if((LoopIdx == Nloop - 1) && (mLoopIdx < MLoop - 1)){
                    uint32_t LoopIdxNext = 0;
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    uint32_t m_actual_next = (mLoopIdxNext == MLoop - 1) ? (actualShapeTotal.m() - mLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next =
                        (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].inputA);
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[mLoopIdxNext * strideARow + LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                    FTSELF_THRESHOLD_COMPUTE_TILE_TAIL;
        }

        // UbYTensorList[UbOutListId]
        tileVmuls(UbYTensorList[UbOutListId], UbYTensorList[UbOutListId], (ElementY)alpha, m_actual_total);

        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].inputX);


        // UbWTensorList[UbListId],

        auto layoutCompareInUb = layoutY.GetTileLayout(MakeCoord(TileMTotalRound));
        auto layoutTileCompare = layoutY.GetTileLayout(MakeCoord(m_actual_total));

        tileCompare(
            UbZTensorList[UbOutListId],
            UbInXTensorList[UbOutListId],
            UbInCTensorList[UbOutListId],
            UbYTensorList[UbOutListId],
            layoutCompareInUb, layoutTileCompare, (ElementY)0.002f);

        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);

        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].output);
        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].output);

        if(outputThre){
            auto layoutDstYThre = layoutY.GetTileLayout(TensorCoord(m_actual_total));
            auto layoutComputeThreInUb = layoutY.GetTileLayout(TensorCoord(m_actual_total));
            vecCopyUbToGm(gmThreZ, UbYTensorList[UbOutListId], layoutDstYThre, layoutComputeThreInUb);
        }

        auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
        auto layoutComputeZInUb = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
        vecCopyUbToGmZ(gmOutZ, UbZTensorList[UbOutListId], layoutDstZ, layoutComputeZInUb);

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].output);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].inputX);

        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    FTSELF_THRESHOLD_COMMON_STORAGE;
    uint32_t TileMRound, TileNRound;
    uint32_t TileMTotalRound, TileNTotalRound;
    uint32_t strideA, strideARow;
    FTSELF_THRESHOLD_TILE_OPERATORS;
};



template <
    class UBTileShape_,
    class UBTileShapeTotal_,
    class L1TileShape_,
    FTSELF_THRESHOLD_TYPE_PARAMS
>
struct BlockThresholdCalcFused <
    FTSelf::Gemv::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
    FTSelf::Gemv::helper::FT_ENC_TYPE::RCE,
    FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    UBTileShapeTotal_,
    L1TileShape_,
    FTSELF_THRESHOLD_TYPE_ARGS
> {
public:
    FTSELF_THRESHOLD_COMMON_TYPES;
    using UBTileShape = UBTileShape_;
    using UBTileShapeTotal = UBTileShapeTotal_;
    using L1TileShape = L1TileShape_;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 12 * 1024;
    static constexpr uint32_t Zbuf_SIZE = 4 *1024;
    static constexpr uint32_t InXbuf_SIZE = 8 * 1024;
    static constexpr uint32_t InCbuf_SIZE = 8 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static constexpr uint32_t ELE_NUM_PER_REPEAT = FTSelf::Gemv::BYTE_PER_C0 * 8 / sizeof(ElementY);

    static_assert(L1TileShape::M == UBTileShapeTotal::M,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert(L1TileShape::N == UBTileShapeTotal::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(UBTileShapeTotal::N == UBTileShape::N,
        "The situation where the basic Tile of UB in tile and block differ on the n axes is not supported yet");
    FTSELF_THRESHOLD_COMMON_LIFECYCLE(BlockThresholdCalcFused)

    // AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,

    FTSELF_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmInX, LayoutY const &layoutInX,
        AscendC::GlobalTensor<ElementY> const &gmInC, LayoutY const &layoutInC,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementZ> const &gmOutZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &actualShapeTotal,
        ElementY alpha,bool outputThre, uint32_t aiv_part_num=2)
    {
        aiv_part_num = aiv_part_num == 0 ? 1 : aiv_part_num;
        TileMRound = FTSelf::helper::RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        BlockMRound = FTSelf::helper::RoundUp(UBTileShapeTotal::M, UBAlignHelper::ALIGN);
        BlockNRound = FTSelf::helper::RoundUp(UBTileShapeTotal::N, UBAlignHelper::ALIGN);

        strideACol = layoutA.stride(1) * TileNRound;
        strideARow = layoutA.stride(0) * TileMRound;
        // tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        // tileActual.SetN((actualShape.n() < TileNRound) ? actualShape.n() : TileNRound);

        m_actual_total = (actualShapeTotal.m() < BlockMRound) ? actualShapeTotal.m() : BlockMRound;
        n_actual_total = (actualShapeTotal.n() < BlockNRound) ? actualShapeTotal.n() : BlockNRound;

        if(actualShape.n() < n_actual_total){
            n_actual_total = actualShape.n();
        }

        m_actual_part = m_actual_total / aiv_part_num;

        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_actual_part;
        uint32_t Z_start_offset = (M_start_offset + 8 - 1) / 8;

        if(AscendC::GetSubBlockIdx() == (aiv_part_num -1)) {
            m_actual_part = m_actual_total - M_start_offset;
        }

        uint32_t MLoop = FTSelf::helper::CeilDiv(m_actual_part, TileMRound);
        uint32_t Nloop = FTSelf::helper::CeilDiv(n_actual_total, TileNRound);

        dst_offset_ratio = MLoop;

        out_z_actual_total = (m_actual_part + 8 - 1) / 8;

        for(uint32_t mLoopIdx = 0; mLoopIdx < MLoop; mLoopIdx++){
            tileActual.SetM((mLoopIdx < (MLoop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound);
            tileActual.SetN((n_actual_total < TileNRound) ? n_actual_total : TileNRound);

            out_z_actual = (tileActual.m + 8 - 1) / 8;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t mLoopOffset_for_z = (mLoopOffset + 8 - 1) / 8;
            uint32_t A_row_offset = mLoopOffset;
            uint32_t A_col_offset = 0;
            uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].output);
            FTSelf::Gemv::helper::ZeroTensor(UbYTensorList[UbOutListId], tileActual.m);
            FTSelf::Gemv::helper::VectorBarrier();

            auto OutYTile = UbYTensorList[UbOutListId];

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].inputX);
            vecCopyGmToUbInX(UbInXTensorList[UbOutListId], gmInX[mLoopOffset], tileActual.m);
            vecCopyGmToUbInY(UbInCTensorList[UbOutListId], gmInC[mLoopOffset], tileActual.m);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].inputX);

            if(mLoopIdx == 0){
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputA);
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputA);
            }

            for (uint32_t nLoopIdx = 0; nLoopIdx < Nloop; nLoopIdx++) {
                tileActual.SetN((nLoopIdx == Nloop - 1) ? (n_actual_total - nLoopIdx * TileNRound) : TileNRound);

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if (nLoopIdx < Nloop - 1) {
                    uint32_t nLoopIdxNext = nLoopIdx + 1;
                    uint32_t m_actual_next = tileActual.m;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].inputA);
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].inputA);
                }else if((nLoopIdx == Nloop - 1) && (mLoopIdx < MLoop - 1)){
                    uint32_t nLoopIdxNext = 0;
                    uint32_t mLoopIdxNext = mLoopIdx + 1;
                    uint32_t m_actual_next = (mLoopIdxNext == MLoop - 1) ? (m_actual_part - mLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == Nloop - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t A_block_offset_next = A_block_offset + strideARow;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbATensorList[UbInListIdNext];

                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].inputA);
                    auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmA[A_block_offset_next + nLoopIdxNext * strideACol], layoutAInUb, layoutTileA);
                    FTSELF_THRESHOLD_COMPUTE_TILE_TAIL;

            // UbYTensorList[UbOutListId]
            tileVmuls(UbYTensorList[UbOutListId], UbYTensorList[UbOutListId], (ElementY)alpha, tileActual.m);

            FTSelf::Gemv::helper::VectorBarrier();
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].inputX);

            auto layoutCompareInUb = layoutY.GetTileLayout(MakeCoord(TileMRound));
            auto layoutTileCompare = layoutY.GetTileLayout(MakeCoord(tileActual.m));

            tileCompare(
                UbZTensorList[UbOutListId],
                UbInXTensorList[UbOutListId],
                UbInCTensorList[UbOutListId],
                UbYTensorList[UbOutListId],
                layoutCompareInUb,
                layoutTileCompare,
                (ElementY)0.002f
            );

            FTSelf::Gemv::helper::VectorBarrier();

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);

            if(outputThre){
                auto layoutDstYThre = layoutY.GetTileLayout(TensorCoord(tileActual.m));
                auto layoutComputeThreInUb = layoutY.GetTileLayout(TensorCoord(tileActual.m));
                vecCopyUbToGm(gmThreZ[mLoopOffset],
                    UbYTensorList[UbOutListId],
                    layoutDstYThre,
                    layoutComputeThreInUb);
            }

            auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            auto layoutComputeZInUb = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            vecCopyUbToGmZ(gmOutZ[mLoopOffset_for_z], UbZTensorList[UbOutListId], layoutDstZ, layoutComputeZInUb);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].output);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].inputX);

            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
        }
    }

protected:
    FTSELF_THRESHOLD_COMMON_STORAGE;
    uint32_t m_actual_part;
    uint32_t out_z_actual_total;
    uint32_t TileMRound, TileNRound;
    uint32_t BlockMRound, BlockNRound;
    uint32_t strideACol, strideARow;
    FTSELF_THRESHOLD_TILE_OPERATORS;
};




#undef FTSELF_THRESHOLD_TILE_OPERATORS
#undef FTSELF_THRESHOLD_COMMON_STORAGE
#undef FTSELF_THRESHOLD_COMPUTE_TILE_TAIL
#undef FTSELF_THRESHOLD_COMMON_LIFECYCLE
#undef FTSELF_THRESHOLD_COMMON_TYPES
#undef FTSELF_THRESHOLD_TYPE_ARGS
#undef FTSELF_THRESHOLD_TYPE_PARAMS

} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
