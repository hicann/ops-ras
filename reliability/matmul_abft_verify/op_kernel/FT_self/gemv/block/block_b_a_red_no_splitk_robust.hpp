/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_SUM_AIV_AE_BMAX_MIXED_NOSPLITK_ROBUST_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_SUM_AIV_AE_BMAX_MIXED_NOSPLITK_ROBUST_HPP

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "block_reduce_common.hpp"
namespace FTSelf::Gemv::Block {


template <
    class UBTileShapeforB_,
    class UBBlockShapeforB_,
    class UBTileShapeforA_,
    class L1TileShape_,
    class AType_,
    class BType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileFaultSum_
>
struct BlockFTSumNoSplitK <
    FTSelf::Gemv::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED,
    UBTileShapeforB_,
    UBBlockShapeforB_,
    UBTileShapeforA_,
    L1TileShape_,
    AType_,
    BType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_
> : detail::BlockReduceDimensions {
public:
    // Type Aliases
    using CommonTypes = detail::BlockReduceTypes<AType_, BType_, XType_, YType_>;
    using DispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShapeforB = UBTileShapeforB_;
    using UBBlockShapeforB = UBBlockShapeforB_;
    using UBTileShapeforA = UBTileShapeforA_;
    using L1TileShape = L1TileShape_;

    using FT_AIV_PIPE_FUSE_TYPE = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    using ElementA = typename CommonTypes::ElementA;
    using LayoutA = typename CommonTypes::LayoutA;
    using ElementB = typename CommonTypes::ElementB;
    using LayoutB = typename CommonTypes::LayoutB;
    using ElementX = typename CommonTypes::ElementX;
    using LayoutX = typename CommonTypes::LayoutX;
    using ElementY = typename CommonTypes::ElementY;
    using LayoutY = typename CommonTypes::LayoutY;

    using TileFaultSum = TileFaultSum_;

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;

    using VecCopyUbToGmforAMax = typename TileCopy_::VecCopyUbToGmforAMax;
    using VecCopyUbToGmforARed = typename TileCopy_::VecCopyUbToGmforAMean;
    using VecCopyUbToGmforBRed = typename TileCopy_::VecCopyUbToGmforBMax;

    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;

    using MatrixCopyGmToUbSimplingContinue = typename TileCopy_::MatrixCopyGmToUbSimplingContinue;
    using MatrixCopyGmToUbSimplingStrided = typename TileCopy_::MatrixCopyGmToUbSimplingStrided;


    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;

    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord;

    using FT_REDUCE_TYPE = FTSelf::Gemv::helper::FT_REDUCE_TYPE;

    using TileFaultSumBRed = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX_MIN, BType_, XType_>;
    using TileFaultSumBSum = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, BType_, XType_>;
    using TileFaultSumARed = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX_MIN, AType_, YType_>;
    using TileVmulsforMean = FTSelf::Gemv::Tile::TileVmuls<FTSelf::Arch::AtlasA2, YType_>;
    // using TileFaultSumA =

    // FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t YMinbuf_forA_SIZE_ = 8 * 1024;
    static constexpr uint32_t YMinbuf_forB_SIZE_ = 8 * 1024;
    static constexpr uint32_t YMaxbuf_forA_SIZE_ = 8 * 1024;
    static constexpr uint32_t YMaxbuf_forB_SIZE_ = 8 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    static_assert(UBTileShapeforB::N == L1TileShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert((UBBlockShapeforB::N % UBTileShapeforB::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(std::is_same_v<LayoutA, LayoutB>,
        "The LayoutA and LayoutB of Gemm should be consistent.");

    static_assert(std::is_same_v<ElementA, ElementB>,
        "The ElementA and ElementB of Gemm should be consistent.");


    FTSELF_DEVICE
    BlockFTSumNoSplitK() {}

    /// Construct
    FTSELF_DEVICE
    BlockFTSumNoSplitK(FTSelf::Arch::Resource<FTSelf::Arch::AtlasA2> &resource, uint32_t UBufAddrStart = 0)
    {
        InitBuffers(resource, UBufAddrStart);
    }

    /// Construct
    FTSELF_DEVICE
    BlockFTSumNoSplitK(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        InitBuffers(resource, UBufAddrStart);
    }

private:
    template <class Resource>
    FTSELF_DEVICE void InitBuffers(Resource &resource, uint32_t UBufAddrStart)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbYMinOffsetforA = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYMinOffsetforB = UBufAddrStart + Abuf_SIZE_ + YMinbuf_forA_SIZE_;

        uint32_t UbYMaxOffsetforA = UBufAddrStart + Abuf_SIZE_ + YMinbuf_forA_SIZE_ + YMinbuf_forB_SIZE_;
        uint32_t UbYMaxOffsetforB = UBufAddrStart + Abuf_SIZE_ + YMinbuf_forA_SIZE_ + YMinbuf_forB_SIZE_ + YMaxbuf_forA_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + YMinbuf_forA_SIZE_ + YMinbuf_forB_SIZE_ + YMaxbuf_forA_SIZE_ + YMaxbuf_forB_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYMinTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYMinOffsetforA + i * (YMinbuf_forA_SIZE_ / 2));
            UbYMinTensorforBList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbYMinOffsetforB + i * (YMinbuf_forB_SIZE_ / 2));

            UbYMaxTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYMaxOffsetforA + i * (YMaxbuf_forA_SIZE_ / 2));
            UbYMaxTensorforBList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbYMaxOffsetforB + i * (YMaxbuf_forB_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementA>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            stageEvents[i].input = i;

            stageEvents[i].outputForA = i;
            stageEvents[i].outputForB = i + STAGES;

            // The event id that needs to be set before the loop
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].input);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForA);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForB);
        }
    }

    /// Destructor
public:
    FTSELF_DEVICE
    ~BlockFTSumNoSplitK()
    {
        detail::WaitForBlockReduceEvents(stageEvents);
    }

private:
    FTSELF_DEVICE void PrefetchTile(AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        uint32_t stageId, uint32_t rowOffset, uint32_t colOffset, uint32_t actualM, uint32_t actualN)
    {
        auto matrixTensor = UbATensorList[stageId];
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[stageId].input);
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualM, actualN));
        matrixCopyGmToUb(matrixTensor, gmA[rowOffset + colOffset], layoutAInUb, layoutTileA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[stageId].input);
    }

public:

    FTSELF_DEVICE
    void BlockRed(FTSelf::Gemv::BlockReduceArgs<ElementA, ElementX, LayoutA, LayoutX> const &args)
    {
        FTSELF_BIND_BLOCK_REDUCE_ARGS(args, layoutX);
        auto setup = detail::MakeBlockReduceSetup<UBTileShapeforB, UBBlockShapeforB, UBAlignHelper>(layoutA);
        TileMRound = setup.geometry.tileM;
        TileNRound = setup.geometry.tileN;
        BlockMRound = setup.geometry.blockM;
        BlockNRound = setup.geometry.blockN;
        uint32_t strideACol = setup.strideACol;
        uint32_t strideARow = setup.strideARow;
        uint32_t strideOut = setup.strideOut;

        uint32_t NloopBlock = 1;
        uint32_t MBLoopIdx = 0;

        uint32_t MloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;

        // ;
        uint32_t splitNnum = FTSelf::helper::CeilDiv(n_actual_total, TileNRound);
        uint32_t tileMnum = FTSelf::helper::CeilDiv(m_actual_total, TileMRound);

        uint32_t Nloop = splitNnum;
        uint32_t Mloop = tileMnum;

        y_actual_total = m_actual_total;
        x_actual_total = n_actual_total;

        uint32_t A_row_offset = MBLoopIdx * BlockMRound;
        uint32_t A_col_offset = NBLoopIdx * BlockNRound;
        uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

        m_actual = (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA[A_block_offset], layoutAInUb, layoutTileA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);

        // main loop
        for (uint32_t NLoopIdx = 0; NLoopIdx < Nloop; NLoopIdx++) {
            n_actual = (NLoopIdx == Nloop - 1) ? (n_actual_total - NLoopIdx * TileNRound) : TileNRound;
            auto UbYMaxTensor = UbYMaxTensorforBList[UbOutListId];
            auto UbYMinTensor = UbYMinTensorforBList[UbOutListId];
            detail::PrepareBlockReduceOutput(stageEvents[UbOutListId], UbYMinTensor, UbYMaxTensor, m_actual_total);

            // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);
            // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);

            auto UbYMaxTile = UbYMaxTensor[A_row_offset];
            auto UbYMinTile = UbYMinTensor[A_row_offset];

            uint32_t TileA_Col_offset = NLoopIdx * strideACol;
            uint32_t TileY_Row_offset = NLoopIdx * strideOut;

            for(uint32_t MLoopIdx = 0; MLoopIdx < Mloop; MLoopIdx++) {
                m_actual = (MLoopIdx == Mloop - 1) ? (m_actual_total - MLoopIdx * TileMRound) : TileMRound;
                y_actual = m_actual;
                x_actual = n_actual;
                uint32_t TileA_Row_offset = MLoopIdx * strideARow;
                uint32_t TileY_col_offset = MLoopIdx * TileMRound;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;

                if (MLoopIdx < (Mloop - 1)) {
                    uint32_t MLoopIdxNext = MLoopIdx + 1;
                    uint32_t m_actual_next = (MLoopIdxNext == Mloop - 1) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = n_actual;

                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t TileA_Row_offset_next = MLoopIdxNext * strideARow;

                    PrefetchTile(gmA, layoutA, UbInListIdNext, TileA_Row_offset_next, TileA_Col_offset,
                        m_actual_next, n_actual_next);
                }else if((MLoopIdx == (Mloop - 1)) && (NLoopIdx < (Nloop - 1))) {
                    uint32_t NLoopIdxNext = NLoopIdx + 1;
                    uint32_t MLoopIdxNext = 0;
                    uint32_t m_actual_next = (MLoopIdxNext == Mloop - 1) ? (m_actual_total - MLoopIdxNext * TileMRound) : TileMRound;
                    uint32_t n_actual_next = (NLoopIdxNext == Nloop - 1) ? (n_actual_total - NLoopIdxNext * TileNRound) : TileNRound;

                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    uint32_t TileA_Row_offset_next = MLoopIdxNext * strideARow;
                    uint32_t TileA_Col_offset_next = NLoopIdxNext * strideACol;
                    PrefetchTile(gmA, layoutA, UbInListIdNext, TileA_Row_offset_next, TileA_Col_offset_next,
                        m_actual_next, n_actual_next);
                }

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));


                detail::ReduceBlockTile(tileFaultSumBRed,
                    UbYMinTile[TileY_col_offset], UbYMaxTile[TileY_col_offset], UbATensorList[UbInListId],
                    UbWTensorList[UbInListId], layoutComputeInUb, layoutTileCompute);

                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
                UbInListId = UbInListIdNext;
            }

            // uint32_t ubTileOutOffset = TileY_Row_offset;
            auto layoutDstY = layoutX.GetTileLayout(TensorCoord(y_actual_total));
            auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(y_actual_total));

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);

            vecCopyUbToGmforBRed(gmZMax[TileY_Row_offset], UbYMaxTensorforBList[UbOutListId], layoutDstY, layoutComputeInUb);
            vecCopyUbToGmforBRed(gmZMin[TileY_Row_offset], UbYMinTensorforBList[UbOutListId], layoutDstY, layoutComputeInUb);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForB);

            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
        }
    }

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZMin,
        AscendC::GlobalTensor<ElementY> const &gmZMax,
        LayoutY const &layoutY,
        GemvCoord const &actualShape)
    {
        TileMRound = FTSelf::helper::RoundUp(UBTileShapeforA::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShapeforA::N, UBAlignHelper::ALIGN);

        uint32_t strideACol = layoutA.stride(1) * TileNRound;
        uint32_t strideARow = layoutA.stride(0) * TileMRound;

        uint32_t strideOut = 1;

        uint32_t NloopBlock = 1;
        uint32_t MBLoopIdx = 0;

        uint32_t MloopBlock = 1;
        uint32_t NBLoopIdx = 0;

        m_actual_total = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual_total = actualShape.n();

        ElementY A_row_scale_ratio = (ElementY)(1.0f / (n_actual_total * 1.0f));

        uint32_t splitNnum = FTSelf::helper::CeilDiv(n_actual_total, TileNRound);
        uint32_t tileMnum = FTSelf::helper::CeilDiv(m_actual_total, TileMRound);

        uint32_t Nloop = splitNnum;
        uint32_t Mloop = tileMnum;

        y_actual_total = m_actual_total;
        x_actual_total = n_actual_total;

        uint32_t A_row_offset = MBLoopIdx * TileMRound;
        uint32_t A_col_offset = NBLoopIdx * TileNRound;
        uint32_t A_block_offset = A_row_offset * layoutA.stride(0) + A_col_offset * layoutA.stride(1);

        m_actual = m_actual_total;
        // (m_actual_total < TileMRound) ? m_actual_total : TileMRound;
        n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForA);
        auto UbYMinTensor = UbYMinTensorforAList[UbOutListId];
        auto UbYMaxTensor = UbYMaxTensorforAList[UbOutListId];

        FTSelf::Gemv::helper::InitializeExtrema(UbYMinTensor, UbYMaxTensor, m_actual_total);
        FTSelf::Gemv::helper::VectorBarrier();

        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForA);
        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForA);


        // main loop
        for (uint32_t NLoopIdx = 0; NLoopIdx < Nloop; NLoopIdx++) {
            n_actual = (NLoopIdx == Nloop - 1) ? (n_actual_total - NLoopIdx * TileNRound) : TileNRound;
            m_actual = m_actual_total;
            uint32_t TileA_Col_offset = NLoopIdx * strideACol;

            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t TileA_Row_offset = 0;
            uint32_t TileY_col_offset = 0;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;

            if(NLoopIdx < (Nloop - 1)) {
                uint32_t NLoopIdxNext = NLoopIdx + 1;
                uint32_t m_actual_next = m_actual_total;
                uint32_t n_actual_next = (NLoopIdxNext == Nloop - 1) ? (n_actual_total - NLoopIdxNext * TileNRound) : TileNRound;

                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;

                uint32_t TileA_Row_offset_next = 0;
                uint32_t TileA_Col_offset_next = NLoopIdxNext * strideACol;
                PrefetchTile(gmA, layoutA, UbInListIdNext, TileA_Row_offset_next, TileA_Col_offset_next,
                    m_actual_next, n_actual_next);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));


            tileFaultSumARed(UbYMinTensor, UbYMaxTensor,
                UbATensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
            UbInListId = UbInListIdNext;
        }

        // FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForA);

        // uint32_t ubTileOutOffset = TileY_Row_offset;
        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(m_actual_total));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(m_actual_total));

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForA);

        vecCopyUbToGmforARed(gmZMin, UbYMinTensorforAList[UbOutListId], layoutDstY, layoutComputeInUb);
        vecCopyUbToGmforARed(gmZMax, UbYMaxTensorforAList[UbOutListId], layoutDstY, layoutComputeInUb);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForA);
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];

    AscendC::LocalTensor<ElementY> UbYMinTensorforAList[STAGES];
    AscendC::LocalTensor<ElementY> UbYMaxTensorforAList[STAGES];

    AscendC::LocalTensor<ElementX> UbYMinTensorforBList[STAGES];
    AscendC::LocalTensor<ElementX> UbYMaxTensorforBList[STAGES];

    AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    struct StageEvents {
        int32_t input;
        int32_t outputForA;
        int32_t outputForB;
    };

    StageEvents stageEvents[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual, n_actual_local;
    TileFaultSumBRed tileFaultSumBRed;
    TileFaultSumARed tileFaultSumARed;
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyUbToGmforARed vecCopyUbToGmforARed;
    VecCopyUbToGmforBRed vecCopyUbToGmforBRed;

};

} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
