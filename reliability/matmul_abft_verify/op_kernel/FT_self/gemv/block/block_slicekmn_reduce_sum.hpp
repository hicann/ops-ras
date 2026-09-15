/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_HPP

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "block_slice_common.hpp"

namespace FTSelf::Gemv::Block {


template <
    class UBTileShape_,
    class AType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileMatrixAdd_
>
struct BlockSliceKMNSum <
    FTSelf::Gemv::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
    UBTileShape_,
    AType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileMatrixAdd_
> {
public:
    // Type Aliases
    using DispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementX = typename AType_::Element;
    using LayoutX = FTSelf::layout::VectorLayout;

    using TileMatrixAdd = TileMatrixAdd_;

    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyUbToGm = typename TileCopy_::MatrixCopyUbToGm;
    using VecCopyGmToUbforX = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGmforY = typename TileCopy_::VecCopyUbToGm;

    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 64 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 64 * 1024;
    static constexpr uint32_t Xbuf_SIZE_for_Ae_ = 8 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_Ae_ = 8 * 1024;

    FTSELF_DECLARE_SLICE_KMN_BUFFER_CONSTRUCTORS();

private:
    template <class Resource>
    FTSELF_DEVICE
    void InitializeBuffers(Resource &resource, uint32_t UBufAddrStart)
    {
        auto offsets = detail::MakeSliceBufferOffsets(UBufAddrStart, Abuf_SIZE_);
        uint32_t UbAOffset = offsets.a;
        uint32_t UbYOffset = offsets.y;
        uint32_t UbXOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_;
        uint32_t UbYOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Ybuf_SIZE_ + Xbuf_SIZE_for_Ae_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbXTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbXOffsetforAe + i * (Xbuf_SIZE_for_Ae_ / 2));
            UbYTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforAe + i * (Ybuf_SIZE_for_Ae_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    FTSELF_DEVICE
    uint32_t BeginNLoop(uint32_t loopIdx, uint32_t loopCount, GemvCoord const &actualShape)
    {
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (loopIdx == loopCount - 1) ?
            (actualShape.n() - loopIdx * TileNRound) : TileNRound;
        y_actual = m_actual;
        x_actual = n_actual;
        return (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    FTSELF_DEVICE
    uint32_t PreloadNextY(uint32_t stage, uint32_t loopIdx, uint32_t loopCount,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape)
    {
        uint32_t nextLoop = loopIdx + 1;
        uint32_t nextN = (nextLoop == loopCount - 1) ?
            (actualShape.n() - nextLoop * TileNRound) : TileNRound;
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[stage]);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(m_actual, nextN));
        matrixCopyGmToUb(UbYTensorList[stage], gmY[nextLoop * strideY], layoutYInUb, layoutTileY);
        return nextN;
    }

    FTSELF_DEVICE
    void FinishYPreload(uint32_t stage)
    {
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbOutEventList[stage]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE3>(UbOutEventList[stage]);
    }

    FTSELF_DEVICE
    void PreloadATile(uint32_t stage, AscendC::GlobalTensor<ElementA> const &gmA,
        LayoutA const &layoutA, uint32_t offset, uint32_t tileM, uint32_t tileN)
    {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[stage]);
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(tileM, tileN));
        matrixCopyGmToUb(UbATensorList[stage], gmA[offset], layoutAInUb, layoutTileA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[stage]);
    }

    FTSELF_DEVICE
    void FinishInputStage(uint32_t nextStage)
    {
        FTSelf::Gemv::helper::VectorBarrier();
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListId]);
        UbInListId = nextStage;
    }

    FTSELF_DEVICE
    void StoreYTile(AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        uint32_t outputOffset)
    {
        FTSelf::Gemv::helper::VectorBarrier();
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_MTE3>(UbOutEventList[UbOutListId]);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(m_actual, TileNRound));
        auto layoutDstY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyUbToGm(gmY[outputOffset], UbYTensorList[UbOutListId], layoutDstY, layoutYInUb);
    }

    FTSELF_DEVICE
    void AdvanceOutputStage(uint32_t nextStage)
    {
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[UbOutListId]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);
        UbOutListId = nextStage;
    }

public:
    /// Destructor
    FTSELF_DEVICE
    ~BlockSliceKMNSum()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {

        auto config = detail::MakeSliceAeConfig<UBTileShape, UBAlignHelper>(layoutA, layoutY);
        TileMRound = config.tileM;
        TileNRound = config.tileN;
        // FTSelf::helper::RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        strideY = layoutY.stride(1) * TileNRound;
        strideSlice = layoutA.shape(0) * layoutA.shape(1);

        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        // main loop
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[UbOutListId]);
        auto UbYTensor = UbYTensorList[UbOutListId];
        // auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbYTensor, gmY, layoutYInUb, layoutTileY);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbOutEventList[UbOutListId]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE3>(UbOutEventList[UbOutListId]);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            uint32_t UbOutListIdNext = BeginNLoop(LoopIdx, Nloop, actualShape);
            uint32_t Y_block_offset = LoopIdx * strideY;
            uint32_t A_block_offset = LoopIdx * strideA;

            if (LoopIdx < Nloop - 1) {
                PreloadNextY(UbOutListIdNext, LoopIdx, Nloop, gmY, layoutY, actualShape);
                FinishYPreload(UbOutListIdNext);
            }

            if(LoopIdx == 0){
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListId]);
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListId]);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbOutEventList[UbOutListId]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);

            auto UbYIterTensor = UbYTensorList[UbOutListId];

            for(uint32_t kFTRoundIdx = 0; kFTRoundIdx < KFTRoundCount; kFTRoundIdx++){
                uint32_t kOffset = kFTRoundIdx * strideSlice;
                uint32_t kFTRoundIdxNext = kFTRoundIdx + 1;
                uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if(kFTRoundIdx < KFTRoundCount - 1){
                    // Preload next K FT round data
                    PreloadATile(UbInListIdNext, gmA, layoutA,
                        LoopIdx * strideA + kOffsetNext, m_actual, n_actual);
                }else if(LoopIdx < (Nloop - 1)){
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t m_actual_next = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
                    uint32_t n_actual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    PreloadATile(UbInListIdNext, gmA, layoutA,
                        LoopIdxNext * strideA, m_actual_next, n_actual_next);
                }

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListId]);
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                auto UbATensor = UbATensorList[UbInListId];
                tileMatrixAdd(UbYIterTensor,UbATensor,UbYIterTensor,layoutComputeInUb,layoutTileCompute);
                FinishInputStage(UbInListIdNext);
            }
            StoreYTile(gmY, layoutY, Y_block_offset);
            AdvanceOutputStage(UbOutListIdNext);
        }
    }

    FTSELF_DEVICE
    void add_ae_op(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementA> const &gmXforAe, LayoutX const &layoutXforAe,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmYforAe, LayoutX const &layoutYforAe,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {
        auto config = detail::MakeSliceAeConfig<UBTileShape, UBAlignHelper>(layoutA, layoutY);
        TileMRound = config.tileM;
        TileNRound = config.tileN;
        // FTSelf::helper::RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = config.strideA;
        strideY = config.strideY;
        strideSlice = config.strideSlice;
        strideXforAe = config.strideXforAe;
        strideYforAe = config.strideYforAe;
        strideSliceforAe = config.strideSliceforAe;

        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        // main loop
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[UbOutListId]);
        auto UbYTensor = UbYTensorList[UbOutListId];
        auto UbYTensorforAe = UbYTensorforAeList[UbOutListId];
        // auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbYTensor, gmY, layoutYInUb, layoutTileY);
        vecCopyGmToUbforX(UbYTensorforAe, gmYforAe, n_actual);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbOutEventList[UbOutListId]);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE3>(UbOutEventList[UbOutListId]);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            uint32_t UbOutListIdNext = BeginNLoop(LoopIdx, Nloop, actualShape);
            uint32_t Y_block_offset = LoopIdx * strideY;
            uint32_t Y_block_offset_for_Ae = LoopIdx * strideYforAe;

            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t n_actual_next = PreloadNextY(
                    UbOutListIdNext, LoopIdx, Nloop, gmY, layoutY, actualShape);
                auto UbYTensorforAeNext = UbYTensorforAeList[UbOutListIdNext];
                vecCopyGmToUbforX(UbYTensorforAeNext, gmYforAe[LoopIdxNext * strideYforAe], n_actual_next);
                FinishYPreload(UbOutListIdNext);
            }

            if(LoopIdx == 0){
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListId]);
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                matrixCopyGmToUb(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                vecCopyGmToUbforX(UbXTensorforAeList[UbInListId], gmXforAe[LoopIdx * strideXforAe], n_actual);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListId]);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbOutEventList[UbOutListId]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);

            auto UbYIterTensor = UbYTensorList[UbOutListId];
            auto UbYIterTensorforAe = UbYTensorforAeList[UbOutListId];

            for(uint32_t kFTRoundIdx = 0; kFTRoundIdx < KFTRoundCount; kFTRoundIdx++){
                uint32_t kOffset = kFTRoundIdx * strideSlice;
                uint32_t kFTRoundIdxNext = kFTRoundIdx + 1;
                uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;

                uint32_t kOffsetforAe = kFTRoundIdx * strideSliceforAe;
                uint32_t kOffsetforAeNext = kFTRoundIdxNext * strideSliceforAe;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if(kFTRoundIdx < KFTRoundCount - 1){
                    // Preload next K FT round data
                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListIdNext]);
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
                    matrixCopyGmToUb(UbATensorList[UbInListIdNext], gmA[LoopIdx * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    vecCopyGmToUbforX(UbXTensorforAeList[UbInListIdNext], gmXforAe[LoopIdx * strideXforAe + kOffsetforAeNext], n_actual);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListIdNext]);
                }else if(LoopIdx < (Nloop - 1)){
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t kFTRoundIdxNext = 0;

                    uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                    uint32_t kOffsetforAeNext = kFTRoundIdxNext * strideSliceforAe;
                    // Preload next N loop first K FT round data
                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListIdNext]);
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    uint32_t m_actual_next = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
                    uint32_t n_actual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(UbATensorList[UbInListIdNext], gmA[LoopIdxNext * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    vecCopyGmToUbforX(UbXTensorforAeList[UbInListIdNext], gmXforAe[LoopIdxNext * strideXforAe + kOffsetforAeNext], n_actual_next);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListIdNext]);
                }

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListId]);
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

                auto UbATensor = UbATensorList[UbInListId];
                auto UbXTensorforAe = UbXTensorforAeList[UbInListId];

                tileMatrixAdd(UbYIterTensor,UbATensor,UbYIterTensor,layoutComputeInUb,layoutTileCompute);

                AscendC::Add(UbYIterTensorforAe, UbXTensorforAe, UbYIterTensorforAe, n_actual);
                FinishInputStage(UbInListIdNext);
            }
            StoreYTile(gmY, layoutY, Y_block_offset);
            auto layoutDstYforAe = layoutYforAe.GetTileLayout(TensorCoord(n_actual));
            auto layoutYforAeInUb = layoutYforAe.GetTileLayout(TensorCoord(n_actual));
            vecCopyUbToGmforY(gmYforAe[Y_block_offset_for_Ae],
                UbYTensorforAeList[UbOutListId], layoutDstYforAe, layoutYforAeInUb);

            AdvanceOutputStage(UbOutListIdNext);
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    // AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    AscendC::LocalTensor<ElementA> UbXTensorforAeList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforAeList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t TileMRound, TileNRound;
    uint32_t strideA, strideY;
    uint32_t strideSlice;

    uint32_t strideXforAe, strideYforAe;
    uint32_t strideSliceforAe;

    TileMatrixAdd tileMatrixAdd;

    MatrixCopyGmToUb matrixCopyGmToUb;
    MatrixCopyUbToGm matrixCopyUbToGm;

    VecCopyGmToUbforX vecCopyGmToUbforX;
    VecCopyUbToGmforY vecCopyUbToGmforY;
};

} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
