/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_HPP_ROBUST
#define FTSELF_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_HPP_ROBUST

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "../../gemv/tile/tile_reduce_mean_var.hpp"
#include "block_slice_common.hpp"
#include "block_slicekmn_reduce_sum_fused_common.hpp"

namespace FTSelf::Gemv::Block {

FTSELF_DECLARE_SLICE_KMN_FUSED_SPECIALIZATION(
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_ROBUST) {
public:
    // Type Aliases
    using DispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShapeforA = UBTileShapeforA_;
    using UBTileShapeforB = UBTileShapeforB_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using TileMatrixAdd = TileMatrixAdd_;

    using TileFaultSum = TileFaultSum_;
    using TileVmuls = TileVmuls_;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using FT_REDUCE_TYPE = FTSelf::Gemv::helper::FT_REDUCE_TYPE;

    using TileFaultMeanforB = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, BType_, XType_>;
    using TileFaultMaxforB = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX, BType_, XType_>;

    using TileReduceMeanforB = FTSelf::Gemv::Tile::TileReduce<
        ArchTag,
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FT_REDUCE_TYPE::MEAN_SQUARE,
        BType_,
        XType_,
        XType_,
        void>;

    using TileReduceVarforB = FTSelf::Gemv::Tile::TileReduce<
        ArchTag,
        FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FT_REDUCE_TYPE::VAR,
        BType_,
        XType_,
        XType_,
        void>;

    using MatrixCopyGmToUbforA = typename TileCopyforA_::MatrixCopyGmToUb;
    using MatrixCopyUbToGmforA = typename TileCopyforA_::MatrixCopyUbToGm;
    using VecCopyGmToUbforX = typename TileCopyforA_::VecCopyGmToUb;
    using VecCopyUbToGmforY = typename TileCopyforA_::VecCopyUbToGm;

    using VecCopyGmToUbforB = typename TileCopyforB_::VecCopyGmToUb;
    using VecCopyUbToGmforB = typename TileCopyforB_::VecCopyUbToGm;
    using MatrixCopyGmToUbforB = typename TileCopyforB_::MatrixCopyGmToUb;
    using MatrixCopyGmToUbSimplingContinueforB = typename TileCopyforB_::MatrixCopyGmToUbSimplingContinue;
    using MatrixCopyGmToUbSimplingStridedforB = typename TileCopyforB_::MatrixCopyGmToUbSimplingStrided;

    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t Abuf_SIZE_ = 84 * 1024;
    static constexpr uint32_t Xbuf_SIZE_for_Ae_ = 8 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_Ae_ = 8 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_BVar_ = 6 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_BMean_ = 6 * 1024;
    static constexpr uint32_t Ybuf_SIZE_for_A_ = 64 * 1024;

    static constexpr uint32_t Abuf_SIZE_for_BMean_ = 28 * 1024;
    static constexpr uint32_t Abuf_SIZE_for_BMin_ = 28 * 1024;
    static constexpr uint32_t Abuf_SIZE_for_BMax_ = 28 * 1024;
    // static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;


    FTSELF_DECLARE_SLICE_KMN_BUFFER_CONSTRUCTORS();

private:
    template <class Resource>
    FTSELF_DEVICE
    void InitializeBuffers(Resource &resource, uint32_t UBufAddrStart)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbAOffsetforBMean = UBufAddrStart;
        uint32_t UbAOffsetforBMax = UBufAddrStart + Abuf_SIZE_for_BMean_;
        uint32_t UbAOffsetforBMin = UBufAddrStart + Abuf_SIZE_for_BMean_ + Abuf_SIZE_for_BMax_;

        uint32_t UbXOffsetforAe = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffsetforA = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_;

        uint32_t UbYOffsetforBMean = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_;
        uint32_t UbYOffsetforBVar = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_;
        uint32_t UbYOffsetforAe = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_;

        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_for_Ae_ + Ybuf_SIZE_for_A_ + Ybuf_SIZE_for_BMean_ + Ybuf_SIZE_for_BVar_ + Ybuf_SIZE_for_Ae_;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbBMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMean + i * (Abuf_SIZE_for_BMean_/ 2));
            UbBMaxTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMax + i * (Abuf_SIZE_for_BMax_ / 2));
            UbBMinTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(UbAOffsetforBMin + i * (Abuf_SIZE_for_BMin_ / 2));


            UbXTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbXOffsetforAe + i * (Xbuf_SIZE_for_Ae_ / 2));

            UbYTensorforAList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforA + i * (Ybuf_SIZE_for_A_ / 2));

            UbYTensorforBMeanList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBMean + i * (Ybuf_SIZE_for_BMean_ / 2));
            UbYTensorforBVarList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforBVar + i * (Ybuf_SIZE_for_BVar_ / 2));

            UbYTensorforAeList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffsetforAe + i * (Ybuf_SIZE_for_Ae_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset + i * (workspace_SIZE_ / 2));
            UbWTensorforRawList[i] = UbWTensorList[i].template ReinterpretCast<ElementB>();

            // Assign event ID for each stages
            stageEvents[i].input = i;
            stageEvents[i].outputForA = i;
            stageEvents[i].outputForB = i + STAGES;

            // The event id that needs to be set before the loop
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].input);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForA);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[i].outputForA);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForB);
        }
    }

    FTSELF_DEVICE
    uint32_t NextStage(uint32_t stage) const
    {
        return (stage + 1 < STAGES) ? (stage + 1) : 0;
    }

    FTSELF_DEVICE
    uint32_t CopyNextY(uint32_t stage, uint32_t loopIdx, uint32_t loopCount,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape)
    {
        uint32_t nextLoop = loopIdx + 1;
        uint32_t nextN = (nextLoop == loopCount - 1) ?
            (actualShape.n() - nextLoop * TileNRound) : TileNRound;
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[stage].outputForA);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(tileActual.m, nextN));
        matrixCopyGmToUbforA(UbYTensorforAList[stage], gmY[nextLoop * strideY], layoutYInUb, layoutTileY);
        return nextN;
    }

    FTSELF_DEVICE
    void FinishYPreload(uint32_t stage)
    {
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[stage].outputForA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE3>(stageEvents[stage].outputForA);
    }

    FTSELF_DEVICE
    void FinishInputStage(uint32_t nextStage)
    {
        FTSelf::Gemv::helper::VectorBarrier();
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
        UbInListId = nextStage;
    }

    FTSELF_DEVICE
    void StoreATile(AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY, uint32_t offset)
    {
        FTSelf::Gemv::helper::VectorBarrier();
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForA);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForA);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_MTE3>(stageEvents[UbOutListId].outputForA);
        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(tileActual.m, TileNRound));
        auto layoutDstY = layoutY.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
        matrixCopyUbToGmforA(gmY[offset], UbYTensorforAList[UbOutListId], layoutDstY, layoutYInUb);
    }

    FTSELF_DEVICE
    void PrepareBReduction(LayoutB const &layoutB, GemvCoord const &actualShape)
    {
        TileMRound = FTSelf::helper::RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        strideB = layoutB.stride(1) * TileNRound;
        tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        tileActual.SetN((actualShape.n() < TileNRound) ? actualShape.n() : TileNRound);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForB);
    }

    template <class Tensor>
    FTSELF_DEVICE
    void LoadBTile(uint32_t stage, Tensor dst, AscendC::GlobalTensor<ElementB> const &gmB,
        LayoutB const &layoutB, uint32_t offset, uint32_t tileM, uint32_t tileN)
    {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[stage].input);
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(tileM, tileN));
        matrixCopyGmToUbforB(dst, gmB[offset], layoutBInUb, layoutTileB);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[stage].input);
    }

    FTSELF_DEVICE
    uint32_t BeginBLoop(uint32_t loopIdx, uint32_t loopCount, GemvCoord const &actualShape)
    {
        tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        tileActual.SetN((loopIdx == loopCount - 1) ?
            (actualShape.n() - loopIdx * TileNRound) : TileNRound);
        return NextStage(UbInListId);
    }

    template <class Tensor>
    FTSELF_DEVICE
    void PreloadNextB(uint32_t stage, Tensor dst, AscendC::GlobalTensor<ElementB> const &gmB,
        LayoutB const &layoutB, uint32_t loopIdx, uint32_t loopCount, GemvCoord const &actualShape)
    {
        uint32_t nextLoop = loopIdx + 1;
        uint32_t nextN = (nextLoop == loopCount - 1) ?
            (actualShape.n() - nextLoop * TileNRound) : TileNRound;
        LoadBTile(stage, dst, gmB, layoutB, nextLoop * strideB, tileActual.m, nextN);
    }

    FTSELF_DEVICE
    void PrepareMeanOutputs(LayoutB const &layoutB, GemvCoord const &actualShape)
    {
        strideB = layoutB.stride(1) * TileNRound;
        tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        tileActual.SetN((actualShape.n() < TileNRound) ? actualShape.n() : TileNRound);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForB);
        FTSelf::Gemv::helper::ZeroTensor(UbYTensorforBMeanList[UbOutListId], tileActual.m);
        FTSelf::Gemv::helper::ZeroTensor(UbYTensorforBVarList[UbOutListId], tileActual.m);
        FTSelf::Gemv::helper::VectorBarrier();
    }

    template <bool STRIDED>
    FTSELF_DEVICE
    void LoadSampledB(uint32_t stage, AscendC::GlobalTensor<ElementB> const &gmB,
        LayoutB const &layoutB, uint32_t offset, uint32_t tileN,
        uint32_t samplingStride, uint32_t strideUnit)
    {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[stage].input);
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(tileActual.m, tileN));
        if (tileN > TileNRoundSimpling && samplingStride > 1) {
            if constexpr (STRIDED) {
                matrixCopyGmToUbSimplingSforB(UbBMeanTensorList[stage], gmB[offset],
                    layoutBInUb, layoutTileB, samplingStride, strideUnit);
            } else {
                matrixCopyGmToUbSimplingCforB(UbBMeanTensorList[stage], gmB[offset],
                    layoutBInUb, layoutTileB, samplingStride);
            }
        } else {
            matrixCopyGmToUbforB(UbBMeanTensorList[stage], gmB[offset], layoutBInUb, layoutTileB);
        }
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[stage].input);
    }

    template <bool STRIDED>
    FTSELF_DEVICE
    void PreloadNextSampledB(uint32_t stage, AscendC::GlobalTensor<ElementB> const &gmB,
        LayoutB const &layoutB, uint32_t loopIdx, uint32_t loopCount,
        GemvCoord const &actualShape, uint32_t samplingStride, uint32_t strideUnit)
    {
        uint32_t nextLoop = loopIdx + 1;
        uint32_t nextN = (nextLoop == loopCount - 1) ?
            (actualShape.n() - nextLoop * TileNRound) : TileNRound;
        LoadSampledB<STRIDED>(stage, gmB, layoutB, nextLoop * strideB,
            nextN, samplingStride, strideUnit);
    }

    template <class LayoutComputeInUb, class LayoutTileCompute>
    FTSELF_DEVICE
    void ReduceMeanTile(LayoutComputeInUb const &layoutComputeInUb,
        LayoutTileCompute const &layoutTileCompute, float scaleRatio, uint32_t nextStage)
    {
        tileReduceMeanforB(UbYTensorforBMeanList[UbOutListId],
            UbYTensorforBVarList[UbOutListId], UbBMeanTensorList[UbInListId],
            UbWTensorList[UbInListId], layoutComputeInUb, layoutTileCompute,
            static_cast<ElementB>(scaleRatio));
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
        UbInListId = nextStage;
    }

    FTSELF_DEVICE
    void StoreMeanOutputs(AscendC::GlobalTensor<ElementY> const &gmMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmMeanSquare, LayoutX const &layoutX)
    {
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);
        auto layoutY = layoutX.GetTileLayout(TensorCoord(tileActual.y));
        vecCopyUbToGmforB(gmMeanAbs, UbYTensorforBMeanList[UbOutListId], layoutY, layoutY);
        vecCopyUbToGmforB(gmMeanSquare, UbYTensorforBVarList[UbOutListId], layoutY, layoutY);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForB);
        UbOutListId = NextStage(UbOutListId);
    }

    FTSELF_DEVICE
    uint32_t ConfigureStrideSampling(uint32_t samplingStride, uint32_t strideUnit)
    {
        samplingStride = samplingStride == 0 ? 1 : samplingStride;
        uint32_t strideUnitAligned = FTSelf::helper::RoundUp(strideUnit, UBAlignHelper::ALIGN);
        uint32_t chunkCount = TileNRound / (strideUnitAligned * samplingStride);
        if (chunkCount < 1) {
            strideUnitAligned = FTSelf::helper::RoundUp(TileNRound / samplingStride, UBAlignHelper::ALIGN);
            chunkCount = 1;
        }
        TileNRound = chunkCount * strideUnitAligned * samplingStride;
        TileNRoundSimpling = (samplingStride < 2) ? TileNRound : chunkCount * strideUnitAligned;
        return strideUnitAligned;
    }

    FTSELF_DEVICE
    uint32_t GetStridedActualN(uint32_t loopIdx, uint32_t loopCount,
        uint32_t samplingStride, uint32_t strideUnit) const
    {
        if (loopIdx < loopCount - 1) {
            return TileNRoundSimpling;
        }
        if (tileActual.n <= TileNRoundSimpling) {
            return tileActual.n;
        }
        uint32_t chunkSize = strideUnit * samplingStride;
        uint32_t complete = (tileActual.n / chunkSize) * strideUnit;
        uint32_t remain = tileActual.n % chunkSize;
        return complete + ((remain < strideUnit) ? remain : strideUnit);
    }

    template <bool Strided>
    FTSELF_DEVICE
    void ReduceSampledMean(AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare, LayoutX const &layoutX,
        GemvCoord const &actualShape, float nScaleRatio, uint32_t samplingStride,
        uint32_t strideUnitAligned)
    {
        samplingStride = samplingStride == 0 ? 1 : samplingStride;
        uint32_t loopCount = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t loopIdx = 0; loopIdx < loopCount; loopIdx++) {
            tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
            tileActual.SetN((loopIdx == loopCount - 1) ?
                (actualShape.n() - loopIdx * TileNRound) : TileNRound);
            if constexpr (Strided) {
                n_actual_local = GetStridedActualN(
                    loopIdx, loopCount, samplingStride, strideUnitAligned);
            } else {
                n_actual_local = (tileActual.n > TileNRoundSimpling) ?
                    (tileActual.n / samplingStride) : tileActual.n;
            }

            uint32_t nextInputStage = NextStage(UbInListId);
            if (loopIdx < loopCount - 1) {
                PreloadNextSampledB<Strided>(nextInputStage, gmB, layoutB, loopIdx,
                    loopCount, actualShape, samplingStride, strideUnitAligned);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRoundSimpling));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(tileActual.m, n_actual_local));
            ReduceMeanTile(layoutComputeInUb, layoutTileCompute, nScaleRatio, nextInputStage);
        }

        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Muls(UbYTensorforBMeanList[UbOutListId], UbYTensorforBMeanList[UbOutListId],
            static_cast<ElementY>(samplingStride), tileActual.y);
        AscendC::Muls(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId],
            static_cast<ElementY>(samplingStride), tileActual.y);
        FTSelf::Gemv::helper::VectorBarrier();
        StoreMeanOutputs(gmZMeanAbs, gmZMeanSquare, layoutX);
    }

public:
    /// Destructor
    FTSELF_DEVICE
    ~BlockSliceKMNSum()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].input);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForA);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[i].outputForA);

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[i].outputForB);

        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {

        auto config = detail::MakeSliceKmnFusedLoopConfig<UBTileShapeforA, UBAlignHelper>(layoutA, layoutY);
        TileMRound = config.tileM;
        TileNRound = config.tileN;
        strideA = config.strideA;
        strideY = config.strideY;
        strideSlice = config.strideSlice;

        tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        tileActual.SetN((actualShape.n() < TileNRound) ? actualShape.n() : TileNRound);

        // main loop
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[UbOutListId].outputForA);
        auto UbYTensor = UbYTensorforAList[UbOutListId];

        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
        matrixCopyGmToUbforA(UbYTensor, gmY, layoutYInUb, layoutTileY);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].outputForA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE3>(stageEvents[UbOutListId].outputForA);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {

            tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
            tileActual.SetN((LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound);

            uint32_t Y_block_offset = LoopIdx * strideY;
            uint32_t UbOutListIdNext = NextStage(UbOutListId);
            if (LoopIdx < Nloop - 1) {
                CopyNextY(UbOutListIdNext, LoopIdx, Nloop, gmY, layoutY, actualShape);
                FinishYPreload(UbOutListIdNext);
            }

            if(LoopIdx == 0){
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
                matrixCopyGmToUbforA(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].outputForA);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForA);

            auto UbYIterTensor = UbYTensorforAList[UbOutListId];

            for(uint32_t kFTRoundIdx = 0; kFTRoundIdx < KFTRoundCount; kFTRoundIdx++){
                uint32_t kOffset = kFTRoundIdx * strideSlice;
                uint32_t kFTRoundIdxNext = kFTRoundIdx + 1;
                uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
                if(kFTRoundIdx < KFTRoundCount - 1){
                    // Preload next K FT round data
                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].input);
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdx * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].input);
                }else if(LoopIdx < (Nloop - 1)){
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t kFTRoundIdxNext = 0;
                    uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                    // Preload next N loop first K FT round data
                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].input);
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    uint32_t m_actual_next = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
                    uint32_t n_actual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdxNext * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].input);
                }

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));

                auto UbATensor = UbATensorList[UbInListId];
                tileMatrixAdd(UbYIterTensor,UbATensor,UbYIterTensor,layoutComputeInUb,layoutTileCompute);
                FinishInputStage(UbInListIdNext);
            }

            StoreATile(gmY, layoutY, Y_block_offset);
            // AscendC::PipeBarrier<PIPE_MTE3>();

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[UbOutListId].outputForA);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForA);
            UbOutListId = UbOutListIdNext;
        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void add_ae_op(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementA> const &gmXforAe, LayoutX const &layoutXforAe,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmYforAe, LayoutX const &layoutYforAe,
        GemvCoord const &actualShape, uint32_t NRealRound, uint32_t KFTRoundCount)
    {
        auto config = detail::MakeSliceAeConfig<UBTileShapeforA, UBAlignHelper>(layoutA, layoutY);
        TileMRound = config.tileM;
        TileNRound = config.tileN;
        // FTSelf::helper::RoundUp(NRealRound, UBAlignHelper::ALIGN);
        strideA = config.strideA;
        strideY = config.strideY;
        strideSlice = config.strideSlice;
        strideXforAe = config.strideXforAe;
        strideYforAe = config.strideYforAe;
        strideSliceforAe = config.strideSliceforAe;

        tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        tileActual.SetN((actualShape.n() < TileNRound) ? actualShape.n() : TileNRound);

        // main loop
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[UbOutListId].outputForA);
        auto UbYTensor = UbYTensorforAList[UbOutListId];
        auto UbYTensorforAe = UbYTensorforAeList[UbOutListId];

        auto layoutYInUb = LayoutY::template MakeLayoutInUb<ElementY>(MakeCoord(TileMRound, TileNRound));
        auto layoutTileY = layoutY.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
        matrixCopyGmToUbforA(UbYTensor, gmY, layoutYInUb, layoutTileY);
        vecCopyGmToUbforX(UbYTensorforAe, gmYforAe, tileActual.n);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].outputForA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_MTE3>(stageEvents[UbOutListId].outputForA);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {

            tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
            tileActual.SetN((LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound);

            uint32_t Y_block_offset = LoopIdx * strideY;
            uint32_t Y_block_offset_for_Ae = LoopIdx * strideYforAe;
            uint32_t UbOutListIdNext = NextStage(UbOutListId);

            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t n_actual_next = CopyNextY(
                    UbOutListIdNext, LoopIdx, Nloop, gmY, layoutY, actualShape);
                auto UbYTensorforAeNext = UbYTensorforAeList[UbOutListIdNext];
                vecCopyGmToUbforX(UbYTensorforAeNext, gmYforAe[LoopIdxNext * strideYforAe], n_actual_next);
                FinishYPreload(UbOutListIdNext);
            }

            if(LoopIdx == 0){
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
                matrixCopyGmToUbforA(UbATensorList[UbInListId], gmA[LoopIdx * strideA], layoutAInUb, layoutTileA);
                vecCopyGmToUbforX(UbXTensorforAeList[UbInListId], gmXforAe[LoopIdx * strideXforAe], tileActual.n);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].outputForA);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForA);

            auto UbYIterTensor = UbYTensorforAList[UbOutListId];
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
                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].input);
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdx * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    vecCopyGmToUbforX(UbXTensorforAeList[UbInListIdNext], gmXforAe[LoopIdx * strideXforAe + kOffsetforAeNext], tileActual.n);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].input);
                }else if(LoopIdx < (Nloop - 1)){
                    uint32_t LoopIdxNext = LoopIdx + 1;
                    uint32_t kFTRoundIdxNext = 0;

                    uint32_t kOffsetNext = kFTRoundIdxNext * strideSlice;
                    uint32_t kOffsetforAeNext = kFTRoundIdxNext * strideSliceforAe;
                    // Preload next N loop first K FT round data
                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].input);
                    auto layoutAInUbNext = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    uint32_t m_actual_next = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
                    uint32_t n_actual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                    auto layoutTileANext = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUbforA(UbATensorList[UbInListIdNext], gmA[LoopIdxNext * strideA + kOffsetNext], layoutAInUbNext, layoutTileANext);
                    vecCopyGmToUbforX(UbXTensorforAeList[UbInListIdNext], gmXforAe[LoopIdxNext * strideXforAe + kOffsetforAeNext], n_actual_next);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].input);
                }

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
                auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));

                auto UbATensor = UbATensorList[UbInListId];
                auto UbXTensorforAe = UbXTensorforAeList[UbInListId];
                tileMatrixAdd(UbYIterTensor, UbATensor, UbYIterTensor, layoutComputeInUb, layoutTileCompute);

                AscendC::Add(UbYIterTensorforAe, UbXTensorforAe, UbYIterTensorforAe, tileActual.n);
                FinishInputStage(UbInListIdNext);
            }

            StoreATile(gmY, layoutY, Y_block_offset);
            // AscendC::PipeBarrier<PIPE_MTE3>();
            auto layoutDstYforAe = layoutYforAe.GetTileLayout(TensorCoord(tileActual.n));
            auto layoutYforAeInUb = layoutYforAe.GetTileLayout(TensorCoord(tileActual.n));
            vecCopyUbToGmforY(gmYforAe[Y_block_offset_for_Ae],
                UbYTensorforAeList[UbOutListId], layoutDstYforAe, layoutYforAeInUb);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[UbOutListId].outputForA);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForA);

            UbOutListId = UbOutListIdNext;
        }
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void RowMeanAbsSquare(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio)
    {
        TileMRound = FTSelf::helper::RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        PrepareMeanOutputs(layoutB, actualShape);

        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);
        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);

        LoadBTile(UbInListId, UbBMeanTensorList[UbInListId], gmB,
            layoutB, 0, tileActual.m, tileActual.n);
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            uint32_t UbInListIdNext = BeginBLoop(LoopIdx, Nloop, actualShape);
            if (LoopIdx < Nloop - 1) {
                PreloadNextB(UbInListIdNext, UbBMeanTensorList[UbInListIdNext],
                    gmB, layoutB, LoopIdx, Nloop, actualShape);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));


            ReduceMeanTile(layoutComputeInUb, layoutTileCompute, n_scale_ratio, UbInListIdNext);
        }
        StoreMeanOutputs(gmZMeanAbs, gmZMeanSquare, layoutX);
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void RowMeanAbsSquareWithSimpleCon(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio, uint32_t simpling_stride)
    {

        TileMRound = FTSelf::helper::RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileMRoundSimpling = TileMRound;

        TileNRound = FTSelf::helper::RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        TileNRoundSimpling = (simpling_stride < 2) ? TileNRound :
            (TileNRound / (simpling_stride == 0 ? 1 : simpling_stride));
        TileNRoundSimpling = FTSelf::helper::RoundUp(TileNRoundSimpling, UBAlignHelper::ALIGN);

        uint32_t simpling_stride_round = (simpling_stride < 2) ? 1 : simpling_stride;
        // TileNRound / TileNRoundSimpling;
        TileNRound = TileNRoundSimpling * simpling_stride_round;

        // FTSelf::helper::RoundUp(NRealRound, UBAlignHelper::ALIGN);
        PrepareMeanOutputs(layoutB, actualShape);
        LoadSampledB<false>(UbInListId, gmB, layoutB, 0, tileActual.n,
            simpling_stride_round, 0);

        ReduceSampledMean<false>(gmB, layoutB, gmZMeanAbs, gmZMeanSquare, layoutX,
            actualShape, n_scale_ratio, simpling_stride_round, 0);
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void RowMeanAbsSquareWithSimpleStride(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmZMeanSquare,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio,
        uint32_t simpling_stride, uint32_t stride_unit)
    {
        simpling_stride = simpling_stride == 0 ? 1 : simpling_stride;


        TileMRound = FTSelf::helper::RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileMRoundSimpling = TileMRound;

        TileNRound = FTSelf::helper::RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        uint32_t stride_unit_aligned = ConfigureStrideSampling(simpling_stride, stride_unit);
        uint32_t simpling_stride_round = (simpling_stride < 2) ? 1 : simpling_stride;
        PrepareMeanOutputs(layoutB, actualShape);
        LoadSampledB<true>(UbInListId, gmB, layoutB, 0, tileActual.n,
            simpling_stride_round, stride_unit_aligned);

        ReduceSampledMean<true>(gmB, layoutB, gmZMeanAbs, gmZMeanSquare, layoutX,
            actualShape, n_scale_ratio, simpling_stride_round, stride_unit_aligned);
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void RowVariance(
        AscendC::GlobalTensor<ElementB> const &gmBMean,
        AscendC::GlobalTensor<ElementB> const &gmBMax,
        AscendC::GlobalTensor<ElementB> const &gmBMin,
        LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZVar,
        LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float n_scale_ratio)
    {
        PrepareBReduction(layoutB, actualShape);

        auto UbYTensorforVar = UbYTensorforBVarList[UbOutListId];

        FTSelf::Gemv::helper::ZeroTensor(UbYTensorforVar, tileActual.m);
        FTSelf::Gemv::helper::VectorBarrier();

        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);
        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
        auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));

        matrixCopyGmToUbforB(UbBMeanTensorList[UbInListId], gmBMean, layoutBInUb, layoutTileB);
        matrixCopyGmToUbforB(UbBMaxTensorList[UbInListId], gmBMax, layoutBInUb, layoutTileB);
        matrixCopyGmToUbforB(UbBMinTensorList[UbInListId], gmBMin, layoutBInUb, layoutTileB);

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
        // main loop
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
            tileActual.SetN((LoopIdx == (Nloop - 1)) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound);

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = tileActual.m;
                uint32_t n_actual_next =
                    (LoopIdxNext == (Nloop - 1)) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensorforMean = UbBMeanTensorList[UbInListIdNext];
                auto matrixTensorforMax = UbBMaxTensorList[UbInListIdNext];
                auto matrixTensorforMin = UbBMinTensorList[UbInListIdNext];

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].input);
                auto layoutBInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUbforB(matrixTensorforMean, gmBMean[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                matrixCopyGmToUbforB(matrixTensorforMax, gmBMax[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                matrixCopyGmToUbforB(matrixTensorforMin, gmBMin[LoopIdxNext * strideB], layoutBInUb, layoutTileB);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].input);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));


            tileReduceVarforB(UbYTensorforBVarList[UbOutListId],
                UbBMeanTensorList[UbInListId],
                UbBMaxTensorList[UbInListId],
                UbBMinTensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute,
                (ElementB)n_scale_ratio);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
            UbInListId = UbInListIdNext;
        }

        FTSelf::Gemv::helper::VectorBarrier();

        // AscendC::Sqrt(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], tileActual.m);
        AscendC::Abs(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], tileActual.m);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Sqrt(UbYTensorforBVarList[UbOutListId], UbYTensorforBVarList[UbOutListId], tileActual.m);
        FTSelf::Gemv::helper::VectorBarrier();
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);

        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(tileActual.y));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(tileActual.y));

        vecCopyUbToGmforB(gmZVar, UbYTensorforBVarList[UbOutListId], layoutDstY, layoutComputeInUb);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForB);
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void RowMean(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float kn_scale_ratio)
    {
        uint32_t tileN = FTSelf::helper::RoundUp(UBTileShapeforB::N, UBAlignHelper::ALIGN);
        ReduceRows<true>(gmB, layoutB, gmZ, layoutX, actualShape,
            tileN, kn_scale_ratio, UbBMeanTensorList, tileFaultMeanforB);
    }

    // AscendC::GlobalTensor<ElementY> const &gmZ,
    FTSELF_DEVICE
    void RowMax(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutX const &layoutX,
        GemvCoord const &actualShape,
        uint32_t NRealRound,
        float kn_scale_ratio)
    {
        ReduceRows<false>(gmB, layoutB, gmZ, layoutX, actualShape,
            NRealRound, kn_scale_ratio, UbBMaxTensorList, tileFaultMaxforB);
    }

private:
    template <bool SCALE_RESULT, class Tensor, class TileReduce>
    FTSELF_DEVICE
    void ReduceRows(
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutX const &layoutX,
        GemvCoord const &actualShape, uint32_t tileN, float scaleRatio,
        Tensor (&inputList)[STAGES], TileReduce &tileReduce)
    {
        TileMRound = FTSelf::helper::RoundUp(UBTileShapeforB::M, UBAlignHelper::ALIGN);
        TileNRound = tileN;
        strideB = layoutB.stride(1) * TileNRound;
        tileActual.SetM((actualShape.m() < TileMRound) ? actualShape.m() : TileMRound);
        tileActual.SetN((actualShape.n() < TileNRound) ? actualShape.n() : TileNRound);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForB);

        auto UbYTensor = UbYTensorforBMeanList[UbOutListId];

        FTSelf::Gemv::helper::ZeroTensor(UbYTensor, tileActual.m);
        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].outputForB);

        LoadBTile(UbInListId, inputList[UbInListId], gmB,
            layoutB, 0, tileActual.m, tileActual.n);
        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            uint32_t UbInListIdNext = BeginBLoop(LoopIdx, Nloop, actualShape);
            if (LoopIdx < Nloop - 1) {
                PreloadNextB(UbInListIdNext, inputList[UbInListIdNext],
                    gmB, layoutB, LoopIdx, Nloop, actualShape);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].input);
            auto layoutComputeInUb = layoutB.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutB.GetTileLayout(MakeCoord(tileActual.m, tileActual.n));


            tileReduce(UbYTensorforBMeanList[UbOutListId],
                inputList[UbInListId],
                UbWTensorforRawList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].input);
            UbInListId = UbInListIdNext;
        }
        FTSelf::Gemv::helper::VectorBarrier();

        if constexpr (SCALE_RESULT) {
            tileVmuls(UbYTensorforBMeanList[UbOutListId], UbYTensorforBMeanList[UbOutListId],
                static_cast<ElementY>(scaleRatio), tileActual.m);
        }

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].outputForB);
        auto layoutDstY = layoutX.GetTileLayout(TensorCoord(tileActual.y));
        auto layoutComputeInUb = layoutX.GetTileLayout(TensorCoord(tileActual.y));
        vecCopyUbToGmforB(gmZ, UbYTensorforBMeanList[UbOutListId], layoutDstY, layoutComputeInUb);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(stageEvents[UbOutListId].outputForB);
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];

    AscendC::LocalTensor<ElementB> UbBMeanTensorList[STAGES];
    AscendC::LocalTensor<ElementB> UbBMaxTensorList[STAGES];
    AscendC::LocalTensor<ElementB> UbBMinTensorList[STAGES];

    AscendC::LocalTensor<ElementA> UbXTensorforAeList[STAGES];

    AscendC::LocalTensor<ElementY> UbYTensorforAList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforBVarList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorforBMeanList[STAGES];

    AscendC::LocalTensor<ElementY> UbYTensorforAeList[STAGES];
    AscendC::LocalTensor<ElementY> UbWTensorList[STAGES];
    AscendC::LocalTensor<ElementB> UbWTensorforRawList[STAGES];
    // AscendC::LocalTensor<ElementA> UbWTensorList[STAGES];

    struct StageEvents {
        int32_t input;
        int32_t outputForA;
        int32_t outputForB;
    };

    StageEvents stageEvents[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    FTSelf::Gemv::TileActualShape tileActual;
    uint32_t n_actual_local;
    uint32_t TileMRound, TileNRound;
    uint32_t TileMRoundSimpling, TileNRoundSimpling;
    uint32_t strideA, strideY, strideB;
    uint32_t strideSlice;
    uint32_t strideXforAe, strideYforAe;
    uint32_t strideSliceforAe;

    TileMatrixAdd tileMatrixAdd;

    TileFaultMeanforB tileFaultMeanforB;
    TileFaultMaxforB tileFaultMaxforB;

    TileReduceMeanforB tileReduceMeanforB;
    TileReduceVarforB tileReduceVarforB;

    TileVmuls tileVmuls;

    MatrixCopyGmToUbforB matrixCopyGmToUbforB;

    MatrixCopyGmToUbSimplingContinueforB matrixCopyGmToUbSimplingCforB;
    MatrixCopyGmToUbSimplingStridedforB matrixCopyGmToUbSimplingSforB;

    VecCopyGmToUbforB vecCopyGmToUbforB;
    VecCopyUbToGmforB vecCopyUbToGmforB;

    VecCopyGmToUbforX vecCopyGmToUbforX;
    VecCopyUbToGmforY vecCopyUbToGmforY;

    MatrixCopyGmToUbforA matrixCopyGmToUbforA;

    MatrixCopyUbToGmforA matrixCopyUbToGmforA;

};

} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
