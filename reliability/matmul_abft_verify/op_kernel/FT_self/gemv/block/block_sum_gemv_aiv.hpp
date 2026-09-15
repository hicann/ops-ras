/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_SUM_GEMV_AIV_NOW_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_SUM_GEMV_AIV_NOW_HPP

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "../../gemv/tile/tile_vmad.hpp"

namespace FTSelf::Gemv::Block {


template <
    class UBTileShape_,
    class AType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileVmad_,
    class TileVmuls_
>
struct BlockSumGemv <
    FTSelf::Gemv::GemvAtlasA2,
    UBTileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileVmad_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using UBTileShape = UBTileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using LayoutACol = typename std::conditional<
        std::is_same<LayoutA, FTSelf::layout::RowMajor>::value,
        FTSelf::layout::ColumnMajor,
        FTSelf::layout::RowMajor>::type;
    using AColType = FTSelf::GemmType<ElementA, LayoutACol>;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using TileVmad = TileVmad_;
    using TileVmuls = TileVmuls_;
    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;

    using TileCopyCol_ = FTSelf::Gemv::Tile::TileCopyGemvAiv<FTSelf::Gemv::Arch::AtlasA2, AColType, XType_, YType_, BiasType_>;
    using TileVmadCol = FTSelf::Gemv::Tile::TileVmad<FTSelf::Gemv::Arch::AtlasA2, AColType, XType_, YType_, BiasType_>;
    using VecCopyGmToUbCol = typename TileCopyCol_::VecCopyGmToUb;
    using VecCopyUbToGmCol = typename TileCopyCol_::VecCopyUbToGm;
    using MatrixCopyGmToUbCol = typename TileCopyCol_::MatrixCopyGmToUb;

    using LayoutAInUbCol = typename MatrixCopyGmToUbCol::LayoutDst;

    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;

    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;

    FTSELF_DEVICE
    BlockSumGemv() {}

    /// Construct
    FTSELF_DEVICE
    BlockSumGemv(FTSelf::Arch::Resource<FTSelf::Arch::AtlasA2> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

    /// Construct
    FTSELF_DEVICE
    BlockSumGemv(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

    /// Destructor
    FTSELF_DEVICE
    ~BlockSumGemv()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputA);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputX);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[i].output);
        }
    }

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmZ,
        GemvCoord const &actualShape,
        float alpha,
        float beta)
    {
        RunGemv(gmA, layoutA, gmX, layoutX, gmY, layoutY, gmZ, actualShape, alpha, beta,
            matrixCopyGmToUb, vecCopyGmToUb, vecCopyUbToGm, tileVmad);
    }

    FTSELF_DEVICE
    void GemvCol(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutACol const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmZ,
        GemvCoord const &actualShape,
        float alpha,
        float beta)
    {
        RunGemv(gmA, layoutA, gmX, layoutX, gmY, layoutY, gmZ, actualShape, alpha, beta,
            matrixCopyGmToUbCol, vecCopyGmToUbCol, vecCopyUbToGmCol, tileVmadCol);
    }

private:
    template <class Resource>
    FTSELF_DEVICE
    void InitializeBuffers(Resource &resource, uint32_t UBufAddrStart)
    {
        uint32_t UbAOffset = UBufAddrStart;
        uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
        uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_ + Xbuf_SIZE_ + Ybuf_SIZE_;
        for (uint32_t i = 0; i < STAGES; i++) {
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(
                UbWOffset + i * (workspace_SIZE_ / 2));
            stageEvents[i].inputA = i;
            stageEvents[i].inputX = i + STAGES;
            stageEvents[i].output = i;
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputA);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[i].inputX);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[i].output);
        }
    }

    template <class MatrixLayout, class MatrixCopy, class VecGmToUb, class VecUbToGm, class Vmad>
    FTSELF_DEVICE
    void RunGemv(AscendC::GlobalTensor<ElementA> const &gmA, MatrixLayout const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmZ, GemvCoord const &actualShape,
        float alpha, float beta, MatrixCopy &matrixCopy, VecGmToUb &copyGmToUb,
        VecUbToGm &copyUbToGm, Vmad &vmad)
    {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[UbOutListId].output);
        copyGmToUb(UbYTensorList[UbOutListId], gmY, actualShape.m());
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].output);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbOutListId].output);
        tileVmuls(UbYTensorList[UbOutListId], UbYTensorList[UbOutListId], (ElementY)beta, actualShape.m());
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].output);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbOutListId].output);

        TileMRound = FTSelf::helper::RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputX);
        copyGmToUb(UbXTensorList[UbInListId], gmX, n_actual);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputX);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputA);
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopy(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputA);

        uint32_t Nloop = FTSelf::helper::CeilDiv(actualShape.n(), TileNRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;
            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t mActualNext = m_actual;
                uint32_t nActualNext = (LoopIdxNext == Nloop - 1) ?
                    (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                auto matrixTensor = UbATensorList[UbInListIdNext];
                auto vecTensor = UbXTensorList[UbInListIdNext];
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].inputX);
                copyGmToUb(vecTensor, gmX[LoopIdxNext * TileNRound], nActualNext);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].inputX);
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListIdNext].inputA);
                auto nextLayoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto nextLayoutTileA = layoutA.GetTileLayout(MakeCoord(mActualNext, nActualNext));
                matrixCopy(matrixTensor, gmA[LoopIdxNext * strideA], nextLayoutAInUb, nextLayoutTileA);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListIdNext].inputA);
            }
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputX);
            tileVmuls(UbXTensorList[UbInListId], UbXTensorList[UbInListId], (ElementA)alpha, x_actual);
            FTSelf::Gemv::helper::VectorBarrier();
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(stageEvents[UbInListId].inputA);
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            vmad(UbYTensorList[UbOutListId], UbXTensorList[UbInListId], UbATensorList[UbInListId],
                UbWTensorList[UbInListId], layoutComputeInUb, layoutTileCompute);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputA);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(stageEvents[UbInListId].inputX);
            UbInListId = UbInListIdNext;
        }
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(stageEvents[UbOutListId].output);
        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_actual));
        copyUbToGm(gmZ, UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(stageEvents[UbOutListId].output);
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> UbWTensorList[STAGES];

    // Multi-stage event id list
    struct StageEvents {
        int32_t inputA;
        int32_t inputX;
        int32_t output;
    };

    StageEvents stageEvents[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t TileMRound, TileNRound;
    uint32_t strideA;

    TileVmad tileVmad;
    TileVmuls tileVmuls;
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    TileVmadCol tileVmadCol;
    MatrixCopyGmToUbCol matrixCopyGmToUbCol;
    VecCopyGmToUbCol vecCopyGmToUbCol;
    VecCopyUbToGmCol vecCopyUbToGmCol;
};

} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
