/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_SLICE_REDUCE_SUM_AIV_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_SLICE_REDUCE_SUM_AIV_HPP

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "block_slice_common.hpp"

namespace FTSelf::Gemv::Block {


// class TileVmuls_

template <
    class UBTileShape_,
    class AType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileSliceSum_
>
struct BlockSliceSum <
    FTSelf::Gemv::GemvAtlasA2,
    UBTileShape_,
    AType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileSliceSum_
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
    using TileSliceSum = TileSliceSum_;
    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Abuf_SIZE_ = 160 * 1024;
    // static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 32 * 1024;
    // static constexpr uint32_t workspace_SIZE_ = 48 * 1024;

    static constexpr uint32_t ELE_NUM_PER_REPEAT = FTSelf::Gemv::BYTE_PER_C0 * 8 / sizeof(ElementY);

    FTSELF_DEVICE
    BlockSliceSum() {}

    /// Construct
    FTSELF_DEVICE
    BlockSliceSum(FTSelf::Arch::Resource<FTSelf::Arch::AtlasA2> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

    /// Construct
    FTSELF_DEVICE
    BlockSliceSum(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

private:
    template <class Resource>
    FTSELF_DEVICE
    void InitializeBuffers(Resource &resource, uint32_t UBufAddrStart)
    {
        auto offsets = detail::MakeSliceBufferOffsets(UBufAddrStart, Abuf_SIZE_);
        uint32_t UbAOffset = offsets.a;
        // + Xbuf_SIZE_ + Xbuf_SIZE_
        uint32_t UbYOffset = offsets.y;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset + i * (Abuf_SIZE_ / 2));
            // UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            // UbWTensorList[i] =
            //     resource.ubBuf.template GetBufferByByte<ElementX>(UbWOffset + i * (workspace_SIZE_ / 2));

            // Assign event ID for each stages
            UbInAEventList[i] = i;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

public:
    /// Destructor
    FTSELF_DEVICE
    ~BlockSliceSum()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,

    FTSELF_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementY> const &gmZ, LayoutY const &layoutY,
        GemvCoord const &actualShape)
    {
        TileMRound = FTSelf::helper::RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        strideA = layoutA.stride(0) * TileMRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m() : TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);

        /*
        void Duplicate(const LocalTensor<T>& dstLocal, const T& scalarValue, const int32_t& calCount)
        */
        FTSelf::Gemv::helper::ZeroTensor(UbYTensorList[UbOutListId], actualShape.n());

        FTSelf::Gemv::helper::VectorBarrier();

        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);
        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);

        // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[UbInListId]);
        // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInXEventList[UbInListId]);

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListId]);
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListId]);
        // main loop
        uint32_t Mloop = FTSelf::helper::CeilDiv(actualShape.m(), TileMRound);
        for (uint32_t LoopIdx = 0; LoopIdx < Mloop; LoopIdx++) {
            n_actual = (actualShape.n() < TileNRound) ? actualShape.n() : TileNRound;
            m_actual = (LoopIdx == Mloop - 1) ? (actualShape.m() - LoopIdx * TileMRound) : TileMRound;
            y_actual = n_actual;
            x_actual = m_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if (LoopIdx < Mloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t n_actual_next = n_actual;
                uint32_t m_actual_next =
                    (LoopIdxNext == Mloop - 1) ? (actualShape.m() - LoopIdxNext * TileMRound) : TileMRound;
                uint32_t y_actual_next = n_actual_next;
                uint32_t x_actual_next = m_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbATensorList[UbInListIdNext];

                // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[UbInListIdNext]);
                // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInXEventList[UbInListIdNext]);

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListIdNext]);
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(matrixTensor, gmA[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListIdNext]);
            }
            // FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInXEventList[UbInListId]);

            // FTSelf::Gemv::helper::VectorBarrier();

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInAEventList[UbInListId]);
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));

            //     UbXTensorList[UbInListId],
            //     UbATensorList[UbInListId],
            //     UbWTensorList[UbInListId],
            //     layoutComputeInUb,
            //     layoutTileCompute);



            tileSliceSum(
                UbYTensorList[UbOutListId],
                UbATensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInAEventList[UbInListId]);
            // FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[UbInListId]);
            UbInListId = UbInListIdNext;
        }

        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);

        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(n_actual));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(n_actual));
        vecCopyUbToGm(gmZ, UbYTensorList[UbOutListId], layoutDstY, layoutComputeInUb);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    // AscendC::LocalTensor<ElementX> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInAEventList[STAGES];
    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbInListId{0};

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t TileMRound, TileNRound;
    uint32_t strideA;

    // TileVmad tileVmad;
    TileSliceSum tileSliceSum;
    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;
};




} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
