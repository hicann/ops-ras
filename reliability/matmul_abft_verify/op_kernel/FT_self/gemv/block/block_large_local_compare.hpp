/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*
一共开发两个功能，即1）进行作差；2）进行异或比较
在这里因为是简单的比较两个变量，所以不需要区分矩阵和向量了
直接加载一块连续的GM空间，使用AIV 核，
对于每个AIV核，结果直接存放到GM的相应位置上即可。

在这里我定义Z为输出的位置，X 为操作数向量1，Y为操作数向量2，即比较两个操作数
我这里先给出
*/

#ifndef FTSELF_GEMV_BLOCK_LARGE_LOCAL_COPARE_AIV_HPP
#define FTSELF_GEMV_BLOCK_LARGE_LOCAL_COPARE_AIV_HPP

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "../../gemv/tile/tile_fault_compare.hpp"
#include "../../gemv/block/block_gemv.hpp"

namespace FTSelf::Gemv::Block {

template <class UBTileShape, class ElementX, class ElementWork>
struct BlockCompareLoopConfig {
    uint32_t actualLength;
    uint32_t actualNextLength;
    uint32_t tileRound;
    uint32_t loopCount;
    uint32_t nextLoopCount;
    uint32_t tileOutputCount;
};

template <class UBTileShape, class UBAlignHelper, class ElementX, class ElementWork>
FTSELF_DEVICE
BlockCompareLoopConfig<UBTileShape, ElementX, ElementWork> MakeBlockCompareLoopConfig(
    GemvCoord const &actualShape, GemvCoord const &actualShapeNext)
{
    BlockCompareLoopConfig<UBTileShape, ElementX, ElementWork> config;
    config.actualLength = actualShape.m() * actualShape.n();
    config.actualNextLength = actualShapeNext.m() * actualShapeNext.n();
    config.tileRound = FTSelf::helper::RoundUp(UBTileShape::M * UBTileShape::N, UBAlignHelper::ALIGN);
    config.loopCount = FTSelf::helper::CeilDiv(config.actualLength, config.tileRound);
    config.nextLoopCount = FTSelf::helper::CeilDiv(config.actualNextLength, config.tileRound);
    config.tileOutputCount = config.tileRound / 8;
    return config;
}

template <class ElementZ, class LayoutZ, class VecCopyUbToGmZ>
FTSELF_DEVICE
void StoreBlockCompareResult(AscendC::GlobalTensor<ElementZ> const &gmZ,
    AscendC::LocalTensor<ElementZ> ubZ, LayoutZ const &layoutZ, VecCopyUbToGmZ &copy,
    int32_t eventId, uint32_t outputOffset, uint32_t elementCount)
{
    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(eventId));
    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(eventId));
    uint32_t outputCount = elementCount / 8;
    auto layoutDst = layoutZ.GetTileLayout(FTSelf::layout::VectorLayout::TensorCoord(outputCount));
    auto layoutSrc = layoutZ.GetTileLayout(FTSelf::layout::VectorLayout::TensorCoord(outputCount));
    copy(gmZ[outputOffset], ubZ, layoutDst, layoutSrc);
    AscendC::PipeBarrier<PIPE_MTE3>();
    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(eventId);
}

#define FTSELF_IMPORT_BLOCK_COMPARE_TYPES                                                        \
    using DispatchPolicy = FTSelf::Gemv::GemvAtlasA2;                                            \
    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;                                     \
    using ArchTag = typename DispatchPolicy::ArchTag;                                             \
    using UBTileShape = UBTileShape_;                                                             \
    using ElementZ = typename ZType_::Element;                                                    \
    using LayoutZ = typename ZType_::Layout;                                                      \
    using ElementX = typename XType_::Element;                                                    \
    using LayoutX = typename XType_::Layout;                                                      \
    using ElementY = typename YType_::Element;                                                    \
    using LayoutY = typename YType_::Layout


// FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
template <
    class UBTileShape_,
    class ZType_,
    class XType_,
    class YType_,
    class TileCopy_
>
struct BlockCompare <
    FTSelf::Gemv::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
    UBTileShape_,
    ZType_,
    XType_,
    YType_,
    TileCopy_
> {
public:
    // Type Aliases
    FTSELF_IMPORT_BLOCK_COMPARE_TYPES;



    using ElementWork = ElementX;

    using VecCopyGmToUbX = typename TileCopy_::VecCopyGmToUbX;
    using VecCopyGmToUbY = typename TileCopy_::VecCopyGmToUbY;
    using VecCopyGmToUbW = typename TileCopy_::VecCopyGmToUbW;

    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;
    using VecCopyUbToGmW = typename TileCopy_::VecCopyUbToGmW;

    /*
    <
    /// Tag indicating architecture
    FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class ArchTag,
    class ZType,
    class XType,
    class YType
    >
    */

    using TileCompare = FTSelf::Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag,
                                        ZType_, XType_, YType_>;


    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementX>;
    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t Xbuf_SIZE_ = 60 * 1024;
    static constexpr uint32_t Ybuf_SIZE_ = 60 * 1024;
    static constexpr uint32_t Wbuf_SIZE_ = 60 * 1024;
    // static constexpr uint32_t workspace_SIZE_ = 16 * 1024;
    static constexpr uint32_t Zbuf_SIZE_ = 12 * 1024;
    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;



    FTSELF_DEVICE
    BlockCompare() {}

    /// Construct
    FTSELF_DEVICE
    BlockCompare(FTSelf::Arch::Resource<FTSelf::Arch::AtlasA2> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

    /// Construct
    FTSELF_DEVICE
    BlockCompare(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

private:
    template <class Resource>
    FTSELF_DEVICE
    void InitializeBuffers(Resource &resource, uint32_t UBufAddrStart)
    {
        uint32_t UbXOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Xbuf_SIZE_;
        uint32_t UbWOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_;
        // + workspace_SIZE_
        uint32_t UbZOffset = UBufAddrStart + Xbuf_SIZE_ + Ybuf_SIZE_ + Wbuf_SIZE_;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages


            UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset + i * (Xbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));

            UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementWork>(UbWOffset + i * (Wbuf_SIZE_ / 2));
            // UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementWork>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));

            // Assign event ID for each stages

            UbInXEventList[i] = i;
            UbInYEventList[i] = i + STAGES;
            UbInWEventList[i] = i + STAGES * 2;
            UbOutEventList[i] = i;

            // The event id that needs to be set before the loop
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInWEventList[i]);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

public:
    /// Destructor
    FTSELF_DEVICE
    ~BlockCompare()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInYEventList[i]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInWEventList[i]);

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
        }
    }

    // AscendC::GlobalTensor<ElementWork> const &gmWork, LayoutW const &layoutW,
    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementWork> const &gmW, LayoutX const &layoutW,
        AscendC::GlobalTensor<ElementX> const &gmXNext,
        AscendC::GlobalTensor<ElementY> const &gmYNext,
        AscendC::GlobalTensor<ElementWork> const &gmWNext,
        AscendC::GlobalTensor<ElementZ> const &gmZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape, GemvCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, bool OutputWorkspace,
        ElementX threshold)
    {
        auto config = MakeBlockCompareLoopConfig<UBTileShape, UBAlignHelper, ElementX, ElementWork>(
            actualShape, actualShapeNext);
        uint32_t actuallength = config.actualLength;
        uint32_t actualNextlength = config.actualNextLength;
        TileRound = config.tileRound;
        n_actual = (actuallength < TileRound) ? actuallength : TileRound;
        uint32_t Nloop = config.loopCount;
        uint32_t NloopNext = config.nextLoopCount;
        uint32_t tile_output_actual = config.tileOutputCount;



        for (uint32_t LoopIdx = 0; LoopIdx < Nloop; LoopIdx++) {

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[UbListId]);

            n_actual = (LoopIdx == Nloop - 1) ? (actuallength - LoopIdx * TileRound) : TileRound;
            y_actual = n_actual;
            x_actual = n_actual;

            uint32_t UbListIdNext = (UbListId + 1 < STAGES) ? (UbListId + 1) : 0;
            uint32_t LoopIdxNext{0};
            uint32_t n_actual_next{0};
            uint32_t y_actual_next{0};
            uint32_t x_actual_next{0};
            uint32_t w_actual_next{0};

            if (LoopIdx == 0 && isFirstBlock){
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[UbListId]);
                vecCopyGmToUbX(UbXTensorList[UbListId], gmX[LoopIdx * TileRound], n_actual);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInXEventList[UbListId]);

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInYEventList[UbListId]);
                vecCopyGmToUbY(UbYTensorList[UbListId], gmY[LoopIdx * TileRound], n_actual);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInYEventList[UbListId]);

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInWEventList[UbListId]);
                vecCopyGmToUbW(UbWTensorList[UbListId], gmW[LoopIdx * TileRound], n_actual);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInWEventList[UbListId]);
            }

            if (LoopIdx < Nloop - 1) {
                LoopIdxNext = LoopIdx + 1;
                n_actual_next =
                    (LoopIdxNext == Nloop - 1) ? (actuallength - LoopIdxNext * TileRound) : TileRound;

                y_actual_next = n_actual_next;
                x_actual_next = n_actual_next;
                w_actual_next = n_actual_next;

                // Get L1 tensor for next stage
                auto vecXTensor = UbXTensorList[UbListIdNext];
                auto vecYTensor = UbYTensorList[UbListIdNext];
                auto vecWTensor = UbWTensorList[UbListIdNext];

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[UbListIdNext]);
                vecCopyGmToUbX(vecXTensor, gmX[LoopIdxNext * TileRound], x_actual_next);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInXEventList[UbListIdNext]);

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInYEventList[UbListIdNext]);
                vecCopyGmToUbY(vecYTensor, gmY[LoopIdxNext * TileRound], y_actual_next);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInYEventList[UbListIdNext]);

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInWEventList[UbListIdNext]);
                vecCopyGmToUbW(vecWTensor, gmW[LoopIdxNext * TileRound], w_actual_next);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInWEventList[UbListIdNext]);
            }

            if ((LoopIdx == Nloop - 1) && hasNextBlock){

                LoopIdxNext = 0;
                n_actual_next =
                    (LoopIdxNext == NloopNext - 1) ? (actualNextlength - LoopIdxNext * TileRound) : TileRound;

                y_actual_next = n_actual_next;
                x_actual_next = n_actual_next;
                w_actual_next = n_actual_next;

                // Get L1 tensor for next stage
                auto vecXTensor = UbXTensorList[UbListIdNext];
                auto vecYTensor = UbYTensorList[UbListIdNext];
                auto vecWTensor = UbWTensorList[UbListIdNext];

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[UbListIdNext]);
                vecCopyGmToUbX(vecXTensor, gmXNext[LoopIdxNext * TileRound], x_actual_next);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInXEventList[UbListIdNext]);

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInYEventList[UbListIdNext]);
                vecCopyGmToUbY(vecYTensor, gmYNext[LoopIdxNext * TileRound], y_actual_next);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInYEventList[UbListIdNext]);

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInWEventList[UbListIdNext]);
                vecCopyGmToUbW(vecWTensor, gmWNext[LoopIdxNext * TileRound], w_actual_next);
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInWEventList[UbListIdNext]);
            }

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInXEventList[UbListId]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInYEventList[UbListId]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInWEventList[UbListId]);

            auto layoutComputeInUb = layoutX.GetTileLayout(MakeCoord(TileRound));
            auto layoutTileCompute = layoutX.GetTileLayout(MakeCoord(n_actual));

            AscendC::PipeBarrier<PIPE_MTE2>();



            // UbWTensorList[UbListId],
            tileCompare(
                UbZTensorList[UbListId],
                UbXTensorList[UbListId],
                UbYTensorList[UbListId],
                UbWTensorList[UbListId],
                layoutComputeInUb, layoutTileCompute, threshold);

            FTSelf::Gemv::helper::VectorBarrier();

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInYEventList[UbListId]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInXEventList[UbListId]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInWEventList[UbListId]);
            StoreBlockCompareResult(gmZ, UbZTensorList[UbListId], layoutZ, vecCopyUbToGmZ,
                UbOutEventList[UbListId], tile_output_actual * LoopIdx, n_actual);





            // }

            UbListId = UbListIdNext;
            // n_actual = n_actual_next;
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES];

    AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
    AscendC::LocalTensor<ElementWork> UbWTensorList[STAGES];

    // Multi-stage event id list
    int32_t UbInYEventList[STAGES];
    int32_t UbInXEventList[STAGES];
    int32_t UbInWEventList[STAGES];

    int32_t UbOutEventList[STAGES];

    // The id of current stage
    uint32_t UbListId{0};

    uint32_t n_actual, x_actual, y_actual;
    uint32_t TileRound;

    VecCopyGmToUbX vecCopyGmToUbX;
    VecCopyGmToUbY vecCopyGmToUbY;
    VecCopyGmToUbW vecCopyGmToUbW;

    VecCopyUbToGmZ vecCopyUbToGmZ;
    // VecCopyUbToGmW vecCopyUbToGmW;

    // Tile Compare
    TileCompare tileCompare;
};


#undef FTSELF_IMPORT_BLOCK_COMPARE_TYPES

} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_LARGE_LOCAL_COPARE_AIV_HPP
