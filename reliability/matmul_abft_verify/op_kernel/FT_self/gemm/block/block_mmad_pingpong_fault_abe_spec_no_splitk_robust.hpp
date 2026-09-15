/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef FTSELF_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_FAULT_ABE_SPEC_NO_SPLITK_ROBUST_HPP
#define FTSELF_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_FAULT_ABE_SPEC_NO_SPLITK_ROBUST_HPP

#include <type_traits>
#include "../../core/macros.hpp"
#include "../../core/arch.hpp"
#include "../../core/coord.hpp"
#include "../../core/layout.hpp"
#include "../../core/gemm_type.hpp"
#include "../../arch/cross_core_sync_aiv.hpp"
#include "../../gemm/helper/block_mmad_alignment_helper.hpp"
#include "../../helper/layout_helper.hpp"
#include "../../gemm/tile/gemm_tile_copy.hpp"
#include "../../gemm/tile/tile_mmad.hpp"

namespace FTSelf::Gemm::Block {
template<
    bool ENABLE_UNIT_FLAG_,
    class L1TileShapeforFT_,
    class L0TileShapeforFT_,
    class AType_,
    class BType_,
    class CType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopyFTABonAic_,
    class TileMmad_
>
struct BlockMmadSpecABeNoSplitKRobust<
    Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShapeforFT_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFTABonAic_,
    TileMmad_
>{
public:

    // Type Aliases
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShapeforFT = L1TileShapeforFT_;
    using L0TileShapeforFT = L0TileShapeforFT_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using TileMmad = TileMmad_;

    using CopyGmToL1A = typename TileCopyFTABonAic_::CopyGmToL1A;
    using CopyGmToL1X = typename TileCopyFTABonAic_::CopyGmToL1X;
    using CopyL1ToL0X = typename TileCopyFTABonAic_::CopyL1ToL0X;
    using CopyL1ToL0AforFT = typename TileCopyFTABonAic_::CopyL1ToL0AforFT;
    using CopyL0CToGmforABE = typename TileCopyFTABonAic_::CopyL0CToGmforABE;

    static_assert(std::is_same_v<ElementA, ElementX>,
        "The A and X element types must be identical");
    static_assert(std::is_same_v<ElementA, half> ||
                  std::is_same_v<ElementA, bfloat16_t> ||
                  std::is_same_v<ElementA, float> ||
                  std::is_same_v<ElementA, int8_t>,
        "Unsupported input type for MMAD accumulation");
    using ElementAccumulator =
        std::conditional_t<std::is_same_v<ElementA, int8_t>, int32_t, float>;

    using LayoutAInL1 = typename CopyL1ToL0AforFT::LayoutSrc;
    using LayoutXInL1 = typename CopyL1ToL0X::LayoutSrc;

    // using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;

    using LayoutAInL0forFT = typename CopyL1ToL0AforFT::LayoutDst;
    using LayoutXInL0 = typename CopyL1ToL0X::LayoutDst;

    using LayoutYInL0 = FTSelf::layout::zN;
    using LayoutCInL0 = FTSelf::layout::zN;

    using L1AAlignHelper = helper::MatrixL1AlignTraits<ElementA, LayoutA>;
    static constexpr uint32_t VECTOR_M_ALIGNED = 16;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L1A_SIZE = L1TileShapeforFT::M * L1TileShapeforFT::K * sizeof(ElementA);
    static constexpr uint32_t L1X_SIZE = L1TileShapeforFT::N * L1TileShapeforFT::K * sizeof(ElementX);
    static constexpr uint32_t L1VX_SIZE = 16 * L1TileShapeforFT::K * sizeof(ElementX);

    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    // static constexpr uint32_t L0A_SIZE_FOR_A = L0TileShape::M * L0TileShape::K * sizeof(ElementA) * STAGES;
    // static constexpr uint32_t L0A_SIZE_FOR_X = L0TileShapeforFT::N * L0TileShapeforFT::K * sizeof(ElementX) * STAGES;

    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    // static constexpr uint32_t L0B_SIZE_FOR_B = L0TileShape::K * L0TileShape::N * sizeof(ElementB) * STAGES;
    // static constexpr uint32_t L0B_SIZE_FOR_A = L0TileShape::M * L0TileShape::K * sizeof(ElementA) * STAGES;

    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    // static constexpr uint32_t L0A_PINGPONG_BUF_SIZE_FOR_A = L0A_SIZE_FOR_A / STAGES;
    // static constexpr uint32_t L0A_PINGPONG_BUF_SIZE_FOR_X = L0A_SIZE_FOR_X / STAGES;

    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    // static constexpr uint32_t L0B_PINGPONG_BUF_SIZE_FOR_B = L0B_SIZE_FOR_B / STAGES;
    // static constexpr uint32_t L0B_PINGPONG_BUF_SIZE_FOR_A = L0B_SIZE_FOR_A / STAGES;


    // Check LayoutC
    static_assert(std::is_same_v<LayoutY, FTSelf::layout::RowMajor>, "LayoutY only support RowMajor yet!");

    // Check L1TileShape: 统一通过 A1 存储传输，所以要相加小于整体size
    static_assert((L1A_SIZE * STAGES + L1X_SIZE * STAGES + L1VX_SIZE * STAGES) <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShapeforFT::M * L0TileShapeforFT::K * sizeof(ElementA);
    static constexpr uint32_t L0X_TILE_SIZE = L0TileShapeforFT::N * L0TileShapeforFT::K * sizeof(ElementX);
    static constexpr uint32_t L0VX_TILE_SIZE = VECTOR_M_ALIGNED * L0TileShapeforFT::K * sizeof(ElementX);

    // static constexpr uint32_t L0C_SIZE_for_C = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
    static constexpr uint32_t L0C_SIZE_for_ABE = L0TileShapeforFT::N * L0TileShapeforFT::M * sizeof(ElementAccumulator);
    static constexpr uint32_t L0C_SIZE_for_AE = VECTOR_M_ALIGNED * L0TileShapeforFT::M * sizeof(ElementAccumulator);

    static_assert((L0X_TILE_SIZE * STAGES + L0VX_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the space of L0A for X!");

    static_assert((L0A_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the space OF l0B for A!");

    static_assert((L0C_SIZE_for_ABE + L0C_SIZE_for_AE) <= L0C_SIZE, "L0TileShape for Ae and ABe exceeding the global L0C space");

    static_assert(L1TileShapeforFT::N == L0TileShapeforFT::N && L1TileShapeforFT::M == L0TileShapeforFT::M,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    static_assert(L0TileShapeforFT::N >= VECTOR_M_ALIGNED,
        "The situation where the L0TileShapeforFT::N < 16 is not supported yet");

    static_assert((L0TileShapeforFT::N % VECTOR_M_ALIGNED == 0),
        "The situation where the L0TileShapeforFT::N % 16 != 0 is not supported yet");

    /// Construct
    FTSELF_DEVICE
    BlockMmadSpecABeNoSplitKRobust(uint32_t l1BufAddrStart = 0)
    {
        // block 上 L1 Cache 中 A1 和 B1 的起始地址偏移
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1XOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        uint32_t l1VXOffset = l1BufAddrStart + L1A_SIZE * STAGES + L1X_SIZE * STAGES;

        uint32_t l0AOffset = 0;

        uint32_t l0BOffset = 0;

        uint32_t l0COffset = 0;
        // Init buffers
        for(uint32_t i=0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            /*
            FTSelf::Arch::LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
            */
            /*
            这里 L1 的存储共用 A1 位置，所以要注意起始地址要分隔开来，顺序访问
            */
            l1ATensorList[i] = AscendC::LocalTensor<ElementA>(
                AscendC::TPosition::A1, l1AOffset + L1A_SIZE * i, L1A_SIZE / sizeof(ElementA));
            l1XTensorList[i] = AscendC::LocalTensor<ElementX>(
                AscendC::TPosition::A1, l1XOffset + L1X_SIZE * i, L1X_SIZE / sizeof(ElementX));
            l1VXTensorList[i] = AscendC::LocalTensor<ElementX>(
                AscendC::TPosition::A1, l1VXOffset + L1VX_SIZE * i, L1VX_SIZE / sizeof(ElementX));

            /*
            L0 Cache 上面，分别使用 A2, B2 两个position，所以两个tensor的L0 Cache 均从0开始
            */
            l0ATensorListforX[i] = AscendC::LocalTensor<ElementX>(
                AscendC::TPosition::A2, l0AOffset + L0A_PINGPONG_BUF_SIZE * i,
                L0A_PINGPONG_BUF_SIZE / sizeof(ElementX));
            l0BTensorListforA[i] = AscendC::LocalTensor<ElementA>(
                AscendC::TPosition::B2, l0BOffset + L0B_PINGPONG_BUF_SIZE * i,
                L0B_PINGPONG_BUF_SIZE / sizeof(ElementA));

            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1XEventList[i] = i + STAGES;
            l1VXEventList[i] = i + STAGES * 2;

            l0BEventListforA[i] = i;
            l0AEventListforX[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[i]);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[i]);
        }

        l0CTensor = AscendC::LocalTensor<ElementAccumulator>(
            AscendC::TPosition::CO1, l0COffset, L0C_SIZE / sizeof(ElementAccumulator));

        MMAD_ABE_EVENT_ID1 = 1;
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
    }

    /// Destructor
    FTSELF_DEVICE
    ~BlockMmadSpecABeNoSplitKRobust()
    {
        for(uint32_t i=0; i < STAGES; i++){
            // 等待相关内存事件完成后再结束运行
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    FTSELF_DEVICE
    void add_ae_op(
        AscendC::GlobalTensor<ElementA> const & gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const & gmX, LayoutX const &layoutX,
        ElementX aMeanFactor,
        AscendC::GlobalTensor<ElementY> const & gmY, LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const & gmVY, LayoutY const &layoutVY,
        uint32_t actualM, uint32_t actualN, uint32_t actualK, bool writeAe = true)
    {

        uint32_t mRound = FTSelf::helper::RoundUp<L1AAlignHelper::N_ALIGNED>(actualM);
        uint32_t nRoundforFT = FTSelf::helper::RoundUp<L1AAlignHelper::M_ALIGNED>(actualN);

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShapeforFT::M, L1TileShapeforFT::K);
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1TileShapeforFT::N, L1TileShapeforFT::K);
        auto layoutVXInL1 = LayoutXInL1::template MakeLayout<ElementX>(VECTOR_M_ALIGNED, L1TileShapeforFT::K);

        auto layoutInL0CTotal = FTSelf::helper::MakeL0CLayout<LayoutCInL0>(
            nRoundforFT + VECTOR_M_ALIGNED, mRound);

        auto layoutInL0CforABe = FTSelf::helper::MakeTileLayout2D(layoutInL0CTotal, nRoundforFT, mRound);
        auto layoutInL0CforAe = FTSelf::helper::MakeTileLayout2D(layoutInL0CTotal, VECTOR_M_ALIGNED, mRound);

        uint32_t kActual = min(actualK, L1TileShapeforFT::K);

        // load first matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        // 设定Tile 在global memory 中的layout, 即将一个Tile作为一个矩阵中的一部分，
        // 将其中的元素重新组织为与layoutA相同类型的的layout，其中shape为Tile规模
        // 但是每个元素/分形 行和列之间的 stride 还是按照原来整体layout的 shape/stride 来进行组织
        // 因为这里每个block是只输入并处理一个L1 Tile
        auto layoutTileA = FTSelf::helper::MakeTileLayout2D(layoutA, actualM, kActual);
        copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        // load first vector x tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListId]);
        auto layoutTileX = FTSelf::helper::MakeTileLayout2D(layoutX, actualN, kActual);
        copyGmToL1X(l1XTensorList[l1ListId], gmX, layoutXInL1, layoutTileX);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListId]);

        // Build the constant AMean factor directly in L1. The full zN buffer is
        // initialized so every padded lane has a deterministic value.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[l1VXListId]);
        AscendC::Fill(l1VXTensorList[l1VXListId],
            {1, static_cast<uint16_t>(L1VX_SIZE / FTSelf::BYTE_PER_C0), 0, aMeanFactor});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1VXEventList[l1VXListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
        }

        uint32_t mPartLoop = FTSelf::helper::CeilDiv<L0TileShapeforFT::M>(mRound);
        uint32_t nPartLoop = FTSelf::helper::CeilDiv<L0TileShapeforFT::N>(nRoundforFT);

        // main loop
        uint kTileCount = FTSelf::helper::CeilDiv<L1TileShapeforFT::K>(actualK);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1VXEventList[l1VXListId]);

        constexpr uint32_t ELEX_PER_C0 = BYTE_PER_C0 / sizeof(ElementX);

        for(uint32_t kLoopIdx=0; kLoopIdx < kTileCount; kLoopIdx++)
        {
            // 下一阶段执行的stage id
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            // 流水线，提前将下一阶段的数据从 GM 加载到 L1 中与计算overlap
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1){
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                // 下一阶段 若非最后一个 loop，那么执行一个L1TileShape::K,否则执行剩余的数据
                kActualNext = (kLoopIdxNext < kTileCount - 1) ?
                    L1TileShapeforFT::K : (actualK - kLoopIdxNext * L1TileShapeforFT::K);

                // Get L1 Tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1XTensor = l1XTensorList[l1ListIdNext];
                auto l1VXTensor = l1VXTensorList[l1ListIdNext];

                // Get GM tile for next stage
                uint32_t gmTileAOffset = FTSelf::helper::GetOffset2D(
                    layoutA, 0, kLoopIdxNext * L1TileShapeforFT::K);
                uint32_t gmTileXOffset = FTSelf::helper::GetOffset2D(
                    layoutX, 0, kLoopIdxNext * L1TileShapeforFT::K);
                uint32_t gmTileVXOffset{kLoopIdxNext * L1TileShapeforFT::K};

                auto gmTileA = gmA[gmTileAOffset];
                auto gmTileX = gmX[gmTileXOffset];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                layoutTileA = FTSelf::helper::MakeTileLayout2D(layoutA, actualM, kActualNext);
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);


                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListIdNext]);
                layoutTileX = FTSelf::helper::MakeTileLayout2D(layoutX, actualN, kActualNext);
                copyGmToL1X(l1XTensor, gmTileX, layoutXInL1, layoutTileX);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListIdNext]);
            }

            // Get L1 Tensor for current usage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1XTensor = l1XTensorList[l1ListId];
            auto l1VXTensor = l1VXTensorList[l1VXListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = FTSelf::helper::CeilDiv<L0TileShapeforFT::K>(kActual);

            for(int nPartIdx=0; nPartIdx < nPartLoop; nPartIdx++){
                uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                        L0TileShapeforFT::N : (nRoundforFT - nPartIdx * L0TileShapeforFT::N);

                uint32_t mPartActualforX = nRoundforFT;

                for(int kPartIdx=0; kPartIdx < kPartLoop; kPartIdx++){
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShapeforFT::K : (kActual - kPartIdx * L0TileShapeforFT::K);
                    uint32_t kPartAligned = (kPartActual + ELEX_PER_C0 - 1) / ELEX_PER_C0 * ELEX_PER_C0;
                    // Locate the current tile on L0B
                    auto l0ATensorforX = l0ATensorListforX[l0AListIdforX];

                    LayoutXInL0 layoutXInL0Total =
                        LayoutXInL0::template MakeLayout<ElementX>((nPartActual + VECTOR_M_ALIGNED), kPartActual);

                    LayoutXInL0 layoutXInL0forABe = LayoutXInL0::template MakeLayout<ElementX>(nPartActual, kPartActual);
                    LayoutXInL0 layoutXInL0forAe = LayoutXInL0::template MakeLayout<ElementX>(VECTOR_M_ALIGNED, kPartActual);

                    auto l0XTileforABe = l0ATensorforX[0];

                    auto l0XTileforAe = l0ATensorforX[uint32_t(nPartActual) * uint32_t(kPartAligned)];

                    // Locate the current tile of matrix X for ABe on L1
                    uint32_t l1XOffsetforABe = FTSelf::helper::GetOffset2D(
                        layoutXInL1, 0, kPartIdx * L0TileShapeforFT::K);
                    auto l1XTile = l1XTensor[l1XOffsetforABe];

                    uint32_t l1VXOffsetforAe = FTSelf::helper::GetOffset2D(
                        layoutVXInL1, 0, kPartIdx * L0TileShapeforFT::K);
                    auto l1VXTile = l1VXTensor[l1VXOffsetforAe];

                    // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                    if((nPartIdx == 0) && (kPartIdx == 0)){
                        // 若为当前stage第一次迭代，需要等待到第一批数据，
                        // 即在当前stage涉及的迭代前已经preload 完的B数据成功
                        // preload 到 L1 上才可进行B数据向 L0 上写
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1ListId]);
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventListforX[l0AListIdforX]);
                    copyL1ToL0X(l0XTileforABe, l1XTile, layoutXInL0forABe, layoutXInL1);
                    copyL1ToL0X(l0XTileforAe, l1VXTile, layoutXInL0forAe, layoutVXInL1);

                    // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                    ReleaseL1XIfLast(kPartIdx, kPartLoop, nPartIdx, nPartLoop);

                    for(int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++){
                        uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                            L0TileShapeforFT::M : (mRound - mPartIdx * L0TileShapeforFT::M);

                        // Locate the current tile on L0A
                        auto l0BTileforA = l0BTensorListforA[l0BListIdforA];

                        LayoutAInL0forFT layoutAInL0forFT = LayoutAInL0forFT::template MakeLayout<ElementA>(mPartActual,kPartActual);

                        // Locate the current tile of matrix A on L1
                        uint32_t l1AOffset = FTSelf::helper::GetOffset2D(
                            layoutAInL1, mPartIdx * L0TileShapeforFT::M,
                            kPartIdx * L0TileShapeforFT::K);
                        auto l1ATile = l1ATensor[l1AOffset];

                        if((mPartIdx == 0) && (kPartIdx == 0)){
                            // 若为当前stage第一次迭代，需要等待到第一批数据，
                            // 即在当前stage涉及的迭代前已经preload 完的A数据成功preload 到 L1 上才可进行A数据向 L0 上写
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                        }

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventListforA[l0BListIdforA]);
                        copyL1ToL0AforFT(l0BTileforA, l1ATile, layoutAInL0forFT, layoutAInL1);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MMAD_ABE_EVENT_ID1);

                        ReleaseL1AIfLast(mPartIdx, mPartLoop, kPartIdx, kPartLoop);
                        uint32_t l0COffsetTotal = FTSelf::helper::GetOffset2D(
                            layoutInL0CTotal, nPartIdx * L0TileShapeforFT::N,
                            mPartIdx * L0TileShapeforFT::M);
                        auto l0CTile = l0CTensor[l0COffsetTotal];

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MMAD_ABE_EVENT_ID1);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        // 当前 M,N Tile 的第一个 K 时，需要初始化输出的C矩阵为0

                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        uint8_t unitFlag = helper::GetPingpongMmadUnitFlag<ENABLE_UNIT_FLAG>(
                            kLoopIdx, kTileCount, mPartIdx, mPartLoop,
                            kPartIdx, kPartLoop, nPartIdx, nPartLoop);
                        // Perform calculation operations
                        tileMmadforABe(l0CTile, l0ATensorforX, l0BTileforA,
                            (nPartActual + VECTOR_M_ALIGNED), mPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        // 标记计算完成，即当前已经完成了一个l0 tile 的运算，可以加载下一个l0 tile了
                        // 这里最内层为 B 矩阵
                        AdvanceL0Stage(l0BEventListforA, l0BListIdforA);
                    }
                    // 交替进行阶段，实现 L1 与 L0 之间数据传输与 MMAD 计算的PING-PANG
                    AdvanceL0Stage(l0AEventListforX, l0AListIdforX);
                }
            }
            // 交替进行阶段，实现L1 与 Global 之间数据传输 与 MMAD计算 的PING-PANG
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1VXEventList[l1VXListId]);
        l1VXListId = (l1VXListId + 1 < STAGES)? (l1VXListId + 1) : 0;

        // copy block out
        // 将最终结果从 L0 的 CO1 输出到GM即可
        LayoutY layoutBlockforABe = FTSelf::helper::MakeTileLayout2D(layoutY, actualN, actualM);
        LayoutY layoutBlockforAe = FTSelf::helper::MakeTileLayout2D(layoutVY, 1, actualM);

        if constexpr (!ENABLE_UNIT_FLAG) {
            // 标记开始写入cGM 数据
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(MMAD_ABE_EVENT_ID1);
            // 等待允许写入开始
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(MMAD_ABE_EVENT_ID1);

            uint32_t l0COffsetforABe = FTSelf::helper::GetOffset2D(layoutInL0CTotal, 0, 0);
            auto l0CTileforABe = l0CTensor[l0COffsetforABe];

            uint32_t l0COffsetforAe = FTSelf::helper::GetOffset2D(layoutInL0CTotal, nRoundforFT, 0);
            // 获取当前局部输出
            auto l0CTileforAe = l0CTensor[l0COffsetforAe];
            copyL0CToGmforABE(gmY, l0CTileforABe, layoutBlockforABe, layoutInL0CforABe);
            if (writeAe) {
                copyL0CToGmforABE(gmVY, l0CTileforAe, layoutBlockforAe, layoutInL0CforAe);
            }
            // 标记数据写入 cGM 已经完成
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(MMAD_ABE_EVENT_ID1);
        } else {
            uint32_t l0COffsetforABe = FTSelf::helper::GetOffset2D(layoutInL0CTotal, 0, 0);
            auto l0CTileforABe = l0CTensor[l0COffsetforABe];

            uint32_t l0COffsetforAe = FTSelf::helper::GetOffset2D(layoutInL0CTotal, nRoundforFT, 0);
            // 获取当前局部输出
            auto l0CTileforAe = l0CTensor[l0COffsetforAe];

            copyL0CToGmforABE(gmY, l0CTileforABe, layoutBlockforABe, layoutInL0CforABe, 0b11);
            if (writeAe) {
                copyL0CToGmforABE(gmVY, l0CTileforAe, layoutBlockforAe, layoutInL0CforAe, 0b11);
            }
        }
    }

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
        uint32_t actualM, uint32_t actualN, uint32_t actualK)
    {
        add_ae_op(gmA, layoutA, gmX, layoutX, ElementX(0), gmY, layoutY,
            gmY, layoutY, actualM, actualN, actualK, false);
    }

private:
    FTSELF_DEVICE void ReleaseL1XIfLast(
        uint32_t kPartIdx, uint32_t kPartLoop, uint32_t nPartIdx, uint32_t nPartLoop)
    {
        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1ListId]);
        }
    }

    FTSELF_DEVICE void ReleaseL1AIfLast(
        uint32_t mPartIdx, uint32_t mPartLoop, uint32_t kPartIdx, uint32_t kPartLoop)
    {
        if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        }
    }

    FTSELF_DEVICE void AdvanceL0Stage(int32_t *eventList, uint32_t &listId)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventList[listId]);
        listId = (listId + 1 < STAGES) ? (listId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementX> l1XTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l1VXTensorList[STAGES];

    AscendC::LocalTensor<ElementX> l0ATensorListforX[STAGES];
    AscendC::LocalTensor<ElementA> l0BTensorListforA[STAGES];

    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1XEventList[STAGES];
    int32_t l1VXEventList[STAGES];

    int32_t l0AEventListforX[STAGES];
    int32_t l0BEventListforA[STAGES];

    // The id of current stage
    // 指示当前所处的pipeline 中的阶段（双阶段PING-PONG）
    uint32_t l1ListId{0};
    uint32_t l1VXListId{0};

    uint32_t l0AListIdforX{0};
    uint32_t l0BListIdforA{0};

    int32_t MMAD_ABE_EVENT_ID1;
    int32_t MMAD_ABE_EVENT_ID2;

    TileMmad tileMmadforABe;
    CopyGmToL1A copyGmToL1A;

    CopyGmToL1X copyGmToL1X;
    CopyL1ToL0X copyL1ToL0X;
    CopyL1ToL0AforFT copyL1ToL0AforFT;
    CopyL0CToGmforABE copyL0CToGmforABE;
};
} // namespace FTSelf::Gemm::Block

#endif // FTSELF_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_FAULT_ABE_SPEC_NO_SPLITK_ROBUST_HPP
