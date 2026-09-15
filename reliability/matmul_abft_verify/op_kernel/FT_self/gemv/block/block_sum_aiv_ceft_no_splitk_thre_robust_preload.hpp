/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_GEMV_ASVAR_CE_AIV_NO_SPLIT_HPP_SPEC_ROBUST_PRELOAD
#define FTSELF_GEMV_BLOCK_BLOCK_GEMV_ASVAR_CE_AIV_NO_SPLIT_HPP_SPEC_ROBUST_PRELOAD

#include "../../arch/resource_aiv.hpp"
#include "../../gemv/helper.hpp"
#include "../../gemv/tile/tile_vmuls.hpp"

namespace FTSelf::Gemv::Block {



template <
    class UBTileShape_,
    class UBBlockShape_,
    class L1TileShape_,
    class AType_,
    class XType_,
    class YType_,
    class ZType_,
    class BiasType_,
    class TileCopy_,
    class TileFaultSum_,
    class TileThreCalc_,
    class TileStdEst_>
struct BlockFTGemvCENoSplitKPreload <
    FTSelf::Gemv::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
    FTSelf::Gemv::helper::FT_ENC_TYPE::RCE,
    FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
    FTSelf::Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK,
    UBTileShape_,
    UBBlockShape_,
    L1TileShape_,
    AType_,
    XType_,
    YType_,
    ZType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_,
    TileThreCalc_,
    TileStdEst_>
{
public:
    // Type Aliases
    using DispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FT_ENC_TYPE = FTSelf::Gemv::helper::FT_ENC_TYPE;
    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;
    using FT_REDUCE_TYPE = FTSelf::Gemv::helper::FT_REDUCE_TYPE;
    using FT_AIV_PIPE_FUSE_TYPE = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE;
    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using FT_ABE_TYPE = FTSelf::Gemv::helper::FT_ABE_TYPE;

    using UBTileShape = UBTileShape_;
    using UBBlockShape = UBBlockShape_;
    using L1TileShape = L1TileShape_;

    using ThreCalcUBTileShape = GemvShape<UBTileShape::M,L1TileShape::N>;
    using ThreCalcUBTileShapeTotal = GemvShape<L1TileShape::M,L1TileShape::N>;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;

    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;


    using TileThreCalc = TileThreCalc_;
    using TileStdEst = TileStdEst_;
    using TileFaultSum = TileFaultSum_;

    using TileFaultSumCSum = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, AType_, YType_>;

    using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
    using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
    using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
    using MatrixCopyGmToUbforThre = typename TileCopy_::MatrixCopyGmToUbforThre;
    using VecCopyGmToUbInY = typename TileCopy_::VecCopyGmToUbInY;
    using VecCopyUbToGmZ = typename TileCopy_::VecCopyUbToGmZ;
    using VecCopyUbToGmforThre = typename TileCopy_::VecCopyUbToGmforThre;

    using TileCompare = FTSelf::Gemv::Tile::TileFaultVcompare<FT_COMP_TYPE::RSUB, ArchTag,
                                        ZType_, YType_, YType_>;

    static constexpr FT_COMP_TYPE COMP_TYPE = FT_COMP_TYPE::RSUB;
    static constexpr FT_ENC_TYPE ENC_TYPE = FT_ENC_TYPE::RCE;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR;
    static constexpr FT_ABE_TYPE ABE_TYPE = FT_ABE_TYPE::CENTRAL_BLOCK;

    static constexpr bool NEED_CAST_FOR_RED = std::is_same<ElementX, ElementY>::value;

    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using UBAlignHelper = FTSelf::Gemv::helper::UBAlignHelper<ElementA>;
    using TensorCoord = FTSelf::layout::VectorLayout::TensorCoord;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::NO_FUSED;
    static constexpr uint32_t Cbuf_SIZE_ = 128 * 1024;
    static constexpr uint32_t Meanbuf_SIZE_ = 4 * 1024;
    static constexpr uint32_t Maxbuf_SIZE_forY_ = 4 * 1024;
    static constexpr uint32_t Minbuf_SIZE_forY_ = 4 * 1024;

    static constexpr uint32_t workspace_SIZE_ = 32 * 1024;
    static constexpr uint32_t Threbuf_SIZE_ = 4 * 1024;
    static constexpr uint32_t thre_workspace_SIZE_ = 6 * 1024;

    static constexpr uint32_t Ybuf_SIZE_ = 4 * 1024;
    static constexpr uint32_t ABebuf_SIZE_ = 4 * 1024;
    static constexpr uint32_t Zbuf_SIZE_ = 2 *1024;

    static constexpr uint32_t ELE_NUM_PER_BLK_FOR_C = FTSelf::Gemv::BYTE_PER_BLK / sizeof(ElementY);

    static_assert(L1TileShape::M == UBBlockShape::M,
        "The situation where the basic Tile of UB and L1 for MMA differ on the m axes is not supported yet");

    static_assert(L1TileShape::N == UBBlockShape::N,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");


    static_assert((UBBlockShape::N % UBTileShape::N) == 0,
        "The situation where the basic Tile of UB and L1 for MMA differ on the n axes is not supported yet");

    static_assert(UBBlockShape::M / UBTileShape::M <= 2,
        "The situation where the basic Tile of UB In Total AICores and L1 for MMA differ on the m axes is not supported yet");

    FTSELF_DEVICE
    BlockFTGemvCENoSplitKPreload() {}

    /// Construct
    FTSELF_DEVICE
    BlockFTGemvCENoSplitKPreload(FTSelf::Arch::Resource<FTSelf::Arch::AtlasA2> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

    /// Construct
    FTSELF_DEVICE
    BlockFTGemvCENoSplitKPreload(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0)
    {
        InitializeBuffers(resource, UBufAddrStart);
    }

private:
    template <class Resource>
    FTSELF_DEVICE
    void InitializeBuffers(Resource &resource, uint32_t UBufAddrStart)
    {
        uint32_t UbCOffset = UBufAddrStart;
        uint32_t UbYOffset = UBufAddrStart + Cbuf_SIZE_;
        uint32_t UbABeOffset = UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_;

        uint32_t UbAMeanOffset = UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_ + ABebuf_SIZE_;
        uint32_t UbAMinOffset_forY =  UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_ + ABebuf_SIZE_ + Meanbuf_SIZE_;
        uint32_t UbAMaxOffset_forY =  UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_ + ABebuf_SIZE_ + Meanbuf_SIZE_ + Minbuf_SIZE_forY_;

        uint32_t UbThreOffset = UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_ + ABebuf_SIZE_ + Meanbuf_SIZE_ + Minbuf_SIZE_forY_+ Maxbuf_SIZE_forY_;

        uint32_t UbZOffset = UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_ + ABebuf_SIZE_ + Meanbuf_SIZE_ + Minbuf_SIZE_forY_ + Maxbuf_SIZE_forY_ + Threbuf_SIZE_;

        uint32_t UbWOffset = UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_ + ABebuf_SIZE_ + Meanbuf_SIZE_ + Minbuf_SIZE_forY_ + Maxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_;
        uint32_t UbWOffset_thre = UBufAddrStart + Cbuf_SIZE_ + Ybuf_SIZE_ + ABebuf_SIZE_ + Meanbuf_SIZE_ + Minbuf_SIZE_forY_ + Maxbuf_SIZE_forY_ + Threbuf_SIZE_ + Zbuf_SIZE_ + workspace_SIZE_;

        // Init buffers
        UbInBRedEvent = 0;

        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            UbCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbCOffset + i * (Cbuf_SIZE_ / 2));
            UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset + i * (Ybuf_SIZE_ / 2));
            UbABeTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbABeOffset + i * (ABebuf_SIZE_ / 2));

            UbWTensorList[i] =
                resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset + i * (workspace_SIZE_ / 2));

            UbWforThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbWOffset_thre + i * (thre_workspace_SIZE_ / 2));

            UbThreTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbThreOffset + i * (Threbuf_SIZE_ / 2));

            UbAMeanTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMeanOffset + i * (Meanbuf_SIZE_ / 2));
            UbAMinTensorforYList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMinOffset_forY + i * (Minbuf_SIZE_forY_ / 2));
            UbAMaxTensorforYList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbAMaxOffset_forY + i * (Maxbuf_SIZE_forY_ / 2));

            UbZTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementZ>(UbZOffset + i * (Zbuf_SIZE_ / 2));

            // Assign event ID for each stages
            UbInCEventList[i] = i;
            UbInARedEventList[i] = i + STAGES;
            UbInAMaxEventList[i] = i + STAGES * 2;

            UbOutEventList[i] = i;
            UbOutZEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInCEventList[i]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInARedEventList[i]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_S>(UbInBRedEvent);
    }

    FTSELF_DEVICE
    void PrepareNextStatistics(uint32_t stage, uint32_t count, uint32_t offset,
        AscendC::GlobalTensor<ElementY> const &gmMin,
        AscendC::GlobalTensor<ElementY> const &gmMean,
        AscendC::GlobalTensor<ElementY> const &gmBe,
        AscendC::GlobalTensor<ElementY> const &gmMax)
    {
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutZEventList[stage]);
        FTSelf::Gemv::helper::ZeroTensor(UbYTensorList[stage], count);
        FTSelf::Gemv::helper::ZeroTensor(UbThreTensorList[stage], count);
        FTSelf::Gemv::helper::VectorBarrier();
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_V>(UbOutZEventList[stage]);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInARedEventList[stage]);

        vecCopyGmToUbInY(UbAMinTensorforYList[stage], gmMin[offset], count);
        vecCopyGmToUbInY(UbAMeanTensorList[stage], gmMean[offset], count);
        vecCopyGmToUbInY(UbABeTensorList[stage], gmBe[offset], count);
        vecCopyGmToUbInY(UbAMaxTensorforYList[stage], gmMax[offset], count);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInARedEventList[stage]);
    }

public:
    /// Destructor
    FTSELF_DEVICE
    ~BlockFTGemvCENoSplitKPreload()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInCEventList[i]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInARedEventList[i]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutZEventList[i]);
        }
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_S>(UbInBRedEvent);
    }

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmC, LayoutA const &layoutC,
        AscendC::GlobalTensor<ElementY> const &gmAMax,
        AscendC::GlobalTensor<ElementY> const &gmNextAMax,
        AscendC::GlobalTensor<ElementY> const &gmAMean,
        AscendC::GlobalTensor<ElementY> const &gmNextAMean,
        AscendC::GlobalTensor<ElementY> const &gmAMin,
        AscendC::GlobalTensor<ElementY> const &gmNextAMin,
        LayoutX const &layoutAforFT,
        AscendC::GlobalTensor<ElementY> const &gmY,
        AscendC::GlobalTensor<ElementY> const &gmABe,
        AscendC::GlobalTensor<ElementY> const &gmNextABe,
        LayoutY const &layoutY,
        AscendC::GlobalTensor<ElementY> const &gmBMeanAbs,
        AscendC::GlobalTensor<ElementY> const &gmBMeanSquare,
        AscendC::GlobalTensor<ElementY> const &gmBVar,LayoutX const &layoutBforFT,
        AscendC::GlobalTensor<ElementY> const &gmThreZ, LayoutY const &layoutThre,
        AscendC::GlobalTensor<ElementZ> const &gmCOMPZ, LayoutZ const &layoutZ,
        GemvCoord const &actualShape,
        GemvCoord const &actualShapeNext,
        float n_ratio_factor,
        float n_sqrt_ratio_factor,
        float n_square_ratio_factor,
        float e_max,
        bool outputThre, bool outputCE,
        uint32_t aiv_part_num, bool isFirstBlock, bool hasNextBlock)
    {
        aiv_part_num = aiv_part_num == 0 ? 1 : aiv_part_num;
        // , AscendC::GlobalTensor<ElementY> const &gmABeOut
        TileMRound = FTSelf::helper::RoundUp(UBTileShape::M, UBAlignHelper::ALIGN);
        TileNRound = FTSelf::helper::RoundUp(UBTileShape::N, UBAlignHelper::ALIGN);

        ThreTileMRound = FTSelf::helper::RoundUp(ThreCalcUBTileShape::M, UBAlignHelper::ALIGN);
        ThreTileNRound = FTSelf::helper::RoundUp(ThreCalcUBTileShape::N, UBAlignHelper::ALIGN);

        BlockMRound = FTSelf::helper::RoundUp(UBBlockShape::M, UBAlignHelper::ALIGN);
        BlockNRound = FTSelf::helper::RoundUp(UBBlockShape::N, UBAlignHelper::ALIGN);

        ThreBlockMRound = FTSelf::helper::RoundUp(ThreCalcUBTileShapeTotal::M, UBAlignHelper::ALIGN);
        ThreBlockNRound = FTSelf::helper::RoundUp(ThreCalcUBTileShapeTotal::N, UBAlignHelper::ALIGN);

        strideCCol = layoutC.stride(1) * TileNRound;
        strideCRow = layoutC.stride(0) * TileMRound;

        m_actual_total = (actualShape.m() < BlockMRound) ? actualShape.m() : BlockMRound;
        n_actual_total = (actualShape.n() < BlockNRound) ? actualShape.n() : BlockNRound;

        m_actual_total_next = (actualShapeNext.m() < BlockMRound) ? actualShapeNext.m() : BlockMRound;
        n_actual_total_next = (actualShapeNext.n() < BlockNRound) ? actualShapeNext.n() : BlockNRound;

        m_actual_part = (m_actual_total + aiv_part_num - 1)/ aiv_part_num;
        m_actual_part_next = (m_actual_total_next + aiv_part_num - 1)/ aiv_part_num;

        // 需要对8对齐，不然输出的COMP会在元素上被切开(8行的比较结果存在一个Byte上)
        uint32_t m_aligned_part = (m_actual_part + 8 - 1) / 8 * 8;
        uint32_t m_aligned_part_next = (m_actual_part_next + 8 - 1) / 8 * 8;


        uint32_t M_start_offset = AscendC::GetSubBlockIdx() * m_aligned_part;
        uint32_t M_start_offset_next = AscendC::GetSubBlockIdx() * m_aligned_part_next;

        uint32_t Z_start_offset = (M_start_offset + 8 - 1) / 8;

        // 在对齐之后，有的AIV会直接没有任何任务，因此需要将任务量置零
        m_actual_part = (M_start_offset >= m_actual_total) ? 0 : min(m_aligned_part, m_actual_total - M_start_offset);
        m_actual_part_next = (M_start_offset_next >= m_actual_total_next) ? 0 : min(m_aligned_part_next, m_actual_total_next - M_start_offset_next);


        uint32_t Mloop = FTSelf::helper::CeilDiv(m_actual_part, TileMRound);

        uint32_t NloopV = FTSelf::helper::CeilDiv(n_actual_total, TileNRound);

        dst_offset_ratio = Mloop;
        out_z_actual_total = (m_actual_part + 8 - 1) / 8;

        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_S>(UbInBRedEvent);
        B_slice_meanabs = gmBMeanAbs.GetValue(0);
        B_slice_meansquare = gmBMeanSquare.GetValue(0);
        B_slice_var = gmBVar.GetValue(0);

        B_slice_var_square = B_slice_var * B_slice_var;

        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::S_V>(UbInBRedEvent);
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::S_V>(UbInBRedEvent);

        m_actual = (m_actual_part < TileMRound) ? m_actual_part : TileMRound;
        m_actual_next_A = (m_actual_part_next < TileMRound) ? m_actual_part_next : TileMRound;
        n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;
        // uint32_t mLoopOffset =M_start_offset;






        // }

        if(isFirstBlock && m_actual_part > 0){
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutZEventList[UbOutListId]);
            auto UbYTensor = UbYTensorList[UbOutListId];
            auto UbThreTensor = UbThreTensorList[UbOutListId];

            FTSelf::Gemv::helper::ZeroTensor(UbYTensor, m_actual);
            FTSelf::Gemv::helper::ZeroTensor(UbThreTensor, m_actual);

            FTSelf::Gemv::helper::VectorBarrier();
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_V>(UbOutZEventList[UbOutListId]);

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInARedEventList[UbOutListId]);
            auto UbAMinTensor = UbAMinTensorforYList[UbOutListId];
            vecCopyGmToUbInY(UbAMinTensor, gmAMin[M_start_offset], m_actual);

            auto UbAMeanTensor = UbAMeanTensorList[UbOutListId];
            vecCopyGmToUbInY(UbAMeanTensor, gmAMean[M_start_offset], m_actual);

            auto UbAMaxTensor = UbAMaxTensorforYList[UbOutListId];
            vecCopyGmToUbInY(UbAMaxTensor, gmAMax[M_start_offset], m_actual);

            auto UbABeTensor = UbABeTensorList[UbOutListId];
            vecCopyGmToUbInY(UbABeTensor, gmABe[M_start_offset], m_actual);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInARedEventList[UbOutListId]);
        }


        for(uint32_t mLoopIdx = 0; mLoopIdx < Mloop; mLoopIdx++){

            m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
            n_actual = (n_actual_total < TileNRound) ? n_actual_total : TileNRound;

            out_z_actual = (m_actual + 8 - 1) / 8;

            uint32_t mLoopOffset = mLoopIdx * TileMRound + M_start_offset;
            uint32_t mLoopOffset_for_z = (mLoopOffset + 8 - 1) / 8;

            uint32_t C_row_offset = mLoopOffset;
            uint32_t C_col_offset = 0;
            uint32_t C_block_offset = C_row_offset * layoutC.stride(0) + C_col_offset * layoutC.stride(1);


            if (mLoopIdx < (Mloop -1)) {
                uint32_t UbOutListIdNext = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
                uint32_t mLoopIdxNext = mLoopIdx + 1;
                // uint32_t nLoopIdxNext = 0;
                uint32_t m_actual_next = (mLoopIdxNext < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdxNext * TileMRound;
                uint32_t mLoopOffsetNext = mLoopIdxNext * TileMRound + M_start_offset;

                PrepareNextStatistics(UbOutListIdNext, m_actual_next, mLoopOffsetNext,
                    gmAMin, gmAMean, gmABe, gmAMax);
            }else if(mLoopIdx == (Mloop -1) && hasNextBlock && m_actual_part_next > 0){
                uint32_t UbOutListIdNext = ((UbOutListId + 1) < STAGES) ? (UbOutListId + 1) : 0;
                uint32_t mLoopIdxNext = 0;
                m_actual_next_A = (m_actual_part_next < TileMRound) ? m_actual_part_next : TileMRound;
                uint32_t m_actual_next = m_actual_next_A;
                uint32_t mLoopOffsetNext = mLoopIdxNext * TileMRound + M_start_offset_next;
                PrepareNextStatistics(UbOutListIdNext, m_actual_next, mLoopOffsetNext,
                    gmNextAMin, gmNextAMean, gmNextABe, gmNextAMax);
            }


            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInARedEventList[UbOutListId]);

            auto UbAMaxTensor = UbAMaxTensorforYList[UbOutListId];
            auto UbAMeanTensor = UbAMeanTensorList[UbOutListId];
            auto UbAMinTensor = UbAMinTensorforYList[UbOutListId];

            auto UbThreTensor = UbThreTensorList[UbOutListId];
            auto UbABeTensor = UbABeTensorList[UbOutListId];

            auto layoutStdInUb = layoutThre.GetTileLayout(MakeCoord(m_actual));
            auto layoutTileStd = layoutThre.GetTileLayout(MakeCoord(m_actual));

            tileStdEst(UbAMaxTensor, UbAMeanTensor, UbAMaxTensor, UbAMinTensor,
                layoutStdInUb, layoutTileStd);




            FTSelf::Gemv::helper::VectorBarrier();

             /*
                计算阈值即可
            */

            auto layoutThreInUb = layoutThre.GetTileLayout(MakeCoord(m_actual));
            auto layoutTileThre = layoutThre.GetTileLayout(MakeCoord(m_actual));



            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_V>(UbOutZEventList[UbOutListId]);

            tileThreCalc(
                UbThreTensor, UbAMeanTensor, UbAMaxTensor, UbWforThreTensorList[UbOutListId],
                layoutThreInUb, layoutTileThre,
                (ElementY)n_ratio_factor, (ElementY)n_sqrt_ratio_factor, (ElementY)n_square_ratio_factor,
                (ElementY)B_slice_meanabs, (ElementY)B_slice_meansquare, (ElementY)B_slice_var,
                (ElementY)B_slice_var_square, (ElementY)e_max);


            // UbAMeanTensor,
            //     UbThreTensor, UbAMaxTensor, UbWforThreTensorList[UbOutListId],
            //     layoutThreInUb, layoutTileThre,
            //     (ElementY)n_sqrt_ratio_factor, (ElementY)B_slice_var, (ElementY)e_max);
            FTSelf::Gemv::helper::VectorBarrier();

            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInCEventList[UbInListId]);
            auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileC = layoutC.GetTileLayout(MakeCoord(m_actual, n_actual));
            matrixCopyGmToUb(UbCTensorList[UbInListId], gmC[C_block_offset], layoutCInUb, layoutTileC);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInCEventList[UbInListId]);

            auto UbYTensor = UbYTensorList[UbOutListId];

            // main loop
            for (uint32_t nLoopIdx = 0; nLoopIdx < NloopV; nLoopIdx++) {
                m_actual = (mLoopIdx < (Mloop - 1)) ? TileMRound : m_actual_part - mLoopIdx * TileMRound;
                n_actual = (nLoopIdx == NloopV - 1) ? (n_actual_total - nLoopIdx * TileNRound) : TileNRound;
                y_actual = m_actual;
                x_actual = n_actual;

                uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;

                if (nLoopIdx < NloopV - 1) {
                    uint32_t nLoopIdxNext = nLoopIdx + 1;
                    uint32_t m_actual_next = m_actual;
                    uint32_t n_actual_next =
                        (nLoopIdxNext == NloopV - 1) ? (n_actual_total - nLoopIdxNext * TileNRound) : TileNRound;
                    uint32_t y_actual_next = m_actual_next;
                    uint32_t x_actual_next = n_actual_next;
                    // Get L1 tensor for next stage
                    auto matrixTensor = UbCTensorList[UbInListIdNext];

                    FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE2>(UbInCEventList[UbInListIdNext]);
                    auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                    auto layoutTileC = layoutC.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                    matrixCopyGmToUb(matrixTensor, gmC[C_block_offset + nLoopIdxNext * strideCCol], layoutCInUb, layoutTileC);
                    FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE2_V>(UbInCEventList[UbInListIdNext]);
                }

                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE2_V>(UbInCEventList[UbInListId]);
                auto layoutComputeInUb = layoutC.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileCompute = layoutC.GetTileLayout(MakeCoord(m_actual, n_actual));

                tileFaultSumCSum(UbYTensor,
                    UbCTensorList[UbInListId],
                    UbWTensorList[UbInListId],
                    layoutComputeInUb,
                    layoutTileCompute);

                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInCEventList[UbInListId]);
                UbInListId = UbInListIdNext;
            }

            FTSelf::Gemv::helper::VectorBarrier();

            if(outputCE){
                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(UbOutEventList[UbOutListId]);

                auto layoutDstY = layoutY.GetTileLayout(TensorCoord(m_actual));
                auto layoutOutInUb = layoutY.GetTileLayout(TensorCoord(m_actual));

                vecCopyUbToGm(gmY[mLoopOffset], UbYTensorList[UbOutListId], layoutDstY, layoutOutInUb);
                // AscendC::PipeBarrier<PIPE_ALL>();

                FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);
                FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::MTE3_V>(UbOutEventList[UbOutListId]);
            }

            auto layoutCompareInUb = layoutThre.GetTileLayout(MakeCoord(ThreTileMRound));
            auto layoutTileCompare = layoutThre.GetTileLayout(MakeCoord(m_actual));

            // UbABeTensorList[UbOutListId]

            // UbABeTensorList[UbOutListId],
            tileCompare(
                UbZTensorList[UbOutListId],
                UbYTensorList[UbOutListId],
                UbABeTensorList[UbOutListId],
                UbThreTensorList[UbOutListId],
                layoutCompareInUb,
                layoutTileCompare,
                (ElementY)0.002f
            );

            FTSelf::Gemv::helper::VectorBarrier();

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE3>(UbOutZEventList[UbOutListId]);
            FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_MTE3>(UbOutZEventList[UbOutListId]);

            if(outputThre){
                auto layoutDstYThre = layoutThre.GetTileLayout(TensorCoord(m_actual));
                auto layoutComputeThreInUb = layoutThre.GetTileLayout(TensorCoord(m_actual));
                vecCopyUbToGmforThre(gmThreZ[mLoopOffset],
                    UbThreTensorList[UbOutListId],
                    layoutDstYThre,
                    layoutComputeThreInUb);
            }

            auto layoutDstZ = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            auto layoutComputeZInUb = layoutZ.GetTileLayout(TensorCoord(out_z_actual));
            vecCopyUbToGmZ(gmCOMPZ[mLoopOffset_for_z], UbZTensorList[UbOutListId], layoutDstZ, layoutComputeZInUb);

            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::MTE3_V>(UbOutZEventList[UbOutListId]);
            FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_MTE2>(UbInARedEventList[UbOutListId]);

            UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;

        }
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_S>(UbInBRedEvent);
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> UbCTensorList[STAGES]; // A为矩阵C

    AscendC::LocalTensor<ElementY> UbYTensorList[STAGES]; // Y为CE向量

    AscendC::LocalTensor<ElementA> UbWTensorList[STAGES]; // W为计算CE的工作空间
    AscendC::LocalTensor<ElementY> UbWforThreTensorList[STAGES]; // WforThre为计算阈值向量的工作空间

    AscendC::LocalTensor<ElementY> UbThreTensorList[STAGES]; // Thre为阈值向量
    AscendC::LocalTensor<ElementY> UbABeTensorList[STAGES]; // ABe的运算结果
    AscendC::LocalTensor<ElementY> UbAMeanTensorList[STAGES]; // 存储 A 的行 mean 向量
    AscendC::LocalTensor<ElementY> UbAMinTensorforYList[STAGES]; //  存储 A 的行 min 向量（ElementY）
    AscendC::LocalTensor<ElementY> UbAMaxTensorforYList[STAGES]; // 存储 A 的行 max 向量 (ElementY)
    AscendC::LocalTensor<ElementZ> UbZTensorList[STAGES]; // Z 为比较结果的向量

    // Multi-stage event id list
    int32_t UbInCEventList[STAGES]; // 矩阵C输入向量
    int32_t UbInARedEventList[STAGES];
    int32_t UbInAMaxEventList[STAGES];

    int32_t UbOutEventList[STAGES];
    int32_t UbOutZEventList[STAGES];

    int32_t UbInBRedEvent;
    // int32_t UbInMaxEvent;

    // The id of current stage
    uint32_t UbOutListId{0};
    uint32_t UbZOutListId{0};
    uint32_t UbInListId{0};

    ElementY B_slice_meanabs;
    ElementY B_slice_meansquare;
    ElementY B_slice_var;
    ElementY B_slice_var_square;

    uint32_t m_actual, n_actual, x_actual, y_actual;
    uint32_t m_actual_total, n_actual_total, x_actual_total, y_actual_total;
    uint32_t m_actual_next_A, n_actual_next_A, x_actual_next_A, y_actual_next_A;
    uint32_t m_actual_total_next, n_actual_total_next, x_actual_total_next, y_actual_total_next;
    uint32_t m_actual_part;
    uint32_t m_actual_part_next;

    // uint32_t thre_n_actual_total;
    // uint32_t thre_n_actual;

    uint32_t dst_offset_ratio;
    uint32_t out_z_actual_total, out_z_actual;

    uint32_t TileMRound, TileNRound;
    uint32_t ThreTileMRound, ThreTileNRound;

    uint32_t BlockMRound, BlockNRound;
    uint32_t ThreBlockMRound, ThreBlockNRound;

    uint32_t TaskSplit;
    uint32_t MatrixOffset;
    uint32_t strideCRow, strideCCol;

    uint32_t strideOut;

    MatrixCopyGmToUb matrixCopyGmToUb;
    VecCopyGmToUb vecCopyGmToUb;
    VecCopyUbToGm vecCopyUbToGm;

    MatrixCopyGmToUbforThre matrixCopyGmToUbforThre; // 用来来数据做阈值计算
    VecCopyGmToUbInY vecCopyGmToUbInY; // 用来拉A聚合数据
    VecCopyUbToGmZ vecCopyUbToGmZ; // 用来输出比较结果

    VecCopyUbToGmforThre vecCopyUbToGmforThre; // 用来输出阈值结果

    TileThreCalc tileThreCalc;
    TileFaultSumCSum tileFaultSumCSum;

    // Tile Compare
    TileCompare tileCompare;
    TileStdEst tileStdEst;
};

} // namespace FTSelf::Gemv::Block

#endif // FTSELF_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
