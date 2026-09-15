/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_FTSELF_GEMM_BLOCK_BLOCK_MMAD_HPP
#define FTSELF_FTSELF_GEMM_BLOCK_BLOCK_MMAD_HPP

#include "../../gemm/tile/gemm_tile_copy.hpp"
#include "../../gemm/tile/tile_mmad.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemm::Block {

#define FTSELF_BLOCK_MMAD_BASIC_PARAMS \
    class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType, \
    class BiasType = void, \
    class TileCopy = FTSelf::Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>, \
    class TileMmad = FTSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>

#define FTSELF_BLOCK_MMAD_FT_PARAMS \
    class DispatchPolicy, FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_, \
    FTSelf::Gemv::helper::FT_L02L1_TYPE COPY_TYPE_, class L1TileShape, class L0TileShape, \
    class L0TileShapeforFT, class AType, class BType, class CType, class XType, class YType, \
    class BiasType = void, \
    class TileCopyFT = FTSelf::Gemm::Tile::TileCopyFT<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType, COPY_TYPE_>, \
    class TileMmad = FTSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>

#define FTSELF_BLOCK_MMAD_AB_PARAMS \
    class DispatchPolicy, class L1TileShape, class L1TileShapeforFT, class L0TileShape, \
    class L0TileShapeforFT, class AType, class BType, class CType, class XType, class YType, \
    class BiasType = void, \
    class TileCopyFTABonAic = FTSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>, \
    class TileMmad = FTSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>

#define FTSELF_BLOCK_MMAD_AUGED_PARAMS \
    class DispatchPolicy, class L1TileShape, class L1TileShapeforFT, class L0TileShape, \
    class L0TileShapeforFT, class AType, class BType, class CType, class XType, class XColType, \
    class YType, class BiasType = void, \
    class TileCopyFTABonAicAuged = FTSelf::Gemm::Tile::TileCopyFTABonAicAuged<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, XColType, YType, BiasType>, \
    class TileMmad = FTSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>

#define FTSELF_BLOCK_MMAD_SPEC_PARAMS \
    class DispatchPolicy, class L1TileShapeforFT, class L0TileShapeforFT, class AType, class BType, \
    class CType, class XType, class YType, class BiasType, \
    class TileCopyFTABonAic = FTSelf::Gemm::Tile::TileCopyFTABonAic<typename DispatchPolicy::ArchTag, AType, BType, CType, XType, YType, BiasType>, \
    class TileMmad = FTSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, XType, BiasType>



template <FTSELF_BLOCK_MMAD_BASIC_PARAMS>
struct BlockMmadPreload {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadPreload is not implemented for this DispatchPolicy");
};



// using TileMmadAIC = Gemm::Tile::TileMmad<typename GEMVAICDispatchPolicy::ArchTag, XType, CType, BiasType>;
// class TileMmadforFT = FTSelf::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag,>
//  = FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE



template <FTSELF_BLOCK_MMAD_FT_PARAMS>
struct BlockMmadFTNOSPLIT {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

/*
struct BlockMmadFTABeNoSplitK<
    FTSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    YType_,
    BiasType_,
    TileCopyFTABonAic_,
    TileMmad_
>
*/
template <FTSELF_BLOCK_MMAD_AB_PARAMS>
struct BlockMmadFTABeNoSplitK {
    /*
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    */
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <FTSELF_BLOCK_MMAD_AUGED_PARAMS>
struct BlockMmadFTABeAugedNoSplitK{

    /*
    FTSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitK is not implemented for this DispatchPolicy");
};


template <FTSELF_BLOCK_MMAD_AUGED_PARAMS>
struct BlockMmadFTABeAugedNoSplitKGemv{

    /*
    FTSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitKGemv is not implemented for this DispatchPolicy");
};

template <FTSELF_BLOCK_MMAD_AUGED_PARAMS>
struct BlockMmadFTABeAugedNoSplitKRobust{

    /*
    FTSelf::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L1TileShapeforFT_,
    L0TileShape_,
    L0TileShapeforFT_,
    AType_,
    BType_,
    CType_,
    XType_,
    XColType_,
    YType_,
    BiasType_,
    TileCopyFTABonAicAuged_,
    TileMmad_
    */
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTABeAugedNoSplitKRobust is not implemented for this DispatchPolicy");
};

template <FTSELF_BLOCK_MMAD_FT_PARAMS>
struct BlockMmadFTSpiltK {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFTSpiltK is not implemented for this DispatchPolicy");
};

template <FTSELF_BLOCK_MMAD_SPEC_PARAMS>
struct BlockMmadSpecABeNoSplitK {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecABeNoSplitK is not implemented for this DispatchPolicy");
};

template <FTSELF_BLOCK_MMAD_SPEC_PARAMS>
struct BlockMmadSpecABeNoSplitKRobust {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadSpecABeNoSplitK is not implemented for this DispatchPolicy");
};

template <FTSELF_BLOCK_MMAD_AB_PARAMS>
struct BlockMmadFTABeNoSplitKRobust {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <FTSELF_BLOCK_MMAD_BASIC_PARAMS>
struct BlockMmadFault {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadFault is not implemented for this DispatchPolicy");
};


#undef FTSELF_BLOCK_MMAD_SPEC_PARAMS
#undef FTSELF_BLOCK_MMAD_AUGED_PARAMS
#undef FTSELF_BLOCK_MMAD_AB_PARAMS
#undef FTSELF_BLOCK_MMAD_FT_PARAMS
#undef FTSELF_BLOCK_MMAD_BASIC_PARAMS

} // namespace FTSelf::Gemm::Block

#include "../../gemm/block/block_mmad_pingpong_fault_abe_spec_no_splitk_robust.hpp"
#include "../../gemm/block/block_mmad_pingpong_preload.hpp"


#endif // FTSELF_FTSELF_GEMM_BLOCK_BLOCK_MMAD_HPP
