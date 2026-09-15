/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_COMMON_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_SLICEKMN_RED_SUM_FUSED_COMMON_HPP

#define FTSELF_DECLARE_SLICE_KMN_FUSED_SPECIALIZATION(FUSE_TYPE) \
template < \
    class UBTileShapeforA_, class UBTileShapeforB_, \
    class AType_, class BType_, class XType_, class YType_, class BiasType_, \
    class TileCopyforA_, class TileCopyforB_, class TileMatrixAdd_, \
    class TileFaultSum_, class TileVmuls_> \
struct BlockSliceKMNSum< \
    FTSelf::Gemv::GemvAtlasA2, FUSE_TYPE, UBTileShapeforA_, UBTileShapeforB_, \
    AType_, BType_, XType_, YType_, BiasType_, TileCopyforA_, TileCopyforB_, \
    TileMatrixAdd_, TileFaultSum_, TileVmuls_>

namespace FTSelf::Gemv::Block::detail {

struct SliceKmnFusedLoopConfig {
    uint32_t tileM;
    uint32_t tileN;
    uint32_t strideA;
    uint32_t strideY;
    uint32_t strideSlice;
};

template <class UBTileShape, class AlignHelper, class LayoutA, class LayoutY>
FTSELF_DEVICE
SliceKmnFusedLoopConfig MakeSliceKmnFusedLoopConfig(LayoutA const &layoutA, LayoutY const &layoutY)
{
    SliceKmnFusedLoopConfig config;
    config.tileM = FTSelf::helper::RoundUp(UBTileShape::M, AlignHelper::ALIGN);
    config.tileN = FTSelf::helper::RoundUp(UBTileShape::N, AlignHelper::ALIGN);
    config.strideA = layoutA.stride(1) * config.tileN;
    config.strideY = layoutY.stride(1) * config.tileN;
    config.strideSlice = layoutA.shape(0) * layoutA.shape(1);
    return config;
}

} // namespace FTSelf::Gemv::Block::detail

#endif
