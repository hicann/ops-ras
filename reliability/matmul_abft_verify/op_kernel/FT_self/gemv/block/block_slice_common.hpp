/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_BLOCK_BLOCK_SLICE_COMMON_HPP
#define FTSELF_GEMV_BLOCK_BLOCK_SLICE_COMMON_HPP

#include <cstdint>

#define FTSELF_DECLARE_SLICE_KMN_BUFFER_CONSTRUCTORS() \
    FTSELF_DEVICE BlockSliceKMNSum() = default; \
    FTSELF_DEVICE BlockSliceKMNSum(FTSelf::Arch::Resource<FTSelf::Arch::AtlasA2> &resource, \
        uint32_t UBufAddrStart = 0) \
    { \
        InitializeBuffers(resource, UBufAddrStart); \
    } \
    FTSELF_DEVICE BlockSliceKMNSum(FTSelf::ResourceAIV<ArchTag> &resource, uint32_t UBufAddrStart = 0) \
    { \
        InitializeBuffers(resource, UBufAddrStart); \
    }

namespace FTSelf::Gemv::Block::detail {

struct SliceBufferOffsets {
    uint32_t a;
    uint32_t y;
};

struct SliceAeConfig {
    uint32_t tileM;
    uint32_t tileN;
    uint32_t strideA;
    uint32_t strideY;
    uint32_t strideSlice;
    uint32_t strideXforAe;
    uint32_t strideYforAe;
    uint32_t strideSliceforAe;
};

FTSELF_DEVICE constexpr SliceBufferOffsets MakeSliceBufferOffsets(uint32_t start, uint32_t aSize)
{
    return {start, start + aSize};
}

template <class TileShape, class AlignHelper, class LayoutA, class LayoutY>
FTSELF_DEVICE SliceAeConfig MakeSliceAeConfig(LayoutA const &layoutA, LayoutY const &layoutY)
{
    uint32_t tileM = FTSelf::helper::RoundUp(TileShape::M, AlignHelper::ALIGN);
    uint32_t tileN = FTSelf::helper::RoundUp(TileShape::N, AlignHelper::ALIGN);
    return {tileM, tileN, static_cast<uint32_t>(layoutA.stride(1) * tileN),
        static_cast<uint32_t>(layoutY.stride(1) * tileN),
        static_cast<uint32_t>(layoutA.shape(0) * layoutA.shape(1)), tileN, tileN,
        static_cast<uint32_t>(layoutA.shape(1))};
}

} // namespace FTSelf::Gemv::Block::detail

#endif
