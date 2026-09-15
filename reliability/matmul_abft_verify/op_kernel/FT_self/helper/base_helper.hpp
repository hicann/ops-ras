/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_HELPER_BASE_HELPER_HPP
#define FTSELF_HELPER_BASE_HELPER_HPP

#include <cstdint>
#include "../core/macros.hpp"
#include "../core/arch.hpp"

#if defined(__CCE__)
#include <kernel_operator.h>
#define FTSELF_GEMV_HOST_DEVICE __forceinline__ [host, aicore]
#else
#define FTSELF_GEMV_HOST_DEVICE
#endif

// AscendC tile specializations use the same reduction geometry variables.  Keep
// their declarations in one place: these names intentionally become locals in
// the caller so existing vector-intrinsic expressions remain unchanged.
#define FTSELF_INIT_REDUCE_GEOMETRY(layoutDst, layoutSrc, elementsPerRepeatBlock) \
    const uint32_t m_actual = FTSelf::helper::GetShape((layoutSrc), 0); \
    const uint32_t n_actual = FTSelf::helper::GetShape((layoutSrc), 1); \
    const uint32_t m_round = FTSelf::helper::GetShape((layoutDst), 0); \
    const uint32_t n_round = FTSelf::helper::GetShape((layoutDst), 1); \
    const uint32_t repeat_size = (elementsPerRepeatBlock) * 8U; \
    const uint32_t mask = repeat_size; \
    const uint32_t repeat_num = n_actual / repeat_size; \
    uint32_t remain = n_actual % repeat_size

#define FTSELF_INIT_DUAL_REDUCE_GEOMETRY(layoutDst, layoutSrc, srcElementsPerBlock, dstElementsPerBlock) \
    FTSELF_INIT_REDUCE_GEOMETRY((layoutDst), (layoutSrc), (srcElementsPerBlock)); \
    const uint32_t dst_repeat_size = (dstElementsPerBlock) * 8U; \
    const uint32_t dst_mask = dst_repeat_size; \
    const uint32_t dst_repeat_num = n_actual / dst_repeat_size; \
    uint32_t dst_remain = n_actual % dst_repeat_size

namespace FTSelf::Gemv {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BYTE_PER_FRACTAL = 512;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t STRIDE_LIMIT = 65536;

template <typename DstBlk, typename Src0Blk, typename Src1Blk,
    typename DstRep, typename Src0Rep, typename Src1Rep>
FTSELF_DEVICE AscendC::BinaryRepeatParams MakeBinaryRepeatParams(
    DstBlk dstBlkStride, Src0Blk src0BlkStride, Src1Blk src1BlkStride,
    DstRep dstRepStride, Src0Rep src0RepStride, Src1Rep src1RepStride)
{
    AscendC::BinaryRepeatParams params;
    params.dstBlkStride = dstBlkStride;
    params.src0BlkStride = src0BlkStride;
    params.src1BlkStride = src1BlkStride;
    params.dstRepStride = dstRepStride;
    params.src0RepStride = src0RepStride;
    params.src1RepStride = src1RepStride;
    return params;
}

template <typename DstBlk, typename SrcBlk, typename DstRep, typename SrcRep>
FTSELF_DEVICE AscendC::UnaryRepeatParams MakeUnaryRepeatParams(
    DstBlk dstBlkStride, SrcBlk srcBlkStride, DstRep dstRepStride, SrcRep srcRepStride)
{
    AscendC::UnaryRepeatParams params;
    params.dstBlkStride = dstBlkStride;
    params.srcBlkStride = srcBlkStride;
    params.dstRepStride = dstRepStride;
    params.srcRepStride = srcRepStride;
    return params;
}

template <typename Element>
FTSELF_DEVICE void WholeReduceMaxInPlace(
    AscendC::LocalTensor<Element> tensor, uint64_t mask, uint32_t repeat,
    uint16_t sourceRepeatStride)
{
    AscendC::WholeReduceMax<Element, true>(
        tensor, tensor, mask, repeat, 1, 1, sourceRepeatStride,
        AscendC::ReduceOrder::ORDER_ONLY_VALUE);
}

template <typename Element>
FTSELF_DEVICE void WholeReduceSumInPlace(
    AscendC::LocalTensor<Element> tensor, uint64_t mask, uint32_t repeat,
    uint16_t sourceRepeatStride)
{
    AscendC::WholeReduceSum<Element, true>(
        tensor, tensor, mask, repeat, 1, 1, sourceRepeatStride);
}

} // namespace FTSelf::Gemv

#endif // FTSELF_HELPER_BASE_HELPER_HPP
