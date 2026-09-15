/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_VMAD_HPP
#define FTSELF_GEMV_TILE_TILE_VMAD_HPP

#include "../../helper/base_helper.hpp"
#include "../../helper/layout_helper.hpp"
#include "../../helper/math_helper.hpp"

namespace FTSelf::Gemv::Tile {

namespace vmad_detail {
template <class T>
using Accumulator = std::conditional_t<
    std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>, float,
    std::conditional_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, int32_t, T>>;
} // namespace vmad_detail

template <class ArchTag, class AType, class XType, class YType, class BiasType = void>
struct TileVmad {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator = vmad_detail::Accumulator<ElementA>;
    using LayoutDst = typename AType::Layout;
    using LayoutSrc = typename AType::Layout;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t ELEM_REPEAT = ELE_NUM_PER_C0 * 8;
    static constexpr uint32_t ACC_REPEAT = FTSelf::Gemv::BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);

    FTSELF_GEMV_HOST_DEVICE TileVmad() = default;

    FTSELF_GEMV_HOST_DEVICE
    void operator()(AscendC::LocalTensor<ElementY> dst, AscendC::LocalTensor<ElementX> vector,
                    AscendC::LocalTensor<ElementA> matrix, AscendC::LocalTensor<ElementAccumulator> temp,
                    LayoutDst const &dstLayout, LayoutSrc const &srcLayout) const {
        if (FTSelf::helper::GetStride(srcLayout, 0) == 1) {
            ColumnMajor(dst, vector, matrix, temp, dstLayout, srcLayout);
        } else {
            RowMajor(dst, vector, matrix, temp, dstLayout, srcLayout);
        }
    }

private:
    FTSELF_GEMV_HOST_DEVICE
    void RowMajor(AscendC::LocalTensor<ElementY> dst, AscendC::LocalTensor<ElementX> vector,
                  AscendC::LocalTensor<ElementA> matrix, AscendC::LocalTensor<ElementAccumulator> temp,
                  LayoutDst const &dstLayout, LayoutSrc const &srcLayout) const {
        const uint32_t m = FTSelf::helper::GetShape(srcLayout, 0);
        const uint32_t n = FTSelf::helper::GetShape(srcLayout, 1);
        const uint32_t mRound = FTSelf::helper::GetShape(dstLayout, 0);
        const uint32_t nRound = FTSelf::helper::GetShape(dstLayout, 1);
        AscendC::BinaryRepeatParams p;
        p.dstBlkStride = 1; p.src0BlkStride = 1; p.src1BlkStride = 1; p.src1RepStride = 0;

        if constexpr (std::is_same_v<ElementA, ElementAccumulator>) {
            const uint32_t count = n / ELEM_REPEAT;
            uint32_t remain = n % ELEM_REPEAT;
            p.dstRepStride = FTSelf::helper::RoundUp(nRound, ELEM_REPEAT) / ELE_NUM_PER_C0;
            p.src0RepStride = p.dstRepStride;
            FTSelf::Gemv::helper::SetCounterMask<ElementA>(m * ELEM_REPEAT);
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t offset = i * ELEM_REPEAT;
                if (i == 0) {
                    AscendC::Mul<ElementA, false>(matrix, matrix, vector, AscendC::MASK_PLACEHOLDER, 1, p);
                } else {
                    AscendC::MulAddDst<ElementA, ElementA, false>(matrix, matrix[offset], vector[offset],
                        AscendC::MASK_PLACEHOLDER, 1, p);
                }
                FTSelf::Gemv::helper::VectorBarrier();
            }
            AscendC::SetMaskNorm(); AscendC::ResetMask();
            if (remain > 0) {
                const uint32_t offset = count * ELEM_REPEAT;
                remain = offset + remain > nRound ? nRound - offset : remain;
                if (count == 0) {
                    AscendC::Mul<ElementA, true>(matrix, matrix, vector, remain, m, p);
                } else {
                    AscendC::MulAddDst<ElementA, ElementA, true>(matrix, matrix[offset], vector[offset], remain, m, p);
                }
            }
            FTSelf::Gemv::helper::VectorBarrier();
            AscendC::WholeReduceSum<ElementA, true>(matrix, matrix, count == 0 ? remain : ELEM_REPEAT,
                m, 1, 1, FTSelf::helper::RoundUp(nRound, ELEM_REPEAT) / ELE_NUM_PER_C0);
            FTSelf::Gemv::helper::VectorBarrier();
            p.dstRepStride = 8; p.src0RepStride = 8; p.src1RepStride = 8;
            AscendC::Add<ElementY, true>(dst, matrix, dst, m < ELEM_REPEAT ? m : ELEM_REPEAT,
                FTSelf::helper::CeilDiv(mRound, ELEM_REPEAT), p);
        } else {
            AscendC::Duplicate<ElementAccumulator>(temp, static_cast<ElementAccumulator>(0), ACC_REPEAT,
                FTSelf::helper::CeilDiv(mRound * ACC_REPEAT, ACC_REPEAT), 1, 8);
            const uint32_t count = n / ACC_REPEAT;
            uint32_t remain = n % ACC_REPEAT;
            FTSelf::Gemv::helper::VectorBarrier();
            p.dstRepStride = 8;
            p.src0RepStride = FTSelf::helper::RoundUp(nRound, ELEM_REPEAT) / ELE_NUM_PER_C0;
            FTSelf::Gemv::helper::SetCounterMask<ElementAccumulator>(m * ACC_REPEAT);
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t offset = i * ACC_REPEAT;
                AscendC::MulAddDst<ElementAccumulator, ElementA, false>(temp, matrix[offset], vector[offset],
                    AscendC::MASK_PLACEHOLDER, 1, p);
                FTSelf::Gemv::helper::VectorBarrier();
            }
            AscendC::SetMaskNorm(); AscendC::ResetMask();
            if (remain > 0) {
                const uint32_t offset = count * ACC_REPEAT;
                remain = offset + remain > nRound ? nRound - offset : remain;
                AscendC::MulAddDst<ElementAccumulator, ElementA, true>(temp, matrix[offset], vector[offset], remain, m, p);
            }
            FTSelf::Gemv::helper::VectorBarrier();
            AscendC::WholeReduceSum<ElementAccumulator, true>(temp, temp, count == 0 ? remain : ACC_REPEAT,
                m, 1, 1, 8);
            FTSelf::Gemv::helper::VectorBarrier();
            p.dstRepStride = 8; p.src0RepStride = 8; p.src1RepStride = 8;
            AscendC::Add<ElementY, true>(dst, temp, dst, m < ACC_REPEAT ? m : ACC_REPEAT,
                FTSelf::helper::CeilDiv(mRound, ACC_REPEAT), p);
        }
    }

    FTSELF_GEMV_HOST_DEVICE
    void ColumnMajor(AscendC::LocalTensor<ElementY> dst, AscendC::LocalTensor<ElementX> vector,
                     AscendC::LocalTensor<ElementA> matrix, AscendC::LocalTensor<ElementAccumulator> temp,
                     LayoutDst const &dstLayout, LayoutSrc const &srcLayout) const {
        const uint32_t m = FTSelf::helper::GetShape(srcLayout, 0);
        const uint32_t n = FTSelf::helper::GetShape(srcLayout, 1);
        const uint32_t mRound = FTSelf::helper::GetShape(dstLayout, 0);
        FTSelf::Gemv::helper::SetCounterMask<ElementAccumulator>(m);
        AscendC::Duplicate<ElementAccumulator, false>(temp, static_cast<ElementAccumulator>(0),
            AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        FTSelf::Gemv::helper::VectorBarrier();
        ElementX values[32];
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::V_S>(static_cast<event_t>(0));
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::V_S>(static_cast<event_t>(0));
        for (uint32_t i = 0; i < n; ++i) values[i] = vector.GetValue(i);
        FTSelf::Gemv::helper::SetEvent<AscendC::HardEvent::S_V>(static_cast<event_t>(0));
        FTSelf::Gemv::helper::WaitEvent<AscendC::HardEvent::S_V>(static_cast<event_t>(0));
        AscendC::UnaryRepeatParams p;
        p.dstBlkStride = 1; p.srcBlkStride = 1; p.dstRepStride = 8;
        p.srcRepStride = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);
        for (uint32_t i = 0; i < n; ++i) {
            AscendC::Axpy<ElementAccumulator, ElementA, false>(temp, matrix[i * mRound], values[i],
                AscendC::MASK_PLACEHOLDER, 1, p);
            FTSelf::Gemv::helper::VectorBarrier();
        }
        AscendC::BinaryRepeatParams add;
        add.dstBlkStride = 1; add.src0BlkStride = 1; add.src1BlkStride = 1;
        add.dstRepStride = 8; add.src0RepStride = 8; add.src1RepStride = 8;
        AscendC::Add<ElementY, false>(dst, temp, dst, AscendC::MASK_PLACEHOLDER, 1, add);
        AscendC::SetMaskNorm(); AscendC::ResetMask();
    }
};

} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_GEMV_TILE_TILE_VMAD_HPP
