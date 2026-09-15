/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_SLICE_SUM_HPP
#define FTSELF_GEMV_TILE_TILE_SLICE_SUM_HPP

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../helper/type_helper.hpp"

namespace FTSelf::Gemv::Tile {

template <class ArchTag, class AType, class YType, class BiasType = void, bool TreeReduce = false>
struct TileSliceSum
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileSliceSum, can not find the specialization.");
};

template <class ElementA, class ElementY, bool TreeReduce>
struct TileSliceSum<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>,
                void,
                TreeReduce>
{
    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);

    FTSELF_DEVICE
    TileSliceSum() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t nActual = FTSelf::helper::GetShape(layoutSrc, 1);
        uint32_t nRound = FTSelf::helper::GetShape(layoutDst, 1);
        uint32_t repeatSize = ELE_NUM_PER_C0 * 8;
        auto addParams = FTSelf::Gemv::MakeBinaryRepeatParams(1, 1, 1, 8, 8, 8);

        if constexpr (TreeReduce) {
            ReduceRowsByTree(srcTensor, mActual, nActual, repeatSize, addParams);
        } else {
            ReduceRowsSequentially(srcTensor, mActual, nActual, repeatSize, addParams);
        }
        FTSelf::Gemv::helper::VectorBarrier();

        if constexpr (std::is_same_v<ElementA, ElementY>) {
            AscendC::Add(dstTensor, dstTensor, srcTensor, nRound);
        } else {
            uint32_t dstOffset = nRound * 2;
            AscendC::Cast<ElementY, ElementA>(
                dstTensor[dstOffset], srcTensor, AscendC::RoundMode::CAST_NONE, nActual);
            FTSelf::Gemv::helper::VectorBarrier();
            AscendC::Add(dstTensor, dstTensor, dstTensor[dstOffset], nRound);
        }
        FTSelf::Gemv::helper::VectorBarrier();
    }

private:
    FTSELF_DEVICE
    static void ReduceRowsSequentially(
        AscendC::LocalTensor<ElementA> srcTensor,
        uint32_t mActual,
        uint32_t nActual,
        uint32_t repeatSize,
        AscendC::BinaryRepeatParams const &addParams)
    {
        if (repeatSize == 0) {
            return;
        }
        uint32_t repeatNum = nActual / repeatSize;
        uint32_t remain = nActual % repeatSize;
        if (repeatNum > 0) {
            for (uint32_t row = 1; row < mActual; ++row) {
                uint32_t offset = row * nActual;
                AscendC::Add<ElementA, true>(srcTensor, srcTensor[offset], srcTensor,
                    static_cast<uint64_t>(repeatSize), repeatNum, addParams);
                FTSelf::Gemv::helper::VectorBarrier();
            }
        }
        if (remain > 0) {
            uint32_t columnOffset = repeatNum * repeatSize;
            if (columnOffset + remain > nActual) {
                remain = nActual - columnOffset;
            }
            for (uint32_t row = 1; row < mActual; ++row) {
                uint32_t rowOffset = row * nActual;
                AscendC::Add<ElementA>(srcTensor[columnOffset],
                    srcTensor[rowOffset + columnOffset], srcTensor[columnOffset], remain);
                FTSelf::Gemv::helper::VectorBarrier();
            }
        }
    }

    FTSELF_DEVICE
    static void ReduceRowsByTree(
        AscendC::LocalTensor<ElementA> srcTensor,
        uint32_t mActual,
        uint32_t nActual,
        uint32_t repeatSize,
        AscendC::BinaryRepeatParams const &addParams)
    {
        if (repeatSize == 0) {
            return;
        }
        for (uint32_t rows = mActual; rows > 1; rows /= 2) {
            uint32_t remainingRows = rows % 2;
            uint32_t alignedRows = rows - remainingRows;
            uint32_t alignedOffset = alignedRows / 2 * nActual;
            uint32_t repeatNum = alignedOffset / repeatSize;
            uint32_t remain = alignedOffset % repeatSize;
            uint32_t remainOffset = alignedOffset - remain;

            if (repeatNum > 0) {
                AscendC::Add<ElementA, true>(srcTensor, srcTensor[alignedOffset], srcTensor,
                    static_cast<uint64_t>(repeatSize), repeatNum, addParams);
                FTSelf::Gemv::helper::VectorBarrier();
            }
            if (remain > 0) {
                AscendC::Add<ElementA>(srcTensor[remainOffset],
                    srcTensor[remainOffset + alignedOffset], srcTensor[remainOffset], remain);
                FTSelf::Gemv::helper::VectorBarrier();
            }
            if (remainingRows > 0) {
                AscendC::Add<ElementA>(srcTensor, srcTensor[alignedRows * nActual],
                    srcTensor, remainingRows * nActual);
                FTSelf::Gemv::helper::VectorBarrier();
            }
        }
    }
};

}

#endif
