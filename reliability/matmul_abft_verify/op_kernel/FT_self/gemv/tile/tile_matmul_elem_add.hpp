/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_MATMUL_ELEM_ADD_HPP_SELF
#define FTSELF_GEMV_TILE_TILE_MATMUL_ELEM_ADD_HPP_SELF

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

template <class ArchTag, class AType, class YType, class BiasType = void>
struct TileMatmulAdd
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileMatmulAdd, can not find the specialization.");
};

template <class Derived, class ElementA, class ElementY, class Layout>
struct TileMatmulAddInterface
{
    FTSELF_DEVICE
    TileMatmulAddInterface() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensorM1,
        AscendC::LocalTensor<ElementA> srcTensorM2,
        Layout const &layoutDst,
        Layout const &layoutSrc)
    {
        static_cast<Derived *>(this)->Run(dstTensor, srcTensorM1, srcTensorM2, layoutDst, layoutSrc);
    }
};

template <class ElementA, class ElementY>
struct TileMatmulAdd<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<ElementY, FTSelf::layout::RowMajor>,
                void> : TileMatmulAddInterface<
                    TileMatmulAdd<FTSelf::Gemv::Arch::AtlasA2,
                        FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
                        FTSelf::GemmType<ElementY, FTSelf::layout::RowMajor>, void>,
                    ElementA, ElementY, FTSelf::layout::RowMajor>
{
    using ElementAccumulator = ElementY;
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);

    FTSELF_DEVICE
    void Run(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensorM1,
        AscendC::LocalTensor<ElementA> srcTensorM2,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        uint32_t mActual = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t nActual = FTSelf::helper::GetShape(layoutSrc, 1);
        uint32_t nRound = FTSelf::helper::GetShape(layoutDst, 1);
        uint32_t repeatSize = ELE_NUM_PER_C0 * 8;
        uint32_t repeatNum = nActual / repeatSize;
        uint32_t remain = nActual % repeatSize;
        auto params = FTSelf::Gemv::MakeBinaryRepeatParams(
            1, 1, 1,
            FTSelf::helper::RoundUp(nRound, repeatSize) / ELE_NUM_PER_C0,
            FTSelf::helper::RoundUp(nRound, repeatSize) / ELE_NUM_PER_C0,
            FTSelf::helper::RoundUp(nRound, repeatSize) / ELE_NUM_PER_C0);

        auto addDst = srcTensorM1;
        if constexpr (std::is_same_v<ElementA, ElementY>) {
            addDst = dstTensor;
        }
        for (uint32_t i = 0; i < repeatNum; ++i) {
            uint32_t offset = i * repeatSize;
            AscendC::Add<ElementA, true>(addDst[offset], srcTensorM1[offset], srcTensorM2[offset],
                static_cast<uint64_t>(repeatSize), mActual, params);
        }
        if (remain > 0) {
            uint32_t offset = repeatNum * repeatSize;
            if (offset + remain > nActual) {
                remain = nActual - offset;
            }
            AscendC::Add<ElementA, true>(addDst[offset], srcTensorM1[offset], srcTensorM2[offset],
                static_cast<uint64_t>(remain), mActual, params);
        }
        FTSelf::Gemv::helper::VectorBarrier();

        if constexpr (!std::is_same_v<ElementA, ElementY>) {
            CastResult(dstTensor, srcTensorM1, mActual, nActual, nRound);
        }
    }

private:
    FTSELF_DEVICE
    static void CastResult(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor,
        uint32_t mActual,
        uint32_t nActual,
        uint32_t nRound)
    {
        uint32_t repeatSize = DST_ELE_NUM_PER_C0 * 8;
        uint32_t repeatNum = nActual / repeatSize;
        uint32_t remain = nActual % repeatSize;
        auto params = FTSelf::Gemv::MakeUnaryRepeatParams(
            1, 1,
            FTSelf::helper::RoundUp(nRound, repeatSize) / DST_ELE_NUM_PER_C0,
            FTSelf::helper::RoundUp(nRound, repeatSize) / ELE_NUM_PER_C0);

        for (uint32_t i = 0; i < repeatNum; ++i) {
            uint32_t offset = i * repeatSize;
            AscendC::Cast<ElementY, ElementA, true>(dstTensor[offset], srcTensor[offset],
                AscendC::RoundMode::CAST_NONE, static_cast<uint64_t>(repeatSize), mActual, params);
        }
        if (remain > 0) {
            uint32_t offset = repeatNum * repeatSize;
            if (offset + remain > nActual) {
                remain = nActual - offset;
            }
            AscendC::Cast<ElementY, ElementA, true>(dstTensor[offset], srcTensor[offset],
                AscendC::RoundMode::CAST_NONE, static_cast<uint64_t>(remain), mActual, params);
        }
        FTSelf::Gemv::helper::VectorBarrier();
    }
};

template <class ElementA, class ElementY>
struct TileMatmulAdd<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::GemmType<ElementA, FTSelf::layout::VectorLayout>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>,
                void> : TileMatmulAddInterface<
                    TileMatmulAdd<FTSelf::Gemv::Arch::AtlasA2,
                        FTSelf::GemmType<ElementA, FTSelf::layout::VectorLayout>,
                        FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>, void>,
                    ElementA, ElementY, FTSelf::layout::VectorLayout>
{
    using ElementAccumulator = ElementY;
    using LayoutDst = FTSelf::layout::VectorLayout;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr uint32_t MAX_COMPUTE_LENGTH = 8192;

    FTSELF_DEVICE
    void Run(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensorM1,
        AscendC::LocalTensor<ElementA> srcTensorM2,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        uint32_t totalElementNum = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t repeatNum = totalElementNum / MAX_COMPUTE_LENGTH;
        uint32_t remain = totalElementNum % MAX_COMPUTE_LENGTH;

        for (uint32_t i = 0; i < repeatNum; ++i) {
            ComputeChunk(dstTensor, srcTensorM1, srcTensorM2,
                i * MAX_COMPUTE_LENGTH, MAX_COMPUTE_LENGTH);
        }
        if (remain > 0) {
            uint32_t offset = repeatNum * MAX_COMPUTE_LENGTH;
            if (offset + remain > totalElementNum) {
                remain = totalElementNum - offset;
            }
            ComputeChunk(dstTensor, srcTensorM1, srcTensorM2, offset, remain);
        }
        FTSelf::Gemv::helper::VectorBarrier();
    }

private:
    FTSELF_DEVICE
    static void ComputeChunk(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensorM1,
        AscendC::LocalTensor<ElementA> srcTensorM2,
        uint32_t offset,
        uint32_t count)
    {
        if constexpr (std::is_same_v<ElementA, ElementY>) {
            AscendC::Add(dstTensor[offset], srcTensorM1[offset], srcTensorM2[offset], count);
        } else {
            AscendC::Add(srcTensorM1[offset], srcTensorM1[offset], srcTensorM2[offset], count);
            AscendC::Cast<ElementY, ElementA>(dstTensor[offset], srcTensorM1[offset],
                AscendC::RoundMode::CAST_NONE, count);
        }
    }
};

template <class ElementA, class ElementY>
struct TileMatmulAdd<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>,
                void>
{
};

}
#endif // FTSELF_GEMV_TILE_TILE_MATMUL_ELEM_ADD_HPP_SELF
