/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMM_TILE_ATLASA2_COPY_GM_TO_L1_HPP
#define FTSELF_GEMM_TILE_ATLASA2_COPY_GM_TO_L1_HPP

#include <cstdint>
#include <type_traits>

#include "../../../core/coord.hpp"
#include "../../../core/gemm_type.hpp"
#include "../../../core/layout.hpp"
#include "../../../core/macros.hpp"

namespace FTSelf::Gemm::Tile {

template <class>
inline constexpr bool COPY_GM_TO_L1_UNSUPPORTED = false;

template <class ArchTag, class GmType, class L1Type = void>
struct CopyGmToL1 {
    static_assert(COPY_GM_TO_L1_UNSUPPORTED<GmType>,
        "Unsupported FTSelf GM to L1 copy specialization");
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag,
    FTSelf::GemmType<Element, FTSelf::layout::RowMajor>,
    FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = FTSelf::layout::zN;
    using LayoutSrc = FTSelf::layout::RowMajor;

    static constexpr uint32_t ELEMENTS_PER_C0 = FTSelf::BYTE_PER_C0 / sizeof(Element);

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc) const
    {
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.dValue = layoutSrc.shape(1);
        params.srcNdMatrixStride = 0;
        params.dstNzC0Stride = layoutDst.stride(3) / ELEMENTS_PER_C0;
        params.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < FTSelf::STRIDE_LIMIT) {
            params.nValue = layoutSrc.shape(0);
            params.srcDValue = layoutSrc.stride(0);
            params.dstNzNStride = layoutDst.stride(0) / ELEMENTS_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, params);
            return;
        }

        params.nValue = 1;
        params.srcDValue = 0;
        params.dstNzNStride = 0;
        for (uint32_t row = 0; row < layoutSrc.shape(0); ++row) {
            AscendC::DataCopy(
                dstTensor[row * ELEMENTS_PER_C0],
                srcTensor[row * layoutSrc.stride(0)],
                params);
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag,
    FTSelf::GemmType<Element, FTSelf::layout::RowMajor>,
    void>
    : CopyGmToL1<
          ArchTag,
          FTSelf::GemmType<Element, FTSelf::layout::RowMajor>,
          FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>> {
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag,
    FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>,
    FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = FTSelf::layout::zN;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr uint32_t ELEMENTS_PER_C0 = FTSelf::BYTE_PER_C0 / sizeof(Element);

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc) const
    {
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = 1;
        params.dValue = layoutSrc.shape(0);
        params.srcDValue = layoutSrc.shape(0);
        params.srcNdMatrixStride = 0;
        params.dstNzNStride = layoutDst.stride(0) / ELEMENTS_PER_C0;
        params.dstNzC0Stride = layoutDst.stride(3) / ELEMENTS_PER_C0;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_GEMM_TILE_ATLASA2_COPY_GM_TO_L1_HPP
