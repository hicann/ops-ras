/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_FTSELF_GEMM_TILE_ATLASA2_COPY_L1_TO_L0B_HPP
#define FTSELF_FTSELF_GEMM_TILE_ATLASA2_COPY_L1_TO_L0B_HPP

#include <cstdint>
#include <type_traits>

#include "../../../core/coord.hpp"
#include "../../../core/gemm_type.hpp"
#include "../../../core/layout.hpp"
#include "../../../core/macros.hpp"

namespace FTSelf::Gemm::Tile {

template <class>
inline constexpr bool COPY_L1_TO_L0B_UNSUPPORTED = false;

template <class ArchTag, class SrcType, class DstType = void>
struct CopyL1ToL0B {
    static_assert(COPY_L1_TO_L0B_UNSUPPORTED<SrcType>,
        "Unsupported FTSelf L1 to L0B copy specialization");
};

// Standard GEMM path: A1(zN) -> B2(nZ).
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag,
    FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>,
    FTSelf::GemmType<Element, FTSelf::layout::nZ, AscendC::TPosition::B2>> {
    using LayoutDst = FTSelf::layout::nZ;
    using LayoutSrc = FTSelf::layout::zN;

    static constexpr uint32_t ELEMENTS_PER_C0 = FTSelf::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELEMENTS_PER_FRACTAL = FTSelf::BYTE_PER_FRACTAL / sizeof(Element);

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc) const
    {
        if constexpr (std::is_same_v<Element, float>) {
            constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
            uint16_t l1K = layoutSrc.shape(0) * layoutSrc.shape(1);
            uint16_t l1N = layoutSrc.shape(2) * layoutSrc.shape(3);
            uint16_t l0K = layoutDst.shape(0) * layoutDst.shape(1);
            uint16_t l0N = layoutDst.shape(2) * layoutDst.shape(3);
            uint16_t l1KAlign = ((l1K + 15) / 16) * 16;
            uint16_t l1NAlign = ((l1N + 15) / 16) * 16;
            uint16_t l0KAlign = ((l0K + 15) / 16) * 16;
            uint16_t l0NAlign = ((l0N + 15) / 16) * 16;

            AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_RIGHT);
            static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
            AscendC::LoadData3DParamsV2<Element> params;
            params.kExtension = l0NAlign;
            params.mExtension = l0KAlign;
            params.channelSize = l1NAlign;
            params.fMatrixCtrl = true;
            AscendC::LoadData<Element, config>(dstTensor, srcTensor, params);
            return;
        }

        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint16_t>(
            (layoutDst.orgShape(1) + ELEMENTS_PER_C0 - 1) / ELEMENTS_PER_C0);
        params.srcStride = layoutSrc.stride(3) / ELEMENTS_PER_FRACTAL;
        params.sid = 0;
        params.dstGap = layoutDst.stride(3) / ELEMENTS_PER_FRACTAL - 1;
        params.ifTranspose = true;
        params.addrMode = 0;

        for (uint32_t row = 0; row < (layoutDst.orgShape(0) + 15) / 16; ++row) {
            AscendC::LoadData(
                dstTensor[row * layoutDst.stride(1)],
                srcTensor[row * layoutSrc.stride(1)],
                params);
        }
    }
};

// Compatibility specialization required by the matmul_ft GEMV path. This
// A1(zN) -> B2(zN) form is not provided by the current upstream FTSelf.
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag,
    FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>,
    FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::B2>> {
    using LayoutDst = FTSelf::layout::zN;
    using LayoutSrc = FTSelf::layout::zN;

    static constexpr uint32_t ELE_NUM_PER_FRACTAL = FTSelf::BYTE_PER_FRACTAL / sizeof(Element);

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc) const
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(3); i++) {
            AscendC::LoadData(
                dstTensor[i * layoutDst.stride(3)],
                srcTensor[i * layoutSrc.stride(3)],
                loadDataParams);
        }
    }
};


} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_FTSELF_GEMM_TILE_ATLASA2_COPY_L1_TO_L0B_HPP
