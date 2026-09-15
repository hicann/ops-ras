/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMM_TILE_ATLASA2_COPY_L1_TO_L0A_HPP
#define FTSELF_GEMM_TILE_ATLASA2_COPY_L1_TO_L0A_HPP

#include <cstdint>
#include <type_traits>

#include "../../../core/coord.hpp"
#include "../../../core/gemm_type.hpp"
#include "../../../core/layout.hpp"
#include "../../../core/macros.hpp"

namespace FTSelf::Gemm::Tile {

template <class>
inline constexpr bool COPY_L1_TO_L0A_UNSUPPORTED = false;

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A {
    static_assert(COPY_L1_TO_L0A_UNSUPPORTED<L1Type>,
        "Unsupported FTSelf L1 to L0A copy specialization");
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<
    ArchTag,
    FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>,
    FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A2>> {
    using LayoutDst = FTSelf::layout::zZ;
    using LayoutSrc = FTSelf::layout::zN;

    static constexpr uint32_t ELEMENTS_PER_FRACTAL = FTSelf::BYTE_PER_FRACTAL / sizeof(Element);

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc) const
    {
        if constexpr (std::is_same_v<Element, float>) {
            constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
            uint16_t l1M = layoutSrc.shape(0) * layoutSrc.shape(1);
            uint16_t l1K = layoutSrc.shape(2) * layoutSrc.shape(3);
            uint16_t l0M = layoutDst.shape(0) * layoutDst.shape(1);
            uint16_t l0K = layoutDst.shape(2) * layoutDst.shape(3);

            AscendC::SetFmatrix(1, l1M, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
            static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
            AscendC::LoadData3DParamsV2<Element> params;
            params.kExtension = l0K;
            params.mExtension = l0M;
            params.channelSize = l1K;
            AscendC::LoadData<Element, config>(dstTensor, srcTensor, params);
            return;
        }

        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        params.srcStride = layoutSrc.stride(3) / ELEMENTS_PER_FRACTAL;
        params.sid = 0;
        params.dstGap = layoutDst.stride(3) / ELEMENTS_PER_FRACTAL - 1;
        params.ifTranspose = false;
        params.addrMode = 0;

        for (uint32_t row = 0; row < layoutDst.shape(1); ++row) {
            AscendC::LoadData(
                dstTensor[row * layoutDst.stride(1)],
                srcTensor[row * layoutSrc.stride(1)],
                params);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<
    ArchTag,
    FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>,
    void>
    : CopyL1ToL0A<
          ArchTag,
          FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>,
          FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A2>> {
};

} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_GEMM_TILE_ATLASA2_COPY_L1_TO_L0A_HPP
