/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMM_HELPER_BLOCK_MMAD_ALIGNMENT_HELPER_HPP
#define FTSELF_GEMM_HELPER_BLOCK_MMAD_ALIGNMENT_HELPER_HPP

#include <cstdint>
#include <type_traits>
#include "../../core/layout.hpp"
#include "../../core/macros.hpp"

namespace FTSelf::Gemm::helper {

template <class>
inline constexpr bool ALWAYS_FALSE = false;

template <class Element, class Layout>
struct MatrixL1AlignTraits {
    static_assert(ALWAYS_FALSE<Layout>, "Unsupported matrix layout for L1 alignment");
};

template <class Element>
struct MatrixL1AlignTraits<Element, FTSelf::layout::RowMajor> {
    static constexpr uint32_t ELEMENTS_PER_C0 = 32 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = 16;
    static constexpr uint32_t K_ALIGNED = ELEMENTS_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELEMENTS_PER_C0;
};

template <class Element>
struct MatrixL1AlignTraits<Element, FTSelf::layout::ColumnMajor> {
    static constexpr uint32_t ELEMENTS_PER_C0 = 32 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = ELEMENTS_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELEMENTS_PER_C0;
    static constexpr uint32_t N_ALIGNED = 16;
};

template <class Element>
struct MatrixL1AlignTraits<Element, FTSelf::layout::PaddingRowMajor>
    : MatrixL1AlignTraits<Element, FTSelf::layout::RowMajor> {};

template <class Element>
struct MatrixL1AlignTraits<Element, FTSelf::layout::PaddingColumnMajor>
    : MatrixL1AlignTraits<Element, FTSelf::layout::ColumnMajor> {};

template <class Element>
struct MatrixL1AlignTraits<Element, FTSelf::layout::zN>
    : MatrixL1AlignTraits<Element, FTSelf::layout::RowMajor> {};

template <class Element>
struct MatrixL1AlignTraits<Element, FTSelf::layout::nZ>
    : MatrixL1AlignTraits<Element, FTSelf::layout::ColumnMajor> {};

template <bool EnableUnitFlag>
FTSELF_DEVICE constexpr uint8_t GetPingpongMmadUnitFlag(
    uint32_t kLoopIdx, uint32_t kTileCount, uint32_t mPartIdx, uint32_t mPartLoop,
    uint32_t kPartIdx, uint32_t kPartLoop, uint32_t nPartIdx, uint32_t nPartLoop)
{
    if constexpr (!EnableUnitFlag) {
        return 0b00;
    }
    return ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
            (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) ? 0b11 : 0b10;
}

} // namespace FTSelf::Gemm::helper

#endif // FTSELF_GEMM_HELPER_BLOCK_MMAD_ALIGNMENT_HELPER_HPP
