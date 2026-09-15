/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_VEC_COPY_UB_TO_GM_PADDING_HPP
#define FTSELF_GEMV_TILE_VEC_COPY_UB_TO_GM_PADDING_HPP

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"

namespace FTSelf::Gemv::Tile {


template <
    class ArchTag,
    class GmType,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE PaddingType,
    bool is_atoadd = false
>
struct VecCopyUBToGmPadding
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>, "Unsupported copy UB to gm, can not find the specialization.");
};


template <class Element, FTSelf::Gemv::helper::VEC_PADDING_TYPE PaddingType, bool AtomicAdd>
struct VecCopyUBToGmPadding<FTSelf::Gemv::Arch::AtlasA2,
    FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>,
    PaddingType,
    AtomicAdd>
{
    using LayoutSrc = FTSelf::layout::VectorLayout;
    using LayoutDst = FTSelf::layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);

    FTSELF_DEVICE
    VecCopyUBToGmPadding() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        FTSelf::layout::VectorLayout const &layoutDst,
        FTSelf::layout::VectorLayout const &)
    {
        static_assert(PaddingType == FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING || !AtomicAdd,
            "Atomic add is only supported by the padding copy strategy.");
        if constexpr (AtomicAdd) {
            AscendC::SetAtomicAdd<Element>();
        }
        if constexpr (PaddingType == FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED) {
            AscendC::DataCopyParams params;
            params.blockCount = 1;
            params.blockLen = FTSelf::helper::CeilDiv(
                FTSelf::helper::GetShape(layoutDst, 0), ELE_NUM_PER_C0);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopy(dstTensor, srcTensor, params);
        } else if constexpr (PaddingType == FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING) {
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = FTSelf::helper::GetShape(layoutDst, 0) * sizeof(Element);
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            AscendC::DataCopyPad(dstTensor, srcTensor, params);
        } else {
            static_assert(FTSelf::helper::DEPENDENT_FALSE<Element>, "Unsupported UB to GM padding strategy.");
        }
        if constexpr (AtomicAdd) {
            AscendC::SetAtomicNone();
        }
    }
};

} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_GEMV_TILE_VEC_COPY_UB_TO_GM_HPP
