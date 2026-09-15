/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_VEC_COPY_GM_TO_UB_PADDING_HPP
#define FTSELF_GEMV_TILE_VEC_COPY_GM_TO_UB_PADDING_HPP

#include "../../helper/math_helper.hpp"
#include "../../helper/base_helper.hpp"

// constexpr uint32_t FTSelf::Gemv::STRIDE_LIMIT = 65536;

namespace FTSelf::Gemv::Tile {


template <
    class ArchTag,
    class VType,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE PaddingType
>
struct VecCopyGmToUBPadding
{
    using Element = typename VType::Element;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);

    FTSELF_DEVICE
    VecCopyGmToUBPadding() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        uint32_t len)
    {
        if constexpr (PaddingType == FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED) {
            AscendC::DataCopyParams params;
            params.blockCount = 1;
            params.blockLen = FTSelf::helper::CeilDiv(len, ELE_NUM_PER_C0);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopy(dstTensor, srcTensor, params);
        } else if constexpr (PaddingType == FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING) {
            AscendC::DataCopyExtParams dataCopyParams(1, len * sizeof(Element), 0, 0, 0);
            AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
        } else {
            static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>,
                "Unsupported copy from GM to UB padding strategy.");
        }
    }
};


} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
