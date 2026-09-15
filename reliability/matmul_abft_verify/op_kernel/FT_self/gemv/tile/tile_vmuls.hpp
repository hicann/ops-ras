/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_VMULS_HPP
#define FTSELF_GEMV_TILE_TILE_VMULS_HPP

#include "../../core/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

template <class ArchTag, class VType>
struct TileVmuls {
    using Element = typename VType::Element;

    FTSELF_DEVICE TileVmuls() = default;

    FTSELF_DEVICE void operator()(AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor, Element scalar, uint32_t len)
    {
        FTSelf::Gemv::helper::SetCounterMask<Element>(len);
        AscendC::Muls<Element, false>(dstTensor, srcTensor, scalar,
            AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{});
        FTSelf::Gemv::helper::ResetCounterMask();
    }
};

} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_GEMV_TILE_TILE_VMULS_HPP
