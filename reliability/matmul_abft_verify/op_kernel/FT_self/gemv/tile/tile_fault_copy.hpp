/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_FAULT_COPY_HPP_SELF
#define FTSELF_GEMV_TILE_TILE_FAULT_COPY_HPP_SELF

#include "../../gemv/tile/gemv_tile_copy.hpp"
#include "../../gemm/tile/atlasa2/copy_l1_to_l0b.hpp"

namespace FTSelf::Gemv::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileFaultCopyGemvAiv {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    // the function of aiv
    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileFaultCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aic
    using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L0AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;

    using CopyGmToL1A = copy_detail::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1B = copy_detail::CopyGmToL1<ArchTag, AType, L1AType>;

    using CopyL1ToL0A = copy_detail::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL0CToGm = copy_detail::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
};

}

#endif
