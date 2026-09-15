/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_FTSELF_GEMM_TILE_TILE_COPY_HPP
#define FTSELF_FTSELF_GEMM_TILE_TILE_COPY_HPP

#include <type_traits>

#include "../../core/gemm_type.hpp"
#include "../../core/layout.hpp"
#include "../../gemm/tile/atlasa2/copy_gm_to_l1.hpp"
#include "../../gemm/tile/atlasa2/copy_l1_to_l0a.hpp"
#include "../../gemm/tile/atlasa2/copy_l1_to_l0b.hpp"
#include "../../gemm/tile/copy_l0c_to_gm.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemm::Tile {

namespace detail {

template <class...>
inline constexpr bool TILE_COPY_UNSUPPORTED = false;

template <class GmType>
struct L1L0ATypeSelector {
    static_assert(TILE_COPY_UNSUPPORTED<GmType>,
        "Unsupported FTSelf GM/L1/L0A type path");
};

template <class Element>
struct L1L0ATypeSelector<FTSelf::GemmType<Element, FTSelf::layout::RowMajor>> {
    using L1Type =
        FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L0Type =
        FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A2>;
};

template <class GmType>
struct L1L0BTypeSelector {
    static_assert(TILE_COPY_UNSUPPORTED<GmType>,
        "Unsupported FTSelf GM/L1/L0B type path");
};

template <class Element>
struct L1L0BTypeSelector<FTSelf::GemmType<Element, FTSelf::layout::RowMajor>> {
    using L1Type =
        FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L0Type =
        FTSelf::GemmType<Element, FTSelf::layout::nZ, AscendC::TPosition::B2>;
};

// The fault-tolerance A operand is intentionally routed through L0B.
template <class GmType>
struct L1L0FaultATypeSelector {
    static_assert(TILE_COPY_UNSUPPORTED<GmType>,
        "Unsupported FTSelf fault-tolerance GM/L1/L0B type path");
};

template <class Element>
struct L1L0FaultATypeSelector<
    FTSelf::GemmType<Element, FTSelf::layout::RowMajor>> {
    using L1Type =
        FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L0Type =
        FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::B2>;
};

} // namespace detail

template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class BiasType = void>
struct TileCopy {
    static_assert(detail::TILE_COPY_UNSUPPORTED<
                      ArchTag, AType, BType, CType, BiasType>,
        "Unsupported FTSelf TileCopy combination; only the production "
        "RowMajor, bias-free path is implemented");
};

template <
    class ArchTag,
    class ElementA,
    class ElementB,
    class ElementC>
struct TileCopy<
    ArchTag,
    FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
    FTSelf::GemmType<ElementB, FTSelf::layout::RowMajor>,
    FTSelf::GemmType<ElementC, FTSelf::layout::RowMajor>,
    void> {
    using AType = FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>;
    using BType = FTSelf::GemmType<ElementB, FTSelf::layout::RowMajor>;
    using CType = FTSelf::GemmType<ElementC, FTSelf::layout::RowMajor>;
    using ElementAccumulator =
        typename FTSelf::core::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using APath = detail::L1L0ATypeSelector<AType>;
    using BPath = detail::L1L0BTypeSelector<BType>;
    using L1AType = typename APath::L1Type;
    using L1BType = typename BPath::L1Type;
    using L0AType = typename APath::L0Type;
    using L0BType = typename BPath::L0Type;

    using CopyGmToL1A = CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1B = CopyGmToL1<ArchTag, BType, L1BType>;
    using CopyL1ToL0A = Tile::CopyL1ToL0A<ArchTag, L1AType, L0AType>;
    using CopyL1ToL0B = Tile::CopyL1ToL0B<ArchTag, L1BType, L0BType>;
    using CopyL0CToGm = Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using CopyGmToL1Bias = void;
    using CopyL1ToBT = void;
};

template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void,
    FTSelf::Gemv::helper::FT_L02L1_TYPE COPY_TYPE =
        FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE>
struct TileCopyFT {
    static_assert(detail::TILE_COPY_UNSUPPORTED<
                      ArchTag, AType, BType, CType, XType, YType, BiasType>,
        "TileCopyFT is not part of the migrated production paths");
};

template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class XType,
    class YType,
    class BiasType = void>
struct TileCopyFTABonAic {
    static_assert(detail::TILE_COPY_UNSUPPORTED<
                      ArchTag, AType, BType, CType, XType, YType, BiasType>,
        "Unsupported FTSelf TileCopyFTABonAic combination; only the "
        "production RowMajor, bias-free path is implemented");
};

template <
    class ArchTag,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementX,
    class ElementY>
struct TileCopyFTABonAic<
    ArchTag,
    FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
    FTSelf::GemmType<ElementB, FTSelf::layout::RowMajor>,
    FTSelf::GemmType<ElementC, FTSelf::layout::RowMajor>,
    FTSelf::GemmType<ElementX, FTSelf::layout::RowMajor>,
    FTSelf::GemmType<ElementY, FTSelf::layout::RowMajor>,
    void> {
    static_assert(std::is_same_v<ElementA, ElementX>,
        "The production fault-tolerance path requires identical A and X element types");

    using AType = FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>;
    using BType = FTSelf::GemmType<ElementB, FTSelf::layout::RowMajor>;
    using CType = FTSelf::GemmType<ElementC, FTSelf::layout::RowMajor>;
    using XType = FTSelf::GemmType<ElementX, FTSelf::layout::RowMajor>;
    using YType = FTSelf::GemmType<ElementY, FTSelf::layout::RowMajor>;
    using ElementAccumulator =
        typename FTSelf::core::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using VXType = FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>;
    using XPath = detail::L1L0ATypeSelector<XType>;
    using FaultAPath = detail::L1L0FaultATypeSelector<AType>;
    using BPath = detail::L1L0BTypeSelector<BType>;
    using L1XType = typename XPath::L1Type;
    using L0XType = typename XPath::L0Type;
    using L1AType = typename FaultAPath::L1Type;
    using L0ATypeforFT = typename FaultAPath::L0Type;
    using L1BType = typename BPath::L1Type;
    using L0BType = typename BPath::L0Type;

    using CopyGmToL1A = CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1B = CopyGmToL1<ArchTag, BType, L1BType>;
    using CopyGmToL1VX = CopyGmToL1<ArchTag, VXType, L1XType>;
    using CopyGmToL1X = CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyL1ToL0A =
        Tile::CopyL1ToL0A<ArchTag, L1AType, typename detail::L1L0ATypeSelector<AType>::L0Type>;
    using CopyL1ToL0B =
        Tile::CopyL1ToL0B<ArchTag, L1BType, L0BType>;
    using CopyL1ToL0X = Tile::CopyL1ToL0A<ArchTag, L1XType, L0XType>;
    using CopyL1ToL0AforFT = Tile::CopyL1ToL0B<ArchTag, L1AType, L0ATypeforFT>;
    using CopyL0CToGm = Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using CopyL0CToGmforABE = Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
    using CopyGmToL1Bias = void;
    using CopyL1ToBT = void;
};

template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class XType,
    class XTypeCol,
    class YType,
    class BiasType = void>
struct TileCopyFTABonAicAuged {
    static_assert(detail::TILE_COPY_UNSUPPORTED<
                      ArchTag, AType, BType, CType, XType, XTypeCol, YType, BiasType>,
        "TileCopyFTABonAicAuged is not part of the migrated production paths");
};

} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_FTSELF_GEMM_TILE_TILE_COPY_HPP
