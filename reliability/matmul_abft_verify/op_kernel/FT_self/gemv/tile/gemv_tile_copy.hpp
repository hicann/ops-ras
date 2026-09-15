/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_FTSELF_GEMV_TILE_TILE_COPY_HPP
#define FTSELF_FTSELF_GEMV_TILE_TILE_COPY_HPP

#include "../../gemv/helper.hpp"
#include "../../gemv/tile/matrix_copy_gm_to_ub_simpling.hpp"
#include "../../gemv/tile/matrix_copy_ub_to_gm.hpp"
#include "../../gemv/tile/vec_copy_gm_to_ub_pad.hpp"
#include "../../gemv/tile/vec_copy_ub_to_gm_pad.hpp"

namespace FTSelf::Gemv::Tile {

namespace copy_detail {

template <class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    using ElementAccumulator = std::conditional_t<
        std::is_same_v<ElementA, int8_t> || std::is_same_v<ElementA, uint8_t>, int32_t,
        std::conditional_t<std::is_same_v<ElementA, half> || std::is_same_v<ElementA, bfloat16_t>,
                           float, ElementA>>;
};

template <class ArchTag, class VType>
struct VecCopyGmToUB {
    using Element = typename VType::Element;
    using LayoutSrc = typename VType::Layout;
    using LayoutDst = typename VType::Layout;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);

    FTSELF_GEMV_HOST_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor,
                    AscendC::GlobalTensor<Element> srcTensor, uint32_t len) const {
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = FTSelf::helper::CeilDiv(len, ELE_NUM_PER_C0);
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

template <class ArchTag, class GmType, bool AtomicAdd = false>
struct VecCopyUBToGm {
    using Element = typename GmType::Element;
    using LayoutSrc = typename GmType::Layout;
    using LayoutDst = typename GmType::Layout;

    template <class LayoutDst, class LayoutSrc>
    FTSELF_GEMV_HOST_DEVICE
    void operator()(AscendC::GlobalTensor<Element> dstTensor,
                    AscendC::LocalTensor<Element> srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &) const {
        if constexpr (AtomicAdd) {
            AscendC::SetAtomicAdd<Element>();
        }
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = layoutDst.shape(0) * sizeof(Element);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        AscendC::DataCopyPad(dstTensor, srcTensor, params);
        if constexpr (AtomicAdd) {
            AscendC::SetAtomicNone();
        }
    }
};

template <class ArchTag, class GmType>
struct MatrixCopyGmToUB {
    using Element = typename GmType::Element;
    using LayoutSrc = typename GmType::Layout;
    using LayoutDst = typename GmType::Layout;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);

    template <class LayoutDst, class LayoutSrc>
    FTSELF_GEMV_HOST_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor,
                    AscendC::GlobalTensor<Element> srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc) const {
        const bool columnMajor = layoutSrc.stride(0) == 1;
        const uint32_t innerActual = columnMajor ? layoutSrc.shape(0) : layoutSrc.shape(1);
        const uint32_t outerActual = columnMajor ? layoutSrc.shape(1) : layoutSrc.shape(0);
        const uint32_t innerRound = columnMajor ? layoutDst.shape(0) : layoutDst.shape(1);
        const uint32_t stride = columnMajor ? layoutSrc.stride(1) : layoutSrc.stride(0);
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = FTSelf::helper::CeilDiv(innerActual, ELE_NUM_PER_C0);
        params.srcStride = 0;
        params.dstStride = 0;
        for (uint32_t i = 0; i < outerActual; ++i) {
            AscendC::DataCopy(dstTensor[i * innerRound], srcTensor[i * stride], params);
        }
    }
};

template <class ArchTag, class GmType, class L1Type>
struct CopyGmToL1 {
    using Element = typename GmType::Element;
    using GmLayout = typename GmType::Layout;
    using LayoutSrc = typename GmType::Layout;
    using LayoutDst = typename L1Type::Layout;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);

    template <class LayoutDst, class LayoutSrc>
    FTSELF_GEMV_HOST_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor,
                    AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc) const {
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.srcNdMatrixStride = 0;
        params.dstNzMatrixStride = 0;
        if constexpr (GmLayout::RANK == 1) {
            params.dValue = layoutSrc.shape(0);
            params.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
            params.nValue = 1;
            params.srcDValue = layoutSrc.shape(0);
            params.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        } else {
            params.dValue = layoutSrc.shape(1);
            params.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
            params.nValue = layoutSrc.shape(0);
            params.srcDValue = layoutSrc.stride(0);
            params.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        }
        AscendC::DataCopy(dstTensor, srcTensor, params);
    }
};

template <class ArchTag, class L1Type, class L0Type>
struct CopyL1ToL0A {
    using Element = typename L1Type::Element;
    using LayoutSrc = typename L1Type::Layout;
    using LayoutDst = typename L0Type::Layout;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = FTSelf::Gemv::BYTE_PER_FRACTAL / sizeof(Element);

    template <class LayoutDst, class LayoutSrc>
    FTSELF_GEMV_HOST_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor,
                    AscendC::LocalTensor<Element> srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc) const {
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        params.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        params.sid = 0;
        params.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        params.ifTranspose = false;
        params.addrMode = 0;
        for (uint32_t i = 0; i < layoutDst.shape(1); ++i) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)],
                              srcTensor[i * layoutSrc.stride(1)], params);
        }
    }
};

template <class ElementSrc, class ElementDst>
struct CopyL0CToGmQuantMode {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

template <>
struct CopyL0CToGmQuantMode<float, half> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <>
struct CopyL0CToGmQuantMode<float, bfloat16_t> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <class ArchTag, class ElementAccumulator, class GmType>
struct CopyL0CToGm {
    using ElementDst = typename GmType::Element;
    using LayoutDst = typename GmType::Layout;

    template <class LayoutDst, class LayoutSrc>
    FTSELF_GEMV_HOST_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
                    AscendC::LocalTensor<ElementAccumulator> const &src,
                    LayoutDst const &dstLayout, LayoutSrc const &srcLayout,
                    uint8_t unitFlag = 0) const {
        AscendC::FixpipeParamsV220 params;
        params.nSize = dstLayout.shape(1);
        params.mSize = dstLayout.shape(0);
        params.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        params.dstStride = dstLayout.stride(0);
        params.quantPre = CopyL0CToGmQuantMode<ElementAccumulator, ElementDst>::VALUE;
        params.reluEn = false;
        params.unitFlag = unitFlag;
        AscendC::Fixpipe<ElementDst, ElementAccumulator, AscendC::CFG_ROW_MAJOR>(dst, src, params);
    }
};

} // namespace copy_detail

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
struct TileCopyGemvAiv {
    using MATRIX_SIMPLING_TYPE = FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE;

    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbSimplingContinue = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::STRIDED_SIMPLING>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    class BType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyFTRedAiv {
    using MATRIX_SIMPLING_TYPE = FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // the function of aiv
    // using ElementCInL1 = typename XType::Element;
    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, XType>;

    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;

    using LayoutVX = FTSelf::layout::VectorLayout;
    using VXType = FTSelf::Gemv::GemmType<ElementX, LayoutVX>;
    using VecCopyUbToGmforBMax = copy_detail::VecCopyUBToGm<ArchTag, XType, is_atoadd>;
    using VecCopyUbToGmforAMax = copy_detail::VecCopyUBToGm<ArchTag, VXType, is_atoadd>;
    using VecCopyUbToGmforAMean = copy_detail::VecCopyUBToGm<ArchTag, YType, is_atoadd>;

    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;


    using MatrixCopyGmToUbSimplingContinue = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>;

    using MatrixCopyGmToUbSimplingStrided = FTSelf::Gemv::Tile::MatrixCopyGmToUBSimpling<
        ArchTag, AType, MATRIX_SIMPLING_TYPE::STRIDED_SIMPLING>;
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
struct TileCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using ElementYforBFAIV = float;
    using LayoutY = typename YType::Layout;
    using YTypeforBFAIV = FTSelf::Gemv::GemmType<ElementYforBFAIV, LayoutY>;

    // the function of aic
    using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L1AColType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BColType;

    using L0AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;
    using L0BColType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BColType;


    using CopyGmToL1A = copy_detail::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1B = copy_detail::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1BCol = copy_detail::CopyGmToL1<ArchTag, AType, L1AColType>;


    using CopyL1ToL0A = copy_detail::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL1ToL0BCol = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AColType, L0BColType>;
    using CopyL1ToL0BBoth = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BColType>;


    using CopyL0CToGm = copy_detail::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
    using CopyL0CToGmforBFAIV = copy_detail::CopyL0CToGm<ArchTag, ElementAccumulator, YTypeforBFAIV>;


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
struct TileCopyMatrixReduceAiv {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, XType>;
    // FTSelf::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    /// Tag indicating architecture
    class ArchTag,
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyCompareAiv {
    using ElementZ = typename ZType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = typename std::conditional<
        (COMP_TYPE_ == FT_COMP_TYPE::XOR),
        uint16_t,
        typename std::conditional<(COMP_TYPE_ == FT_COMP_TYPE::COMPARE), int32_t, ElementX>::type>::type;


    // using ElementAccumulator =
    //     typename copy_detail::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv

    using VecCopyGmToUbX = copy_detail::VecCopyGmToUB<ArchTag, XType>;
    using VecCopyGmToUbY = copy_detail::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = copy_detail::VecCopyGmToUB<ArchTag, XType>;

    using VecCopyUbToGmZ = copy_detail::VecCopyUBToGm<ArchTag, ZType>;

    using WType = FTSelf::Gemv::GemmType<ElementWork, FTSelf::layout::VectorLayout>;
    using VecCopyUbToGmW = copy_detail::VecCopyUBToGm<ArchTag, WType>;



    // static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    // using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    // using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;
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
    /// Output Result operand namely vector Z
    class ZType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixReduceAivFused {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementZ = typename ZType::Element;

    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, XType>;
    // FTSelf::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;

    using VecCopyGmToUbInX = copy_detail::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbInY = copy_detail::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = copy_detail::VecCopyGmToUB<ArchTag, YType>;

    using VecCopyUbToGmZ = copy_detail::VecCopyUBToGm<ArchTag, ZType>;

    using WType = FTSelf::Gemv::GemmType<ElementWork, FTSelf::layout::VectorLayout>;
    using VecCopyUbToGmW = copy_detail::VecCopyUBToGm<ArchTag, WType>;
};

template <class ArchTag, class AType, class CType, class XType, class YType, class ZType,
    class BiasType = void>
struct TileCopyGemvThreCompFusedAivBase {
    using ElementA = typename AType::Element;
    using ElementC = typename CType::Element;

    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;

    using ElementWork = ElementY;

    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;

    using MatrixCopyGmToUbforThre = copy_detail::MatrixCopyGmToUB<ArchTag, CType>;

    using VecCopyGmToUbInY = copy_detail::VecCopyGmToUB<ArchTag, YType>;
    using VecCopyGmToUbW = copy_detail::VecCopyGmToUB<ArchTag, YType>;

    using VecCopyUbToGmZ = copy_detail::VecCopyUBToGm<ArchTag, ZType>;
    using VecCopyUbToGmforThre = copy_detail::VecCopyUBToGm<ArchTag, YType>;
    using WType = FTSelf::Gemv::GemmType<ElementWork, FTSelf::layout::VectorLayout>;
    using VecCopyUbToGmW = copy_detail::VecCopyUBToGm<ArchTag, WType>;
    using ElementFIID = int32_t;
    using ElementFIData = ElementA;

    using FIIndexType = FTSelf::Gemv::GemmType<ElementFIID, FTSelf::layout::VectorLayout>;
    using VecCopyUbToGmforFIIndex = copy_detail::VecCopyUBToGm<ArchTag, FIIndexType, false>;

    using FIDataType = FTSelf::Gemv::GemmType<ElementFIData, FTSelf::layout::VectorLayout>;
    using VecCopyUbToGmforFIData = copy_detail::VecCopyUBToGm<ArchTag, FIDataType, false>;
};

template <class... Args>
struct TileCopyGemvThreCompFusedAiv : TileCopyGemvThreCompFusedAivBase<Args...> {};

template <class... Args>
struct TileCopyGemvThreCompFusedAivFI : TileCopyGemvThreCompFusedAivBase<Args...> {};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixSliceSumAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, YType>;
    // FTSelf::Gemv::helper::AtomicAddSelector<AType>::value
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for Y matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixAddAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;

    using LayoutVX = FTSelf::layout::VectorLayout;
    using VXType = FTSelf::Gemv::GemmType<ElementA, LayoutVX>;

    using LayoutVY = FTSelf::layout::VectorLayout;
    using VYType = FTSelf::Gemv::GemmType<ElementY, LayoutVY>;

    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using VecCopyGmToUb = copy_detail::VecCopyGmToUB<ArchTag, VXType>;
    static constexpr bool is_atoadd = false;
    using VecCopyUbToGm = copy_detail::VecCopyUBToGm<ArchTag, VYType, is_atoadd>;

    // the function of aiv
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;
    // MatrixCopyUbtoGm
    using MatrixCopyUbToGm = FTSelf::Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType for Y matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyMatrixAddVectorizedAiv {
    using ElementA = typename AType::Element;
    using ElementY = typename YType::Element;

    using LayoutVX = FTSelf::layout::VectorLayout;
    using VXType = FTSelf::Gemv::GemmType<ElementA, LayoutVX>;

    using LayoutVY = FTSelf::layout::VectorLayout;
    using VYType = FTSelf::Gemv::GemmType<ElementY, LayoutVY>;

    using ElementAccumulator =
        typename copy_detail::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;


    using VecCopyGmToUbCommon = FTSelf::Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED>;

    using VecCopyGmToUbTail = FTSelf::Gemv::Tile::VecCopyGmToUBPadding<ArchTag, VXType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING>;
    static constexpr bool is_atoadd = false;

    using VecCopyUbToGmCommon = FTSelf::Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED, is_atoadd>;


    using VecCopyUbToGmTail = FTSelf::Gemv::Tile::VecCopyUBToGmPadding<ArchTag, VYType, FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING, is_atoadd>;

    // the function of aiv
    using MatrixCopyGmToUb = copy_detail::MatrixCopyGmToUB<ArchTag, AType>;
    using MatrixCopyUbToGm = FTSelf::Gemv::Tile::MatrixCopyUBToGm<ArchTag, YType>;
};

} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_FTSELF_GEMV_TILE_TILE_COPY_HPP
