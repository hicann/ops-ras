/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_FTSELF_GEMM_TILE_COPY_L0C_TO_GM_HPP
#define FTSELF_FTSELF_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include <cstdint>
#include <type_traits>

#include "../../core/arch.hpp"
#include "../../core/coord.hpp"
#include "../../core/gemm_type.hpp"
#include "../../core/layout.hpp"
#include "../../core/macros.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemm::Tile {

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <class>
inline constexpr bool COPY_L0C_UNSUPPORTED = false;

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
    static_assert(COPY_L0C_UNSUPPORTED<ArchTag>,
        "Unsupported FTSelf L0C quantization mode");
};

template <>
struct CopyL0CToGmQuantMode<
    FTSelf::Arch::AtlasA2,
    float,
    float,
    ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false>
struct CopyL0CToGm {
    static_assert(COPY_L0C_UNSUPPORTED<ArchTag>,
        "Unsupported FTSelf L0C to GM copy specialization");
};

template <class ElementAccumulator, class ElementDst, bool ReluEnable>
struct CopyL0CToGm<
    FTSelf::Arch::AtlasA2,
    ElementAccumulator,
    FTSelf::GemmType<ElementDst, FTSelf::layout::RowMajor>,
    ScaleGranularity::NO_QUANT,
    ReluEnable> {
    using ArchTag = FTSelf::Arch::AtlasA2;
    using ElementSrc = ElementAccumulator;
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::zN;

    static constexpr auto QUANT_MODE =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst>::VALUE;

    FTSELF_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout,
        LayoutSrc const &srcLayout,
        uint8_t unitFlag = 0) const
    {
        AscendC::FixpipeParamsV220 params;
        params.nSize = dstLayout.shape(1);
        params.mSize = dstLayout.shape(0);
        params.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        params.dstStride = dstLayout.stride(0);
        params.quantPre = QUANT_MODE;
        params.reluEn = ReluEnable;
        params.unitFlag = unitFlag;
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, params);
    }
};

template <
    class ArchTag,
    class ElementAccumulator,
    class L1Type,
    FTSelf::Gemv::helper::FT_L02L1_TYPE CopyType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToL1 {
    static_assert(COPY_L0C_UNSUPPORTED<ArchTag>,
        "Unsupported FTSelf L0C to L1 copy specialization");
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToL1Base {
    using ArchTag = FTSelf::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = FTSelf::layout::zN;
    using LayoutDst = FTSelf::layout::zN;

    static constexpr auto QUANT_PRE = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto RELU_EN = ReluEnable_;

    template <class Params>
    FTSELF_DEVICE
    static void SetLayoutParams(Params &params, LayoutDst const &dstLayout,
        LayoutSrc const &srcLayout)
    {
        params.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        params.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        params.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        params.dstStride = dstLayout.stride(3) / (FTSelf::BYTE_PER_C0 / sizeof(ElementDst));
    }
};


template <
    class ElementAccumulator_,
    class ElementDst_,
    FTSelf::Gemv::helper::FT_L02L1_TYPE CopyType_,
    bool ReluEnable_>
struct CopyL0CToL1<FTSelf::Arch::AtlasA2,
                   ElementAccumulator_,
                   FTSelf::GemmType<ElementDst_, FTSelf::layout::zN>,
                   CopyType_,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
    : CopyL0CToL1Base<ElementAccumulator_, ElementDst_, ReluEnable_>
{
    static_assert(CopyType_ == FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE ||
            CopyType_ == FTSelf::Gemv::helper::FT_L02L1_TYPE::CO12DST,
        "Unsupported FTSelf L0C to L1 copy type");

    using Base = CopyL0CToL1Base<ElementAccumulator_, ElementDst_, ReluEnable_>;
    using ElementDst = typename Base::ElementDst;
    using ElementSrc = typename Base::ElementSrc;
    using LayoutSrc = typename Base::LayoutSrc;
    using LayoutDst = typename Base::LayoutDst;

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<ElementDst> const &dst,
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout,
        uint8_t unitFlag = 0, bool isChannelSplit = false,
        uint8_t reluPre = 0, float scalarForDeq = 1.0)
    {
        if constexpr (CopyType_ == FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE) {
            AscendC::FixpipeParamsV220 params;
            Base::SetLayoutParams(params, dstLayout, srcLayout);
            params.quantPre = Base::QUANT_PRE;
            params.reluEn = Base::RELU_EN;
            params.unitFlag = unitFlag;
            params.isChannelSplit = isChannelSplit;
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, params);
        } else {
            AscendC::DataCopyCO12DstParams params;
            Base::SetLayoutParams(params, dstLayout, srcLayout);
            params.quantPre = Base::QUANT_PRE;
            params.reluPre = 0;
            params.channelSplit = isChannelSplit;
            params.nz2ndEn = false;
            static_cast<void>(unitFlag);
            static_cast<void>(reluPre);
            static_cast<void>(scalarForDeq);
            AscendC::Fixpipe<ElementDst, ElementSrc>(dst, src, params);
        }
    }
};

} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_FTSELF_GEMM_TILE_COPY_L0C_TO_GM_HPP
