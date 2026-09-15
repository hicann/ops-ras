/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_HELPER_HPP
#define FTSELF_GEMV_HELPER_HPP

#include <type_traits>

#include "../core/gemm_type.hpp"
#include "../core/layout.hpp"
#include "../helper/arch_helper.hpp"
#include "../helper/layout_helper.hpp"
#include "../helper/type_helper.hpp"

#define FTSELF_BIND_BLOCK_REDUCE_ARGS(args, outputLayoutName) \
    const auto gmA = (args).gmInput; \
    const auto &layoutA = (args).inputLayout; \
    const auto gmZMin = (args).gmMinimum; \
    const auto gmZMax = (args).gmMaximum; \
    const auto &outputLayoutName = (args).outputLayout; \
    const auto &actualShape = (args).actualShape

namespace FTSelf::Gemv {

template <class ArchTag_, bool Async_ = false>
struct MmadBase {
    using ArchTag = ArchTag_;
    static constexpr bool ASYNC = Async_;
};

struct GemvAtlasA2 : MmadBase<FTSelf::Gemv::Arch::AtlasA2, false> {
    static constexpr uint32_t STAGES = 2;
};

template <class ElementInput, class ElementOutput, class InputLayout, class OutputLayout>
struct BlockReduceArgs {
    AscendC::GlobalTensor<ElementInput> gmInput;
    InputLayout const &inputLayout;
    AscendC::GlobalTensor<ElementOutput> gmMinimum;
    AscendC::GlobalTensor<ElementOutput> gmMaximum;
    OutputLayout const &outputLayout;
    GemvCoord const &actualShape;
    uint32_t samplingStride;
    uint32_t strideUnit;

    FTSELF_DEVICE
    BlockReduceArgs(AscendC::GlobalTensor<ElementInput> input, InputLayout const &inputLayoutIn,
        AscendC::GlobalTensor<ElementOutput> minimum, AscendC::GlobalTensor<ElementOutput> maximum,
        OutputLayout const &outputLayoutIn, GemvCoord const &actualShapeIn,
        uint32_t samplingStrideIn = 1U, uint32_t strideUnitIn = 1U)
        : gmInput(input), inputLayout(inputLayoutIn), gmMinimum(minimum), gmMaximum(maximum),
          outputLayout(outputLayoutIn), actualShape(actualShapeIn),
          samplingStride(samplingStrideIn), strideUnit(strideUnitIn)
    {}
};

struct TileActualShape {
    uint32_t m;
    uint32_t n;
    uint32_t x;
    uint32_t y;

    FTSELF_DEVICE void SetM(uint32_t value)
    {
        m = value;
        y = value;
    }

    FTSELF_DEVICE void SetN(uint32_t value)
    {
        n = value;
        x = value;
    }
};

template <bool EnableUnitFlag_ = false, bool EnableShuffleK_ = false>
struct MmadAtlasA2Preload : MmadBase<FTSelf::Gemv::Arch::AtlasA2, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = EnableUnitFlag_;
    static constexpr bool ENABLE_SHUFFLE_K = EnableShuffleK_;
};

} // namespace FTSelf::Gemv

namespace FTSelf::Gemv::helper {

template <AscendC::HardEvent Event, class EventId>
FTSELF_DEVICE inline void WaitEvent(EventId eventId)
{
    AscendC::WaitFlag<Event>((event_t)eventId);
}

template <AscendC::HardEvent Event, class EventId>
FTSELF_DEVICE inline void SetEvent(EventId eventId)
{
    AscendC::SetFlag<Event>((event_t)eventId);
}

FTSELF_DEVICE inline void VectorBarrier()
{
    AscendC::PipeBarrier<PIPE_V>();
}

template <class Element>
FTSELF_DEVICE inline void ZeroTensor(AscendC::LocalTensor<Element> tensor, uint32_t count)
{
    AscendC::Duplicate<Element>(tensor, static_cast<Element>(0), count);
}

template <class Element>
FTSELF_DEVICE inline void ZeroTensorRepeated(
    AscendC::LocalTensor<Element> tensor, uint64_t mask, uint8_t repeatTimes,
    uint8_t dstBlockStride = 1, uint8_t dstRepeatStride = 8)
{
    AscendC::Duplicate<Element>(
        tensor, static_cast<Element>(0), mask, repeatTimes, dstBlockStride, dstRepeatStride);
}

FTSELF_DEVICE inline void SetRepeatStrides(
    AscendC::BinaryRepeatParams &params, uint8_t dstStride, uint8_t src0Stride, uint8_t src1Stride)
{
    params.dstRepStride = dstStride;
    params.src0RepStride = src0Stride;
    params.src1RepStride = src1Stride;
}

FTSELF_DEVICE inline uint32_t ClampTail(uint32_t offset, uint32_t tail, uint32_t actual)
{
    return offset + tail > actual ? actual - offset : tail;
}

template <class Element>
FTSELF_DEVICE inline void InitializeExtrema(
    AscendC::LocalTensor<Element> minTensor,
    AscendC::LocalTensor<Element> maxTensor,
    uint32_t count)
{
    AscendC::Duplicate<Element>(minTensor, static_cast<Element>(1.0e10f), count);
    AscendC::Duplicate<Element>(maxTensor, static_cast<Element>(-1.0e10f), count);
}

template <class Element>
FTSELF_DEVICE inline void SetCounterMask(uint32_t count)
{
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<Element, AscendC::MaskMode::COUNTER>(count);
}

FTSELF_DEVICE inline void ResetCounterMask()
{
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
}

template<class Element>
struct UBAlignHelper
{
    static_assert(sizeof(Element) != 0 && sizeof(Element) <= FTSelf::Gemv::BYTE_PER_BLK,
        "Element size must produce a non-zero UB alignment");
    static constexpr uint32_t ALIGN = FTSelf::Gemv::BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t ALIGN_REMAIN = 8;
};

template<class Element>
struct UBTransposeAlignHelper
{
    static_assert(sizeof(Element) != 0 && sizeof(Element) <= FTSelf::Gemv::BYTE_PER_BLK,
        "Element size must produce a non-zero UB alignment");
    static constexpr uint32_t H_ALIGN = 16;
    static constexpr uint32_t W_ALIGN = 16;
    static constexpr uint32_t BLK_ALIGN = FTSelf::Gemv::BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t C_ALIGN = FTSelf::Gemv::BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t C_UNIT_UPPER_LIMIT = 24;
};

template<class GmAType>
struct AtomicAddSelector
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct AtomicAddSelector<FTSelf::GemmType<Element, FTSelf::layout::RowMajor>>
{
    static constexpr bool value = false;
};

template<class Element>
struct AtomicAddSelector<FTSelf::GemmType<Element, FTSelf::layout::ColumnMajor>>
{
    static constexpr bool value = true;
};

template <class Element, class Layout>
struct L1AlignHelper
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<Element>,
        "Unsupported align helper, can not find the specialization.");
};

template <class Element>
struct L1AlignHelper<Element, FTSelf::layout::RowMajor>
{
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = FTSelf::Gemv::C0_NUM_PER_FRACTAL;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, FTSelf::layout::ColumnMajor>
{
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);

    static constexpr uint32_t getNAligned()
    {
        if constexpr (std::is_same<Element, int8_t>::value) {
            return ELE_NUM_PER_C0 / sizeof(Element);
        } else {
            return FTSelf::Gemv::C0_NUM_PER_FRACTAL;
        }
    }

    static constexpr uint32_t getMAligned()
    {
        if constexpr (std::is_same<Element, int8_t>::value) {
            return ELE_NUM_PER_C0 / sizeof(Element);
        } else {
            return FTSelf::Gemv::C0_NUM_PER_FRACTAL;
        }
    }

    static constexpr uint32_t N_ALIGNED = getNAligned();
    static constexpr uint32_t M_ALIGNED = getMAligned();
};

template <class Element>
struct L1AlignHelper<Element, FTSelf::layout::VectorLayout>
{
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = FTSelf::Gemv::C0_NUM_PER_FRACTAL;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

////////////////////////////////
// new add  gemvaic selector
template<class GmAType, class GmBType>
struct L1AndL0TypeSelectorGemv{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
    static_assert(FTSelf::helper::DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};

enum class FT_ENC_TYPE {
    CE = 0,
    ETC,
    BOTHC,
    RCE,
    NO,
    BE,
    ETA,
    ABE,
    ETAB,
    BOTHAB
};

enum class FT_COMP_TYPE {
    SUB = 0,
    RSUB,
    XOR,
    COMPARE
};

enum class FT_RCE_THRE_TYPE {
    ROUND = 0,
    ROUND_WITH_ACC
};

enum class FT_PIPELINE_TYPE {
    SPLIT_K = 0,
    NO_SPLIT_K,
    NO_PIPELINE
};

enum class FT_L02L1_TYPE{
    FIX_PIPE = 0,
    CO12DST,
    ENHANCED
};

enum class FT_AIV_PIPE_FUSE_TYPE{
    NO_FUSED = 0,
    ABE_FUSED_THRE,
    A_B_MIXED,
    A_B_MIXED_BF,
    A_B_ROBUST,
    A_B_ROBUST_BF,
    A_B_SIMPLIFIED,
    A_B_SIMPLIFIED_BF,
    THRE_FUSED
};

enum class FT_THRESHOLD_ALGORITHM{
    AABFT = 0,
    ASVAR,
    ASVAR_ROBUST,
    ASVAR_SIMPLIFIED
};

enum class FT_REDUCE_TYPE{
    SUM = 0,
    MAX,
    SUM_MAX,
    MAX_MIN,
    SUM_MAX_MIXED,
    SUM_MAX_ABE,
    MEAN_SQUARE,
    VAR,
    VAR_SIMPLIFIED
};

enum class FT_AIC_BE_SCHEME{
    ROWCOMPLETE = 0,
    COLCOMPLETE,
    ROWCOMPLETE_BF
};

enum class FT_ABE_TYPE{
    TILING_BLOCK = 0,
    CENTRAL_BLOCK,
    WAIT_CENTRAL_BLOCK
};

enum class MATRIX_SIMPLING_TYPE{
    CONTINUOUS_SIMPLING,
    STRIDED_SIMPLING
};

enum class VEC_PADDING_TYPE{
    ALIGNED,
    PADDING
};

enum class VEC_ADD_TYPE{
    COUNT,
    MASK
};

template<class Element>
struct L1AndL0TypeSelectorGemv<FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>, FTSelf::GemmType<Element, FTSelf::layout::RowMajor>>{
    using L1AType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L1BType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L1BColType = FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A1>;

    using L0AType = FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A2>;
    using L0BType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::B2>;
    using L0BColType = FTSelf::GemmType<Element, FTSelf::layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemv<FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>, FTSelf::GemmType<Element, FTSelf::layout::zN>>{
    using L1AType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L1BType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L1BColType = FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A1>;

    using L0AType = FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A2>;
    using L0BType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::B2>;
    using L0BColType = FTSelf::GemmType<Element, FTSelf::layout::nZ, AscendC::TPosition::B2>;
};

template<class Element>
struct L1AndL0TypeSelectorGemv<FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>, FTSelf::GemmType<Element, FTSelf::layout::ColumnMajor>>{
    using L1AType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L1BType = FTSelf::GemmType<Element, FTSelf::layout::nN, AscendC::TPosition::A1>;
    using L1BColType = FTSelf::GemmType<Element, FTSelf::layout::nZ, AscendC::TPosition::A1>;

    using L0AType = FTSelf::GemmType<Element, FTSelf::layout::zZ, AscendC::TPosition::A2>;
    using L0BType = FTSelf::GemmType<Element, FTSelf::layout::zN, AscendC::TPosition::B2>;
    using L0BColType = FTSelf::GemmType<Element, FTSelf::layout::nZ, AscendC::TPosition::B2>;
};

template<>
struct L1AndL0TypeSelectorGemv<FTSelf::GemmType<int8_t, FTSelf::layout::VectorLayout>, FTSelf::GemmType<int8_t, FTSelf::layout::ColumnMajor>>{
    using L1AType = FTSelf::GemmType<int8_t, FTSelf::layout::zN, AscendC::TPosition::A1>;
    using L1BType = FTSelf::GemmType<int8_t, FTSelf::layout::nZ, AscendC::TPosition::B1>;
    using L1BColType = FTSelf::GemmType<int8_t, FTSelf::layout::nZ, AscendC::TPosition::A1>;

    using L0AType = FTSelf::GemmType<int8_t, FTSelf::layout::zZ, AscendC::TPosition::A2>;
    using L0BType = FTSelf::GemmType<int8_t, FTSelf::layout::zN, AscendC::TPosition::B2>;
    using L0BColType = FTSelf::GemmType<int8_t, FTSelf::layout::nZ, AscendC::TPosition::B2>;
};

} // namespace FTSelf::Gemv::helper

#endif // FTSELF_GEMV_HELPER_HPP
