/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_REDUCE_MEAN_VAR_STD_FUSED_HPP_BAK
#define FTSELF_GEMV_TILE_TILE_REDUCE_MEAN_VAR_STD_FUSED_HPP_BAK

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../helper/type_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

template <class ElementY>
FTSELF_DEVICE inline void InitializeReduceWorkspace(
    AscendC::LocalTensor<ElementY> workspace, uint32_t repeatSize, uint32_t rowCount)
{
    AscendC::Duplicate<ElementY>(workspace, static_cast<ElementY>(0.0), repeatSize,
        FTSelf::helper::CeilDiv(rowCount * repeatSize, repeatSize), 1, 8);
    FTSelf::Gemv::helper::VectorBarrier();
}

template <class ElementA>
FTSELF_DEVICE inline void ScaleMeanTensor(
    AscendC::LocalTensor<ElementA> tensor, ElementA factor, uint64_t mask,
    uint32_t rowCount, AscendC::UnaryRepeatParams const &params)
{
    AscendC::Muls<ElementA, true>(tensor, tensor, factor, mask, rowCount, params);
}

FTSELF_DEVICE inline uint32_t PrepareReduceTail(
    uint32_t repeatNum, uint32_t repeatSize, uint32_t actualSize, uint32_t &remain)
{
    uint32_t offset = repeatNum * repeatSize;
    if (offset + repeatSize > actualSize) {
        remain = actualSize - offset;
    }
    return offset;
}

#define FTSELF_TILE_REDUCE_MEAN_VAR_SHARED_BODY_1 \
{ \
        FTSELF_INIT_DUAL_REDUCE_GEOMETRY(layoutDst, layoutSrc, ELE_NUM_PER_C0, DST_ELE_NUM_PER_C0); \
        uint64_t abs_mask = dst_repeat_size; \
        uint64_t square_mask = dst_repeat_size; \
        uint64_t add_mask = dst_repeat_size; \
        auto abs_params = FTSelf::Gemv::MakeUnaryRepeatParams( \
            1, 1, FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0); \
        AscendC::BinaryRepeatParams square_params; \
        square_params.dstBlkStride = 1; \
        square_params.src0BlkStride = 1; \
        square_params.src1BlkStride = 1; \
        square_params.dstRepStride = FTSelf::helper::RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0; \
        square_params.src0RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        square_params.src1RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        AscendC::BinaryRepeatParams add_params; \
        add_params.dstBlkStride = 1; \
        add_params.src0BlkStride = 1; \
        add_params.src1BlkStride = 1; \
        add_params.dstRepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        add_params.src0RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        add_params.src1RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        InitializeReduceWorkspace(red_workspace, dst_repeat_size, m_actual); \
        for(uint32_t i=0; i < dst_repeat_num; i++){ \
            uint32_t offset = i * dst_repeat_size; \
            ScaleMeanTensor(srcMeanTensor[offset], n_ratio_factor, abs_mask, m_actual, abs_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::Abs<ElementA, true>( \
                srcMeanTensor[offset], \
                srcMeanTensor[offset], \
                abs_mask, \
                m_actual, \
                abs_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::MulAddDst<ElementY, ElementA, true>( \
                red_workspace, \
                srcMeanTensor[offset], \
                srcMeanTensor[offset], \
                square_mask, m_actual, \
                square_params); \
            if(i > 0){ \
                AscendC::Add<ElementA, true>( \
                    srcMeanTensor, \
                    srcMeanTensor[offset], \
                    srcMeanTensor, \
                    add_mask, \
                    m_actual, \
                    add_params); \
            } \
            FTSelf::Gemv::helper::VectorBarrier(); \
        } \
        if (dst_remain > 0) \
        { \
            uint32_t offset = PrepareReduceTail(dst_repeat_num, dst_repeat_size, n_actual, dst_remain); \
            uint64_t remain_mask = dst_remain; \
            ScaleMeanTensor(srcMeanTensor[offset], n_ratio_factor, remain_mask, m_actual, abs_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::Abs<ElementA, true>( \
                srcMeanTensor[offset], \
                srcMeanTensor[offset], \
                remain_mask, \
                m_actual, \
                abs_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::MulAddDst<ElementY, ElementA, true>( \
                red_workspace, \
                srcMeanTensor[offset], \
                srcMeanTensor[offset], \
                remain_mask, \
                m_actual, \
                square_params); \
            AscendC::Add<ElementA, true>( \
                srcMeanTensor, \
                srcMeanTensor[offset], \
                srcMeanTensor, \
                remain_mask, \
                m_actual, \
                add_params); \
        } \
        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size; \
        FTSelf::Gemv::helper::VectorBarrier(); \
        AscendC::WholeReduceSum<ElementA, true>( \
            srcMeanTensor, \
            srcMeanTensor, \
            reduce_mask, \
            m_actual, \
            1, \
            1, \
            FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0 \
        ); \
        FTSelf::Gemv::WholeReduceSumInPlace<ElementY>( \
            red_workspace, reduce_mask, m_actual, 8); \
        FTSelf::Gemv::helper::VectorBarrier(); \
        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size; \
        uint32_t dstOffset = m_round; \
        add_params.dstRepStride = 8; \
        add_params.src0RepStride = 8; \
        add_params.src1RepStride = 8; \
        square_params.dstRepStride = 8; \
        square_params.src0RepStride = 8; \
        square_params.src1RepStride = 8; \
        if constexpr (std::is_same_v<ElementA, ElementY>) { \
            AscendC::Add<ElementY, true>( \
                dstTensorMeanAbs, \
                srcMeanTensor, \
                dstTensorMeanAbs, \
                final_add_mask, \
                FTSelf::helper::CeilDiv(m_round, dst_repeat_size), \
                add_params); \
        } else { \
            AscendC::Cast<ElementY, ElementA>( \
                dstTensorMeanAbs[dstOffset], \
                srcMeanTensor, \
                AscendC::RoundMode::CAST_NONE, m_actual); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::Add<ElementY, true>( \
                dstTensorMeanAbs, \
                dstTensorMeanAbs[dstOffset], \
                dstTensorMeanAbs, \
                final_add_mask, \
                FTSelf::helper::CeilDiv(m_round, dst_repeat_size), \
                add_params); \
        } \
        AscendC::Add<ElementY, true>( \
            dstTensorMeanSquare, \
            red_workspace, \
            dstTensorMeanSquare, \
            final_add_mask, \
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size), \
            square_params); \
        FTSelf::Gemv::helper::VectorBarrier(); \
    }

#define FTSELF_TILE_REDUCE_MEAN_VAR_SHARED_BODY_3 \
{ \
        FTSELF_INIT_DUAL_REDUCE_GEOMETRY(layoutDst, layoutSrc, ELE_NUM_PER_C0, DST_ELE_NUM_PER_C0); \
        uint64_t mean_mask = dst_repeat_size; \
        uint64_t var_mask = dst_repeat_size; \
        uint64_t sub_mask = dst_repeat_size; \
        auto mean_params = FTSelf::Gemv::MakeUnaryRepeatParams( \
            1, 1, FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0); \
        AscendC::BinaryRepeatParams var_params; \
        var_params.dstBlkStride = 1; \
        var_params.src0BlkStride = 1; \
        var_params.src1BlkStride = 1; \
        var_params.dstRepStride = FTSelf::helper::RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0; \
        var_params.src0RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        var_params.src1RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        AscendC::BinaryRepeatParams sub_params; \
        sub_params.dstBlkStride = 1; \
        sub_params.src0BlkStride = 1; \
        sub_params.src1BlkStride = 1; \
        sub_params.dstRepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        sub_params.src0RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        sub_params.src1RepStride = FTSelf::helper::RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0; \
        InitializeReduceWorkspace(red_workspace, dst_repeat_size, m_actual); \
        for(uint32_t i=0; i < dst_repeat_num; i++){ \
            uint32_t offset = i * dst_repeat_size; \
            ScaleMeanTensor(srcMeanTensor[offset], n_ratio_factor, mean_mask, m_actual, mean_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::Sub<ElementA, true>( \
                srcMaxTensor[offset], \
                srcMaxTensor[offset], \
                srcMeanTensor[offset], \
                sub_mask, \
                m_actual, \
                sub_params); \
            AscendC::Sub<ElementA, true>( \
                srcMinTensor[offset], \
                srcMeanTensor[offset], \
                srcMinTensor[offset], \
                sub_mask, \
                m_actual, \
                sub_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::MulAddDst<ElementY, ElementA, true>( \
                red_workspace, \
                srcMaxTensor[offset], \
                srcMinTensor[offset], \
                var_mask, m_actual, \
                var_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
        } \
        if (dst_remain > 0) \
        { \
            uint32_t offset = PrepareReduceTail(dst_repeat_num, dst_repeat_size, n_actual, dst_remain); \
            uint64_t remain_mask = dst_remain; \
            ScaleMeanTensor(srcMeanTensor[offset], n_ratio_factor, dst_remain, m_actual, mean_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::Sub<ElementA, true>( \
                srcMaxTensor[offset], \
                srcMaxTensor[offset], \
                srcMeanTensor[offset], \
                remain_mask, \
                m_actual, \
                sub_params); \
            AscendC::Sub<ElementA, true>( \
                srcMinTensor[offset], \
                srcMeanTensor[offset], \
                srcMinTensor[offset], \
                remain_mask, \
                m_actual, \
                sub_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
            AscendC::MulAddDst<ElementY, ElementA, true>( \
                red_workspace, \
                srcMaxTensor[offset], \
                srcMinTensor[offset], \
                remain_mask, m_actual, \
                var_params); \
        } \
        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size; \
        FTSelf::Gemv::helper::VectorBarrier(); \
        FTSelf::Gemv::WholeReduceSumInPlace<ElementY>( \
            red_workspace, reduce_mask, m_actual, 8); \
        FTSelf::Gemv::helper::VectorBarrier(); \
        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size; \
        uint32_t dstOffset = m_round; \
        var_params.dstRepStride = 8; \
        var_params.src0RepStride = 8; \
        var_params.src1RepStride = 8; \
        AscendC::Add<ElementY, true>( \
            dstTensorVar, \
            red_workspace, \
            dstTensorVar, \
            final_add_mask, \
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size), \
            var_params); \
        FTSelf::Gemv::helper::VectorBarrier(); \
    }

template <
    /// Tag indicating architecture
    class ArchTag,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    FTSelf::Gemv::helper::FT_REDUCE_TYPE REDUCE_TYPE_,
    class AType,
    class XType,
    class YType,
    class BiasType = void
>
struct TileReduce
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>, "Unsupported TileReduce, can not find the specialization.");
};

template <class ElementA, class ElementY>
struct TileReduceRobustBase
{
    using ElementX = float;
    using ElementAccumulator = typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;
    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::RowMajor;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);
};

template <class ElementA>
struct TileReduceMeanSquareImpl : TileReduceRobustBase<ElementA, float>
{
    using Base = TileReduceRobustBase<ElementA, float>;
    using ElementY = float;
    using typename Base::LayoutDst;
    using typename Base::LayoutSrc;
    using Base::ELE_NUM_PER_C0;
    using Base::DST_ELE_NUM_PER_C0;

    FTSELF_DEVICE TileReduceMeanSquareImpl() = default;

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
        AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    FTSELF_TILE_REDUCE_MEAN_VAR_SHARED_BODY_1
};

template <class ElementA>
struct TileReduceVarianceImpl : TileReduceRobustBase<ElementA, float>
{
    using Base = TileReduceRobustBase<ElementA, float>;
    using ElementY = float;
    using typename Base::LayoutDst;
    using typename Base::LayoutSrc;
    using Base::ELE_NUM_PER_C0;
    using Base::DST_ELE_NUM_PER_C0;

    FTSELF_DEVICE TileReduceVarianceImpl() = default;

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementA> srcMinTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    FTSELF_TILE_REDUCE_MEAN_VAR_SHARED_BODY_3
};

template <FTSelf::Gemv::helper::FT_REDUCE_TYPE ReduceType, class ElementA>
struct TileReduce<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                ReduceType,
                FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<float, FTSelf::layout::VectorLayout>,
                FTSelf::GemmType<float, FTSelf::layout::VectorLayout>, void>
    : std::conditional_t<ReduceType == FTSelf::Gemv::helper::FT_REDUCE_TYPE::MEAN_SQUARE,
        TileReduceMeanSquareImpl<ElementA>, TileReduceVarianceImpl<ElementA>>
{
    static_assert(ReduceType == FTSelf::Gemv::helper::FT_REDUCE_TYPE::MEAN_SQUARE ||
        ReduceType == FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR,
        "Unsupported robust reduction type.");
};

#undef FTSELF_TILE_REDUCE_MEAN_VAR_SHARED_BODY_1
#undef FTSELF_TILE_REDUCE_MEAN_VAR_SHARED_BODY_3
}

#endif
