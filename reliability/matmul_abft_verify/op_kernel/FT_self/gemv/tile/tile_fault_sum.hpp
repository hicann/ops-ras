/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF
#define FTSELF_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

template <class Element>
FTSELF_DEVICE void ReduceAddRowsInPlace(
    AscendC::LocalTensor<Element> tensor, uint32_t rowCount, uint32_t actualWidth,
    uint32_t roundedWidth, uint32_t repeatSize, uint32_t repeatNum, uint32_t remain,
    AscendC::BinaryRepeatParams const &params, bool skipTailWithoutFullRepeat)
{
    for (uint32_t i = 1; i < repeatNum; ++i) {
        uint32_t offset = i * repeatSize;
        AscendC::Add<Element, true>(tensor, tensor[offset], tensor, repeatSize, rowCount, params);
        FTSelf::Gemv::helper::VectorBarrier();
    }
    if (remain > 0 && (!skipTailWithoutFullRepeat || repeatNum > 0)) {
        uint32_t offset = repeatNum * repeatSize;
        remain = FTSelf::Gemv::helper::ClampTail(offset, remain, actualWidth);
        AscendC::Add<Element, true>(tensor, tensor[offset], tensor, remain, rowCount, params);
    }
    uint64_t reduceMask = (repeatNum == 0) ? remain : repeatSize;
    FTSelf::Gemv::helper::VectorBarrier();
    AscendC::WholeReduceSum<Element, true>(tensor, tensor, reduceMask, rowCount, 1, 1,
        FTSelf::helper::RoundUp(roundedWidth, repeatSize) /
            (FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element)));
}

template <class Element>
FTSELF_DEVICE void AccumulateMaxChunk(
    AscendC::LocalTensor<Element> tensor, uint32_t offset, uint64_t mask,
    uint32_t rowCount, AscendC::BinaryRepeatParams const &params)
{
    AscendC::Max<Element, true>(tensor, tensor[offset], tensor, mask, rowCount, params);
}

template <class Element>
FTSELF_DEVICE void FinalizeMaxRows(
    AscendC::LocalTensor<Element> tensor, uint64_t mask, uint32_t rowCount,
    uint32_t roundedWidth, uint32_t repeatSize)
{
    FTSelf::Gemv::helper::VectorBarrier();
    FTSelf::Gemv::WholeReduceMaxInPlace<Element>(tensor, mask, rowCount,
        FTSelf::helper::RoundUp(roundedWidth, repeatSize) /
            (FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element)));
}

#define FTSELF_CLAMP_REDUCE_TAIL(offset_, remain_, actualWidth_) \
    if ((offset_) + (remain_) > (actualWidth_)) { \
        (remain_) = (actualWidth_) - (offset_); \
    }

#define FTSELF_DECLARE_TILE_FAULT_SUM_ROW_VECTOR(reduceType_) \
template <class Element> \
struct TileFaultSum<FTSelf::Gemv::Arch::AtlasA2, reduceType_, \
    FTSelf::GemmType<Element, FTSelf::layout::RowMajor>, \
    FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>, void> \
{ \
    FTSELF_TILE_FAULT_SUM_TYPES(Element, Element, FTSelf::layout::RowMajor); \
    FTSELF_DEVICE TileFaultSum() = default; \
    FTSELF_DEVICE void operator()(AscendC::LocalTensor<ElementY> dstTensor, \
        AscendC::LocalTensor<ElementA> srcTensor_m, \
        AscendC::LocalTensor<ElementAccumulator> temp, \
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)

#define FTSELF_DECLARE_TILE_FAULT_SUM_HALF_FLOAT(reduceType_) \
struct TileFaultSum<FTSelf::Gemv::Arch::AtlasA2, reduceType_, \
    FTSelf::GemmType<half, FTSelf::layout::RowMajor>, \
    FTSelf::GemmType<float, FTSelf::layout::VectorLayout>, void> \
    : TileFaultSumBase<half, float> \
{ \
    using Base = TileFaultSumBase<half, float>; \
    using typename Base::ElementA; \
    using typename Base::ElementY; \
    using typename Base::LayoutDst; \
    using typename Base::LayoutSrc; \
    using Base::ELE_NUM_PER_C0; \
    using Base::DST_ELE_NUM_PER_C0;

#define FTSELF_BEGIN_DUAL_REDUCE_OPERATOR() \
    FTSELF_DEVICE void operator()(AscendC::LocalTensor<ElementY> dstTensor, \
        AscendC::LocalTensor<ElementA> srcTensor_m, \
        AscendC::LocalTensor<ElementA> temp, LayoutDst const &layoutDst, \
        LayoutSrc const &layoutSrc) \
    { \
        FTSELF_INIT_DUAL_REDUCE_GEOMETRY( \
            layoutDst, layoutSrc, ELE_NUM_PER_C0, DST_ELE_NUM_PER_C0)

#define FTSELF_TILE_FAULT_SUM_TYPES(elementA_, elementY_, layout_) \
    using ElementA = elementA_; \
    using ElementX = elementA_; \
    using ElementY = elementY_; \
    using ElementAccumulator = ElementY; \
    using LayoutDst = layout_; \
    using LayoutSrc = layout_; \
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA)


#define FTSELF_TILE_FAULT_SUM_SHARED_BODY_3 \
{ \
        FTSELF_INIT_REDUCE_GEOMETRY(layoutDst, layoutSrc, ELE_NUM_PER_C0); \
        uint64_t add_mask = repeat_size; \
        uint64_t max_mask = repeat_size; \
        /* \
        控制操作数地址步长的参数。BinaryRepeatParams类型， \
        包含操作数相邻迭代间相同datablock的地址步长， \
        操作数同一迭代内不同datablock的地址步长等参数。 \
        相邻迭代间的地址步长参数说明请参考repeatStride； \
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。 \
        */ \
        auto max_params = FTSelf::Gemv::MakeBinaryRepeatParams( \
            1, 1, 1, \
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0); \
        AscendC::BinaryRepeatParams sum_params; \
        sum_params.dstBlkStride = 1; \
        sum_params.src0BlkStride = 1; \
        sum_params.src1BlkStride = 1; \
        sum_params.dstRepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / (FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementAccumulator)); \
        sum_params.src0RepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0; \
        sum_params.src1RepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / (FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementAccumulator)); \
        AscendC::Duplicate<ElementAccumulator>( \
                sum_workspace, \
                (ElementAccumulator)0.0, \
                repeat_size, \
                FTSelf::helper::CeilDiv(m_actual * repeat_size, repeat_size), \
                1, \
                8 \
        ); \
        FTSelf::Gemv::helper::VectorBarrier(); \
        AscendC::Add<ElementAccumulator, true>( \
            sum_workspace, \
            srcTensor_m, \
            sum_workspace, \
            add_mask, \
            m_actual, \
            sum_params); \
        FTSelf::Gemv::helper::VectorBarrier(); \
        for (uint32_t i = 1; i < repeat_num; i++) { \
            uint32_t offset = i * repeat_size; \
            AscendC::Add<ElementAccumulator, true>( \
                sum_workspace, \
                srcTensor_m[offset], \
                sum_workspace, \
                add_mask, \
                m_actual, \
                sum_params); \
            AscendC::Max<ElementA, true>( \
                srcTensor_m, \
                srcTensor_m[offset], \
                srcTensor_m, \
                max_mask, \
                m_actual, \
                max_params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
        } \
        if (remain > 0 && repeat_num > 0) \
        { \
            uint32_t offset = repeat_num * repeat_size; \
            if (offset + remain > n_actual) \
            { \
                remain = n_actual - offset; \
            } \
            uint64_t remain_mask = remain; \
            AscendC::Add<ElementAccumulator, true>( \
                sum_workspace, \
                srcTensor_m[offset], \
                sum_workspace, \
                remain_mask, \
                m_actual, \
                sum_params); \
            AccumulateMaxChunk(srcTensor_m, offset, remain_mask, m_actual, max_params); \
        } \
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size; \
        FinalizeMaxRows(srcTensor_m, reduce_mask, m_actual, n_round, repeat_size); \
        AscendC::WholeReduceSum<ElementAccumulator, true>( \
            sum_workspace, \
            sum_workspace, \
            reduce_mask, \
            m_actual, \
            1, \
            1, \
            FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0); \
        uint64_t add_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size; \
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size; \
        sum_params.dstRepStride = 8; \
        sum_params.src0RepStride = 8; \
        sum_params.src1RepStride = 8; \
        max_params.dstRepStride = 8; \
        max_params.src0RepStride = 8; \
        max_params.src1RepStride = 8; \
        FTSelf::Gemv::helper::VectorBarrier(); \
        AscendC::Add<ElementA, true>( \
            dstTensorSum, \
            sum_workspace, \
            dstTensorSum, \
            add_final_mask, \
            FTSelf::helper::CeilDiv(m_round, repeat_size), \
            sum_params); \
        AscendC::Max<ElementA, true>( \
            dstTensorMax, \
            srcTensor_m, \
            dstTensorMax, \
            max_final_mask, \
            FTSelf::helper::CeilDiv(m_round, repeat_size), \
            max_params); \
    }


#define FTSELF_TILE_FAULT_SUM_SHARED_BODY_5 \
{ \
        FTSELF_INIT_REDUCE_GEOMETRY(layoutDst, layoutSrc, ELE_NUM_PER_C0); \
        uint64_t min_mask = repeat_size; \
        uint64_t max_mask = repeat_size; \
        /* \
        控制操作数地址步长的参数。BinaryRepeatParams类型， \
        包含操作数相邻迭代间相同datablock的地址步长， \
        操作数同一迭代内不同datablock的地址步长等参数。 \
        相邻迭代间的地址步长参数说明请参考repeatStride； \
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。 \
        */ \
        auto max_params = FTSelf::Gemv::MakeBinaryRepeatParams( \
            1, 1, 1, \
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0); \
        AscendC::BinaryRepeatParams min_params; \
        min_params.dstBlkStride = 1; \
        min_params.src0BlkStride = 1; \
        min_params.src1BlkStride = 1; \
        min_params.dstRepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / (FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementAccumulator)); \
        min_params.src0RepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0; \
        min_params.src1RepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / (FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementAccumulator)); \
        AscendC::Duplicate<ElementA>( \
                min_workspace, \
                (ElementA)(1.0e10f), \
                repeat_size, \
                FTSelf::helper::CeilDiv(m_actual * repeat_size, repeat_size), \
                1, \
                8 \
        ); \
        FTSelf::Gemv::helper::VectorBarrier(); \
        AscendC::Min<ElementA, true>( \
            min_workspace, \
            srcTensor_m, \
            min_workspace, \
            min_mask, \
            m_actual, \
            min_params); \
        FTSelf::Gemv::helper::VectorBarrier(); \
        for (uint32_t i = 1; i < repeat_num; i++) { \
            uint32_t offset = i * repeat_size; \
            AscendC::Min<ElementA, true>( \
                min_workspace, \
                srcTensor_m[offset], \
                min_workspace, \
                min_mask, \
                m_actual, \
                min_params); \
            AscendC::Max<ElementA, true>( \
                srcTensor_m, \
                srcTensor_m[offset], \
                srcTensor_m, \
                max_mask, \
                m_actual, \
                max_params); \
\
            FTSelf::Gemv::helper::VectorBarrier(); \
        } \
        if (remain > 0) \
        { \
            uint32_t offset = repeat_num * repeat_size; \
            FTSELF_CLAMP_REDUCE_TAIL(offset, remain, n_actual); \
            uint64_t remain_mask = remain; \
            AscendC::Min<ElementA, true>( \
                min_workspace, \
                srcTensor_m[offset], \
                min_workspace, \
                remain_mask, \
                m_actual, \
                min_params); \
\
            AccumulateMaxChunk(srcTensor_m, offset, remain_mask, m_actual, max_params); \
        } \
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size; \
        FinalizeMaxRows(srcTensor_m, reduce_mask, m_actual, n_round, repeat_size); \
        AscendC::WholeReduceMin<ElementA, true>( \
            min_workspace, \
            min_workspace, \
            reduce_mask, \
            m_actual, \
            1, \
            1, \
            FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0, \
            AscendC::ReduceOrder::ORDER_ONLY_VALUE \
        ); \
        uint64_t min_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size; \
        uint64_t max_final_mask = (m_actual < repeat_size) ? m_actual : repeat_size; \
        min_params.dstRepStride = 8; \
        min_params.src0RepStride = 8; \
        min_params.src1RepStride = 8; \
        max_params.dstRepStride = 8; \
        max_params.src0RepStride = 8; \
        max_params.src1RepStride = 8; \
        FTSelf::Gemv::helper::VectorBarrier(); \
        AscendC::Min<ElementA, true>( \
            dstTensorMin, \
            min_workspace, \
            dstTensorMin, \
            min_final_mask, \
            FTSelf::helper::CeilDiv(m_round, repeat_size), \
            min_params); \
        AscendC::Max<ElementA, true>( \
            dstTensorMax, \
            srcTensor_m, \
            dstTensorMax, \
            max_final_mask, \
            FTSelf::helper::CeilDiv(m_round, repeat_size), \
            max_params); \
    }


template <
    /// Tag indicating architecture
    class ArchTag,
    FTSelf::Gemv::helper::FT_REDUCE_TYPE REDUCE_TYPE_,
    class AType,
    class YType,
    class BiasType = void
>
struct TileFaultSum
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>, "Unsupported TileFaultSum, can not find the specialization.");
};

template <class ElementA_, class ElementY_>
struct TileFaultSumBase
{
    using ElementA = ElementA_;
    using ElementX = ElementA_;
    using ElementY = ElementY_;
    using ElementAccumulator = ElementY;
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);
};


FTSELF_DECLARE_TILE_FAULT_SUM_ROW_VECTOR(FTSelf::Gemv::helper::FT_REDUCE_TYPE::SUM)
    {
        FTSELF_INIT_REDUCE_GEOMETRY(layoutDst, layoutSrc, ELE_NUM_PER_C0);

        uint64_t add_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        auto params = FTSelf::Gemv::MakeBinaryRepeatParams(
            1, 1, 1,
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);






        ReduceAddRowsInPlace(srcTensor_m, m_actual, n_actual, n_round, repeat_size,
            repeat_num, remain, params, true);

        add_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        FTSelf::Gemv::helper::SetRepeatStrides(params, 8, 8, 8);

        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Add<ElementA, true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            add_mask,
            FTSelf::helper::CeilDiv(m_round, repeat_size),
            params);
    }
};


template <>
struct TileFaultSum<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::SUM_MAX,
                FTSelf::GemmType<float, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<float, FTSelf::layout::VectorLayout>,
                void>
{
    FTSELF_TILE_FAULT_SUM_TYPES(float, float, FTSelf::layout::RowMajor);

    // Methods

    FTSELF_DEVICE
    TileFaultSum() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    FTSELF_TILE_FAULT_SUM_SHARED_BODY_3
};


template <>
FTSELF_DECLARE_TILE_FAULT_SUM_HALF_FLOAT(FTSelf::Gemv::helper::FT_REDUCE_TYPE::SUM_MAX)

    // Methods
    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorSum,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> sum_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        FTSELF_INIT_DUAL_REDUCE_GEOMETRY(
            layoutDst, layoutSrc, ELE_NUM_PER_C0, DST_ELE_NUM_PER_C0);
        uint32_t dstOffset = m_round;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        auto max_params = FTSelf::Gemv::MakeBinaryRepeatParams(
            1, 1, 1,
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        AscendC::BinaryRepeatParams sum_params;
        sum_params.dstBlkStride = 1;
        sum_params.src0BlkStride = 1;
        sum_params.src1BlkStride = 1;

        // params.dstRepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        sum_params.dstRepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src0RepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        sum_params.src1RepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;







        AscendC::Duplicate<ElementA>(
                sum_workspace,
                (ElementA)0.0,
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                FTSelf::helper::CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        FTSelf::Gemv::helper::VectorBarrier();

        AscendC::Add<ElementA, true>(
            sum_workspace,
            srcTensor_m,
            sum_workspace,
            add_mask,
            m_actual,
            sum_params);

        FTSelf::Gemv::helper::VectorBarrier();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Add<ElementA, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                add_mask,
                m_actual,
                sum_params);

            AccumulateMaxChunk(srcTensor_m, offset, max_mask, m_actual, max_params);

            FTSelf::Gemv::helper::VectorBarrier();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            remain = FTSelf::Gemv::helper::ClampTail(offset, remain, n_actual);
            // m_actual *
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                sum_workspace,
                srcTensor_m[offset],
                sum_workspace,
                remain_mask,
                m_actual,
                sum_params);

            AccumulateMaxChunk(srcTensor_m, offset, remain_mask, m_actual, max_params);
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        FinalizeMaxRows(srcTensor_m, reduce_mask, m_actual, n_round, repeat_size);

        AscendC::WholeReduceSum<ElementA, true>(
            sum_workspace,
            sum_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0);

        FTSelf::Gemv::helper::VectorBarrier();

        AscendC::Cast<ElementY, ElementA>(
            dstTensorMax[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);

        AscendC::Cast<ElementY, ElementA>(
            dstTensorSum[dstOffset],
            sum_workspace,
            AscendC::RoundMode::CAST_NONE, m_actual);

        FTSelf::Gemv::helper::VectorBarrier();

        uint64_t add_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;
        uint64_t max_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        FTSelf::Gemv::helper::SetRepeatStrides(sum_params, 8, 8, 8);

        FTSelf::Gemv::helper::SetRepeatStrides(max_params, 8, 8, 8);

        AscendC::Add<ElementY, true>(
            dstTensorSum,
            dstTensorSum[dstOffset],
            dstTensorSum,
            add_final_mask,
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size),
            sum_params);

        AscendC::Max<ElementY, true>(
            dstTensorMax,
            dstTensorMax[dstOffset],
            dstTensorMax,
            max_final_mask,
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size),
            max_params);

    }
};



#define FTSELF_TILE_FAULT_SUM_SHARED_BODY_1 \
{ \
        FTSELF_INIT_REDUCE_GEOMETRY(layoutDst, layoutSrc, ELE_NUM_PER_C0); \
        uint64_t max_mask = repeat_size; \
        /* \
        控制操作数地址步长的参数。BinaryRepeatParams类型， \
        包含操作数相邻迭代间相同datablock的地址步长， \
        操作数同一迭代内不同datablock的地址步长等参数。 \
        相邻迭代间的地址步长参数说明请参考repeatStride； \
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。 \
        */ \
        auto params = FTSelf::Gemv::MakeBinaryRepeatParams( \
            1, 1, 1, \
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0); \
        for (uint32_t i = 1; i < repeat_num; i++) { \
            uint32_t offset = i * repeat_size; \
            AscendC::Max<ElementA, true>( \
                srcTensor_m, \
                srcTensor_m[offset], \
                srcTensor_m, \
                max_mask, \
                m_actual, \
                params); \
            FTSelf::Gemv::helper::VectorBarrier(); \
        } \
        if (remain > 0) \
        { \
            uint32_t offset = repeat_num * repeat_size; \
            FTSELF_CLAMP_REDUCE_TAIL(offset, remain, n_actual); \
            uint64_t remain_mask = remain; \
            AscendC::Max<ElementA, true>( \
                srcTensor_m, \
                srcTensor_m[offset], \
                srcTensor_m, \
                remain_mask, \
                m_actual, \
                params); \
        } \
        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size; \
        FTSelf::Gemv::helper::VectorBarrier(); \
        FTSelf::Gemv::WholeReduceMaxInPlace<ElementA>( \
            srcTensor_m, reduce_mask, m_actual, \
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0); \
        max_mask = (m_actual < repeat_size) ? m_actual : repeat_size; \
        params.dstRepStride = 8; \
        params.src0RepStride = 8; \
        params.src1RepStride = 8; \
        FTSelf::Gemv::helper::VectorBarrier(); \
        AscendC::Max<ElementA, true>( \
            dstTensor, \
            srcTensor_m, \
            dstTensor, \
            max_mask, \
            FTSelf::helper::CeilDiv(m_round, repeat_size), \
            params); \
    }

FTSELF_DECLARE_TILE_FAULT_SUM_ROW_VECTOR(FTSelf::Gemv::helper::FT_REDUCE_TYPE::MAX)
    FTSELF_TILE_FAULT_SUM_SHARED_BODY_1
};

template <>
struct TileFaultSum<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::SUM,
                FTSelf::GemmType<half, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<float, FTSelf::layout::VectorLayout>,
                void>
{
    FTSELF_TILE_FAULT_SUM_TYPES(half, float, FTSelf::layout::RowMajor);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods
    FTSELF_DEVICE
    TileFaultSum() = default;

    FTSELF_BEGIN_DUAL_REDUCE_OPERATOR();

        uint64_t add_mask = repeat_size;
        auto params = FTSelf::Gemv::MakeBinaryRepeatParams(
            1, 1, 1,
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);
        ReduceAddRowsInPlace(srcTensor_m, m_actual, n_actual, n_round, repeat_size,
            repeat_num, remain, params, false);

        FTSelf::Gemv::helper::VectorBarrier();

        add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        AscendC::Cast<ElementY, ElementA>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);
        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetRepeatStrides(params, 8, 8, 8);


        AscendC::Add<ElementY, true>(
            dstTensor,
            dstTensor[dstOffset],
            dstTensor,
            add_mask,
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size),
            params);
    }
};


template <>
FTSELF_DECLARE_TILE_FAULT_SUM_HALF_FLOAT(FTSelf::Gemv::helper::FT_REDUCE_TYPE::MAX)
    // Methods

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        FTSELF_INIT_DUAL_REDUCE_GEOMETRY(
            layoutDst, layoutSrc, ELE_NUM_PER_C0, DST_ELE_NUM_PER_C0);

        // m_actual *
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;

        // params.dstRepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        // params.src0RepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        // params.src1RepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;

        params.dstRepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;




        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                max_mask,
                m_actual,
                params);
            FTSelf::Gemv::helper::VectorBarrier();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            remain = FTSelf::Gemv::helper::ClampTail(offset, remain, n_actual);
            // m_actual *
            uint64_t remain_mask = remain;
            AscendC::Max<ElementA, true>(
                srcTensor_m,
                srcTensor_m[offset],
                srcTensor_m,
                remain_mask,
                m_actual,
                params);
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        FTSelf::Gemv::helper::VectorBarrier();



        FTSelf::Gemv::WholeReduceMaxInPlace<ElementA>(
            srcTensor_m, reduce_mask, m_actual,
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        FTSelf::Gemv::helper::VectorBarrier();

        max_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        AscendC::Cast<ElementY, ElementA>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);
        FTSelf::Gemv::helper::VectorBarrier();

        FTSelf::Gemv::helper::SetRepeatStrides(params, 8, 8, 8);

        AscendC::Max<ElementY, true>(
            dstTensor,
            dstTensor[dstOffset],
            dstTensor,
            max_mask,
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size),
            params);
    }
};




//     static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);

//     // Methods

//     FTSELF_DEVICE

template <class Element>
struct TileFaultSum<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::MAX_MIN,
                FTSelf::GemmType<Element, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>,
                void>
{
    FTSELF_TILE_FAULT_SUM_TYPES(Element, Element, FTSelf::layout::RowMajor);

    // Methods

    FTSELF_DEVICE
    TileFaultSum() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMin,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> min_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    FTSELF_TILE_FAULT_SUM_SHARED_BODY_5
};



template <>
FTSELF_DECLARE_TILE_FAULT_SUM_HALF_FLOAT(FTSelf::Gemv::helper::FT_REDUCE_TYPE::MAX_MIN)

    // Methods
    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMin,
        AscendC::LocalTensor<ElementY> dstTensorMax,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementA> min_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        FTSELF_INIT_DUAL_REDUCE_GEOMETRY(
            layoutDst, layoutSrc, ELE_NUM_PER_C0, DST_ELE_NUM_PER_C0);
        uint32_t dstOffset = m_round;

        // m_actual *
        uint64_t min_mask = repeat_size;
        uint64_t max_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        auto max_params = FTSelf::Gemv::MakeBinaryRepeatParams(
            1, 1, 1,
            FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        AscendC::BinaryRepeatParams min_params;
        min_params.dstBlkStride = 1;
        min_params.src0BlkStride = 1;
        min_params.src1BlkStride = 1;

        // params.dstRepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        min_params.dstRepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;
        min_params.src0RepStride = FTSelf::helper::RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        min_params.src1RepStride = FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0;







        AscendC::Duplicate<ElementA>(
                min_workspace,
                (ElementA)(1.0e10f),
                repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                FTSelf::helper::CeilDiv(m_actual * repeat_size, repeat_size), // 求行和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        FTSelf::Gemv::helper::VectorBarrier();

        AscendC::Min<ElementA, true>(
            min_workspace,
            srcTensor_m,
            min_workspace,
            min_mask,
            m_actual,
            min_params);

        FTSelf::Gemv::helper::VectorBarrier();

        for (uint32_t i = 1; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;

            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                min_mask,
                m_actual,
                min_params);
            AccumulateMaxChunk(srcTensor_m, offset, max_mask, m_actual, max_params);
            FTSelf::Gemv::helper::VectorBarrier();
        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            remain = FTSelf::Gemv::helper::ClampTail(offset, remain, n_actual);
            // m_actual *
            uint64_t remain_mask = remain;
            AscendC::Min<ElementA, true>(
                min_workspace,
                srcTensor_m[offset],
                min_workspace,
                remain_mask,
                m_actual,
                min_params);
            AccumulateMaxChunk(srcTensor_m, offset, remain_mask, m_actual, max_params);
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        FinalizeMaxRows(srcTensor_m, reduce_mask, m_actual, n_round, repeat_size);

        AscendC::WholeReduceMin<ElementA, true>(
            min_workspace,
            min_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            FTSelf::helper::RoundUp(repeat_size, repeat_size) / ELE_NUM_PER_C0,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        FTSelf::Gemv::helper::VectorBarrier();

        AscendC::Cast<ElementY, ElementA>(
            dstTensorMax[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);

        AscendC::Cast<ElementY, ElementA>(
            dstTensorMin[dstOffset],
            min_workspace,
            AscendC::RoundMode::CAST_NONE, m_actual);

        FTSelf::Gemv::helper::VectorBarrier();

        uint64_t min_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;
        uint64_t max_final_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        FTSelf::Gemv::helper::SetRepeatStrides(min_params, 8, 8, 8);

        FTSelf::Gemv::helper::SetRepeatStrides(max_params, 8, 8, 8);

        AscendC::Min<ElementY, true>(
            dstTensorMin,
            dstTensorMin[dstOffset],
            dstTensorMin,
            min_final_mask,
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size),
            min_params);

        AscendC::Max<ElementY, true>(
            dstTensorMax,
            dstTensorMax[dstOffset],
            dstTensorMax,
            max_final_mask,
            FTSelf::helper::CeilDiv(m_round, dst_repeat_size),
            max_params);

    }
};


#undef FTSELF_TILE_FAULT_SUM_SHARED_BODY_3
#undef FTSELF_TILE_FAULT_SUM_SHARED_BODY_1
#undef FTSELF_TILE_FAULT_SUM_TYPES
#undef FTSELF_TILE_FAULT_SUM_SHARED_BODY_5
}
#endif // FTSELF_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF
