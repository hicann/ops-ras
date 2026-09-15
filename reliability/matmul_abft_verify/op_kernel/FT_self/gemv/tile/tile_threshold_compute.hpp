/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_THRESHOLD_COMPUTE_HPP
#define FTSELF_GEMV_TILE_TILE_THRESHOLD_COMPUTE_HPP

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../helper/type_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

// template <
//     /// Tag indicating architecture
//     class ArchTag,
//     class AType,
//     class XType,
//     class YType,
//     class BiasType = void
// >
// struct TileThreCalc
// };

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileThreCalc<FTSelf::Gemv::Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
                FTSelf::GemmType<ElementA, FTSelf::layout::RowMajor>,
                FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>,
                FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT;
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    FTSELF_DEVICE
    TileThreCalc() = default;

    //
    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementX> reduceTensor_v,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t dst_offset_ratio = 2
    )
    {
        uint32_t m_actual = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t n_actual = FTSelf::helper::GetShape(layoutSrc, 1);
        uint32_t m_round = FTSelf::helper::GetShape(layoutDst, 0);
        uint32_t n_round = FTSelf::helper::GetShape(layoutDst, 1);
        uint32_t temp_repeat_size = FTSelf::Gemv::BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = FTSelf::helper::CeilDiv(m_actual, temp_repeat_size);


        AscendC::Duplicate<ElementX>(
            reduceTensor_v,
            (ElementX)0.0,
            temp_repeat_size,
            FTSelf::helper::CeilDiv(m_round * temp_repeat_size, temp_repeat_size),
            1,
            8
        );

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;

        FTSelf::Gemv::helper::VectorBarrier();
        auto params = FTSelf::Gemv::MakeBinaryRepeatParams(
            1, 1, 1,
            FTSelf::helper::RoundUp(temp_repeat_size, temp_repeat_size) / (FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementX)), FTSelf::helper::RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0, 0);

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = FTSelf::helper::RoundUp(temp_repeat_size, temp_repeat_size) / (FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementX));
        max_params.src0RepStride = FTSelf::helper::RoundUp(temp_repeat_size, temp_repeat_size) / (FTSelf::Gemv::BYTE_PER_C0 / sizeof(ElementX));;
        max_params.src1RepStride = FTSelf::helper::RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        auto abs_params = FTSelf::Gemv::MakeUnaryRepeatParams(
            1, 1, FTSelf::helper::RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0, FTSelf::helper::RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0);

        uint64_t abs_mask = temp_repeat_size;
        uint8_t unit_repeatTimes = m_actual;
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;
            // AscendC::Abs(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                abs_mask,
                m_actual,
                abs_params);

            FTSelf::Gemv::helper::VectorBarrier();



            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                abs_mask,
                m_actual,
                max_params);

            FTSelf::Gemv::helper::VectorBarrier();

        }
        // AscendC::SetMaskNorm();
        // AscendC::ResetMask();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                remain_mask,
                m_actual,
                abs_params);

            FTSelf::Gemv::helper::VectorBarrier();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);

            // FTSelf::Gemv::helper::VectorBarrier();

        }

        int32_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        FTSelf::Gemv::helper::VectorBarrier();

        // }

        AscendC::WholeReduceMax<ElementX, true>(
            srcTensor_m,
            reduceTensor_v,
            reduce_mask,
            m_actual,
            1,
            1,
            8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);


        FTSelf::Gemv::helper::VectorBarrier();

        /*
        每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float->half转换每次迭代操作64个源/目的元素。
        当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
        dst与src的应为不同Tensor，或同一Tensor的同一元素，不支持同一Tensor的不同元素。
        src为float，dst为float时，取整模式表示向整数取整（仍为float类型），其他情况表示向dst数据类型所能表示的数字取整。
        */

        if constexpr (std::is_same_v<ElementX, ElementY>) {
            AscendC::Max(dstTensor, dstTensor, srcTensor_m, m_round);
        } else {
            uint32_t dstOffset = m_round * dst_offset_ratio;
            AscendC::Cast<ElementY, ElementX>(
                dstTensor[dstOffset],
                srcTensor_m,
                AscendC::RoundMode::CAST_NONE, m_actual);
            FTSelf::Gemv::helper::VectorBarrier();
            AscendC::Max(dstTensor, dstTensor, dstTensor[dstOffset], m_round);
        }
        FTSelf::Gemv::helper::VectorBarrier();

        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // FTSelf::Gemv::helper::VectorBarrier();
    }
};
}

#endif
