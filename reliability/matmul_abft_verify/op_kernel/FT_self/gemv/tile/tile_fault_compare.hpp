/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_FAULT_VCOMP_HPP
#define FTSELF_GEMV_TILE_TILE_FAULT_VCOMP_HPP

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"
#include "../../gemv/helper.hpp"

namespace FTSelf::Gemv::Tile {

template <
    /// Tag indicating architecture
    FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class ArchTag,
    class ZType,
    class XType,
    class YType
>
struct TileFaultVcompare
{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>, "Unsupported TileFaultVmad, can not find the specialization.");
};

template<
    FTSelf::Gemv::helper::FT_COMP_TYPE CompType,
    class ElementZ,
    class ElementX,
    class ElementY
>
struct TileFaultVcompare<
        CompType,
        FTSelf::Gemv::Arch::AtlasA2,
        FTSelf::GemmType<ElementZ, FTSelf::layout::VectorLayout>,
        FTSelf::GemmType<ElementX, FTSelf::layout::VectorLayout>,
        FTSelf::GemmType<ElementY, FTSelf::layout::VectorLayout>
        >
{
    static_assert(CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::XOR ||
                  CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::COMPARE ||
                  CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::SUB ||
                  CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
                  "Unsupported comparison type.");

    using ElementWIn = std::conditional_t<CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::XOR,
        uint16_t, std::conditional_t<CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::COMPARE,
        int32_t, ElementX>>;
    using ElementWTmp = std::conditional_t<
        CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::XOR ||
        CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::COMPARE, int32_t, ElementX>;
    using ElementShuffle = uint8_t;

    using LayoutDst = FTSelf::layout::VectorLayout;
    using LayoutSrc = FTSelf::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK_IN = FTSelf::Gemv::BYTE_PER_BLK / sizeof(ElementWIn);
    static constexpr uint32_t ELE_NUM_PER_BLK_TMP = FTSelf::Gemv::BYTE_PER_BLK / sizeof(ElementWTmp);
    static constexpr uint32_t ELE_NUM_PER_BLK_OUT = FTSelf::Gemv::BYTE_PER_BLK / sizeof(ElementZ);

    // Mehtods

    FTSELF_DEVICE
    TileFaultVcompare() = default;

    // AscendC::LocalTensor<ElementWIn> workSpaceTensor,
    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementZ> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
    ){
        static_assert(CompType != FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
                      "RSUB requires a threshold tensor.");

        uint32_t n_actual = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t n_round = n_actual * sizeof(ElementX) / sizeof(ElementWIn);
        uint32_t ELE_NUM_PER_REPEAT_IN = ELE_NUM_PER_BLK_IN * 8;
        uint32_t ELE_NUM_PER_REPEAT_TMP = ELE_NUM_PER_BLK_TMP * 8;

        n_round = FTSelf::helper::RoundUp(n_round, ELE_NUM_PER_REPEAT_IN);

        if constexpr (CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::XOR) {
            auto srcTensor_x_rep = srcTensor_x.template ReinterpretCast<ElementWIn>();
            auto srcTensor_y_rep = srcTensor_y.template ReinterpretCast<ElementWIn>();
            auto sharedTmpBuffer = dstTensor.template ReinterpretCast<ElementShuffle>();
            AscendC::Xor(srcTensor_x_rep, srcTensor_x_rep, srcTensor_y_rep, sharedTmpBuffer, n_round);
            FTSelf::Gemv::helper::VectorBarrier();

            auto srcTensor_x_rep_int = srcTensor_x_rep.template ReinterpretCast<ElementWTmp>();
            uint32_t n_round_comp = n_round * sizeof(ElementWIn) / sizeof(ElementWTmp);
            n_round_comp = FTSelf::helper::RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_TMP);
            AscendC::CompareScalar(dstTensor, srcTensor_x_rep_int,
                static_cast<ElementWTmp>(0), AscendC::CMPMODE::EQ, n_round_comp);
        } else if constexpr (CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::COMPARE) {
            auto srcTensor_x_rep = srcTensor_x.template ReinterpretCast<ElementWIn>();
            auto srcTensor_y_rep = srcTensor_y.template ReinterpretCast<ElementWIn>();
            AscendC::Compare(dstTensor, srcTensor_x_rep,
                srcTensor_y_rep, AscendC::CMPMODE::EQ, n_round);
        } else {
            SubAndAbs(srcTensor_x, srcTensor_y, n_round);
            uint32_t n_round_comp = n_round * sizeof(ElementWIn) / sizeof(ElementWTmp);
            n_round_comp = FTSelf::helper::RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_TMP);
            AscendC::CompareScalar(dstTensor, srcTensor_x,
                threshold, AscendC::CMPMODE::LE, n_round_comp);
        }
    }

    FTSELF_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementZ> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        AscendC::LocalTensor<ElementX> srcTensor_thre,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc, ElementX threshold
    ){
        static_assert(CompType == FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
                      "The threshold-tensor overload is only valid for RSUB.");

        uint32_t n_actual = FTSelf::helper::GetShape(layoutSrc, 0);
        uint32_t n_round_sub = n_actual * sizeof(ElementX) / sizeof(ElementWIn);
        uint32_t ELE_NUM_PER_REPEAT_IN = ELE_NUM_PER_BLK_IN * 8;
        uint32_t ELE_NUM_PER_REPEAT_TMP = ELE_NUM_PER_BLK_TMP * 8;

        ElementWTmp threshold_row = static_cast<ElementWTmp>(0.0f);
        n_round_sub = FTSelf::helper::RoundUp(n_round_sub, ELE_NUM_PER_REPEAT_IN);
        SubAndAbs(srcTensor_x, srcTensor_y, n_round_sub);

        AscendC::Sub(srcTensor_x, srcTensor_x, srcTensor_thre, n_round_sub);

        FTSelf::Gemv::helper::VectorBarrier();

        uint32_t n_round_comp = n_round_sub * sizeof(ElementWIn) / sizeof(ElementWTmp);
        n_round_comp = FTSelf::helper::RoundUp(n_round_comp, ELE_NUM_PER_REPEAT_TMP);

        AscendC::CompareScalar(dstTensor, srcTensor_x,
            threshold_row, AscendC::CMPMODE::LE, n_round_comp);

        // FTSelf::Gemv::helper::VectorBarrier();
    }

private:
    FTSELF_DEVICE
    static void SubAndAbs(
        AscendC::LocalTensor<ElementX> srcTensor_x,
        AscendC::LocalTensor<ElementY> srcTensor_y,
        uint32_t elementCount)
    {
        AscendC::Sub(srcTensor_x, srcTensor_x, srcTensor_y, elementCount);
        FTSelf::Gemv::helper::VectorBarrier();
        AscendC::Abs(srcTensor_x, srcTensor_x, elementCount);
        FTSelf::Gemv::helper::VectorBarrier();
    }
};
}

#endif // FTSELF_GEMV_TILE_TILE_FAULT_VCOMP_HPP
