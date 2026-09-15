/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMV_TILE_TILE_MATRIX_COPY_UB_TO_GM_HPP
#define FTSELF_GEMV_TILE_TILE_MATRIX_COPY_UB_TO_GM_HPP

#include "../../helper/layout_helper.hpp"
#include "../../helper/base_helper.hpp"
#include "../../helper/math_helper.hpp"


namespace FTSelf::Gemv::Tile {


template <
    class ArchTag,
    class GmType
>
struct MatrixCopyUBToGm {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm for the matrices, can not find the specialization.");
};

template <typename Element>
struct MatrixCopyUBToGm<FTSelf::Gemv::Arch::AtlasA2, FTSelf::GemmType<Element, FTSelf::layout::RowMajor>> {
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);

    // Mehtods
    FTSELF_DEVICE
    MatrixCopyUBToGm() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        /*
                            表4 DataCopyExtParams结构体参数定义
            参数名称                                含义
            blockCount          指定该指令包含的连续传输数据块个数，数据类型为uint16_t，
                                取值范围：blockCount∈[1, 4095]。

            blockLen            指定该指令每个连续传输数据块长度，该指令支持非对齐搬运，
                                每个连续传输数据块长度单位为Byte。数据类型为uint32_t，
                                取值范围：blockLen∈[1, 2097151]。

            srcStride           源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                                如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)。
                                如果源操作数的逻辑位置为GM，则单位为Byte。数据类型为uint32_t，
                                srcStride不要超出该数据类型的取值范围。

            dstStride           目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                                如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)，
                                如果目的操作数的逻辑位置为GM，则单位为Byte。数据类型为uint32_t，
                                dstStride不要超出该数据类型的取值范围。

            rsv                 保留字段。
        */
        AscendC::DataCopyExtParams dataCopyParams(
            FTSelf::helper::GetShape(layoutDst, 0),
            FTSelf::helper::GetShape(layoutDst, 1) * sizeof(Element),
            (FTSelf::helper::GetStride(layoutSrc, 0) - FTSelf::helper::GetShape(layoutSrc, 1)) / ELE_NUM_PER_C0,
            (FTSelf::helper::GetStride(layoutDst, 0) - FTSelf::helper::GetShape(layoutDst, 1)) * sizeof(Element),
            0
        );
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

// new add vectorlayout version
template <typename Element>
struct MatrixCopyUBToGm<FTSelf::Gemv::Arch::AtlasA2, FTSelf::GemmType<Element, FTSelf::layout::VectorLayout>> {
    using LayoutSrc = FTSelf::layout::VectorLayout;
    using LayoutDst = FTSelf::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = FTSelf::Gemv::BYTE_PER_BLK / sizeof(Element);

    // Mehtods
    FTSELF_DEVICE
    MatrixCopyUBToGm() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            1,
            FTSelf::helper::GetShape(layoutDst, 0) * sizeof(Element),
            0,
            0,
            0
        );
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

template <
    class ArchTag,
    class GmType
>
struct MatrixCopyUBToGmAligned {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm for the matrices, can not find the specialization.");
};

template <typename Element>
struct MatrixCopyUBToGmAligned<FTSelf::Gemv::Arch::AtlasA2, FTSelf::GemmType<Element, FTSelf::layout::RowMajor>> {
    using LayoutDst = FTSelf::layout::RowMajor;
    using LayoutSrc = FTSelf::layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = FTSelf::Gemv::BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;
    FTSELF_DEVICE
    MatrixCopyUBToGmAligned() = default;

    FTSELF_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t rows = FTSelf::helper::GetShape(layoutDst, 0);
        uint32_t cols = FTSelf::helper::GetShape(layoutDst, 1);
        uint32_t srcStride = (FTSelf::helper::GetStride(layoutSrc, 0) - FTSelf::helper::GetShape(layoutSrc, 1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (FTSelf::helper::GetStride(layoutDst, 0) - FTSelf::helper::GetShape(layoutDst, 1)) / ELE_NUM_PER_BLK;

        if ((FTSelf::helper::GetShape(layoutSrc, 1) == FTSelf::helper::GetStride(layoutSrc, 0)) && (FTSelf::helper::GetShape(layoutDst, 1) == FTSelf::helper::GetStride(layoutDst, 0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < FTSelf::Gemv::STRIDE_LIMIT && dstStride < FTSelf::Gemv::STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = FTSelf::helper::CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(
                    rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride
                );
                DataCopy(dstTensor[i * MAX_REPEAT * FTSelf::helper::GetStride(layoutDst, 0)],
                         srcTensor[i * MAX_REPEAT * FTSelf::helper::GetStride(layoutSrc, 0)], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * FTSelf::helper::GetStride(layoutDst, 0)], srcTensor[i * FTSelf::helper::GetStride(layoutSrc, 0)], cols);
            }
        }
    }
};
} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_GEMV_TILE_TILE_MATRIX_COPY_UB_TO_GM_HPP
