/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_HELPER_LAYOUT_HELPER_HPP
#define FTSELF_HELPER_LAYOUT_HELPER_HPP

#include <cstdint>

#include "../helper/base_helper.hpp"
#include "../helper/math_helper.hpp"

namespace FTSelf::Gemv::layout {

template <int Rank, class IndexType = uint32_t>
struct Coord {
    using Index = IndexType;
    static constexpr int RANK = Rank;
    Index values[Rank];

    FTSELF_GEMV_HOST_DEVICE constexpr explicit Coord(Index value = Index(0)) : values{} {
        for (int i = 0; i < Rank; ++i) {
            values[i] = value;
        }
    }

    template <class... Values>
    FTSELF_GEMV_HOST_DEVICE constexpr explicit Coord(Values... input) : values{Index(input)...} {}

    FTSELF_GEMV_HOST_DEVICE

    constexpr Index const &operator[](int index) const { return values[index]; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Index &operator[](int index) { return values[index]; }
};

struct RowMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    Shape shape_;
    Stride stride_;

    FTSELF_GEMV_HOST_DEVICE constexpr RowMajor(Index rows = 0, Index columns = 0)
        : shape_(rows, columns), stride_(LongIndex(columns), LongIndex(1)) {}

    FTSELF_GEMV_HOST_DEVICE constexpr RowMajor(Index rows, Index columns, LongIndex leadingDimension)
        : shape_(rows, columns), stride_(leadingDimension, LongIndex(1)) {}

    FTSELF_GEMV_HOST_DEVICE constexpr Index shape(int index) const { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Index &shape(int index) { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Shape shape() const { return shape_; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex stride(int index) const { return stride_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex &stride(int index) { return stride_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Stride stride() const { return stride_; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex Capacity() const { return LongIndex(shape_[0]) * stride_[0]; }

    template <class Element>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr RowMajor MakeLayout(Index rows, Index columns) {
        return RowMajor(rows, columns);
    }

    template <class Element, class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr RowMajor MakeLayoutInUb(Coordinate const &tileShape) {
        constexpr uint32_t elementsPerBlock = FTSelf::Gemv::BYTE_PER_BLK / sizeof(Element);
        return RowMajor(tileShape[0], tileShape[1], FTSelf::helper::RoundUp(tileShape[1], elementsPerBlock));
    }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex GetOffset(Coordinate const &coordinate) const {
        return LongIndex(coordinate[0]) * stride_[0] + LongIndex(coordinate[1]);
    }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr RowMajor GetTileLayout(Coordinate const &tileShape) const {
        return RowMajor(tileShape[0], tileShape[1], stride_[0]);
    }
};

struct ColumnMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    Shape shape_;
    Stride stride_;

    FTSELF_GEMV_HOST_DEVICE constexpr ColumnMajor(Index rows = 0, Index columns = 0)
        : shape_(rows, columns), stride_(LongIndex(1), LongIndex(rows)) {}

    FTSELF_GEMV_HOST_DEVICE constexpr ColumnMajor(Index rows, Index columns, LongIndex leadingDimension)
        : shape_(rows, columns), stride_(LongIndex(1), leadingDimension) {}

    FTSELF_GEMV_HOST_DEVICE constexpr Index shape(int index) const { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Index &shape(int index) { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Shape shape() const { return shape_; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex stride(int index) const { return stride_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex &stride(int index) { return stride_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Stride stride() const { return stride_; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex Capacity() const { return LongIndex(shape_[1]) * stride_[1]; }

    template <class Element>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr ColumnMajor MakeLayout(Index rows, Index columns) {
        return ColumnMajor(rows, columns);
    }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex GetOffset(Coordinate const &coordinate) const {
        return LongIndex(coordinate[0]) + LongIndex(coordinate[1]) * stride_[1];
    }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr ColumnMajor GetTileLayout(Coordinate const &tileShape) const {
        return ColumnMajor(tileShape[0], tileShape[1], stride_[1]);
    }
};

struct VectorLayout {
    static constexpr int RANK = 1;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;
    using TensorCoord = Coord<RANK, Index>;

    Shape shape_;
    Stride stride_;

    FTSELF_GEMV_HOST_DEVICE constexpr explicit VectorLayout(Index size = 0)
        : shape_(size), stride_(LongIndex(1)) {}

    FTSELF_GEMV_HOST_DEVICE constexpr Index shape(int index) const { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Index &shape(int index) { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Shape shape() const { return shape_; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex stride(int index) const { return stride_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr LongIndex &stride(int index) { return stride_[index]; }
    FTSELF_GEMV_HOST_DEVICE constexpr Stride stride() const { return stride_; }

    template <class Element, class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr VectorLayout MakeLayoutInUb(Coordinate const &tileShape) {
        return VectorLayout(tileShape[0]);
    }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex GetOffset(Coordinate const &coordinate) const {
        return stride_[0] * coordinate[0];
    }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr VectorLayout GetTileLayout(Coordinate const &tileShape) const {
        return VectorLayout(tileShape[0]);
    }
};

template <class Derived>
struct FractalLayoutBase {
    static constexpr int RANK = 4;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using OrgShape = Coord<2, Index>;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    OrgShape orgShape_;
    Shape shape_;
    Stride stride_;

    constexpr FractalLayoutBase(Index orgRows = 0, Index orgCols = 0,
        Index rowsInFractal = 0, Index rowsByFractal = 0,
        Index colsInFractal = 0, Index colsByFractal = 0,
        LongIndex strideRowsInFractal = 0, LongIndex strideRowsByFractal = 0,
        LongIndex strideColsInFractal = 0, LongIndex strideColsByFractal = 0)
        : orgShape_(orgRows, orgCols),
          shape_(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
          stride_(strideRowsInFractal, strideRowsByFractal,
                  strideColsInFractal, strideColsByFractal) {}

    constexpr FractalLayoutBase(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    FTSELF_GEMV_HOST_DEVICE

    constexpr Index orgShape(int index) const { return orgShape_[index]; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Index &orgShape(int index) { return orgShape_[index]; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Shape shape() const { return shape_; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Shape &shape() { return shape_; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Index shape(int index) const { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Index &shape(int index) { return shape_[index]; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Stride stride() const { return stride_; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr Stride &stride() { return stride_; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex stride(int index) const { return stride_[index]; }
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex &stride(int index) { return stride_[index]; }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex GetOffset(Coordinate const &coord) const {
        return LongIndex(coord[0]) % shape_[0] * stride_[0] +
            LongIndex(coord[0]) / shape_[0] * stride_[1] +
            LongIndex(coord[1]) % shape_[2] * stride_[2] +
            LongIndex(coord[1]) / shape_[2] * stride_[3];
    }
};

struct nZ : FractalLayoutBase<nZ> {
    using Base = FractalLayoutBase<nZ>;
    using Base::Base;
    template <class Element>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr nZ MakeLayout(Index rows, Index cols) {
        constexpr Index c0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);
        constexpr Index fractal = FTSelf::Gemv::BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = FTSelf::helper::RoundUp(rows, c0);
        Index colsRound = FTSelf::helper::RoundUp(cols, FTSelf::Gemv::C0_NUM_PER_FRACTAL);
        return nZ(rows, cols, c0, rowsRound / c0, FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            colsRound / FTSelf::Gemv::C0_NUM_PER_FRACTAL, 1, colsRound * c0, c0, fractal);
    }
    constexpr LongIndex Capacity() const { return stride_[1] * shape_[1]; }
};

struct zN : FractalLayoutBase<zN> {
    using Base = FractalLayoutBase<zN>;
    using Base::Base;
    template <class Element>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr zN MakeLayout(Index rows, Index cols) {
        constexpr Index c0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);
        constexpr Index fractal = FTSelf::Gemv::BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = FTSelf::helper::RoundUp(rows, FTSelf::Gemv::C0_NUM_PER_FRACTAL);
        Index colsRound = FTSelf::helper::RoundUp(cols, c0);
        return zN(rows, cols, FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            rowsRound / FTSelf::Gemv::C0_NUM_PER_FRACTAL, c0, colsRound / c0,
            c0, fractal, 1, rowsRound * c0);
    }
    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE static constexpr zN MakeLayoutInL0C(Coordinate const &shape) {
        Index rows = shape[0];
        Index cols = shape[1];
        Index rowsRound = FTSelf::helper::RoundUp(rows, FTSelf::Gemv::C0_NUM_PER_FRACTAL);
        Index colsRound = FTSelf::helper::RoundUp(cols, FTSelf::Gemv::C0_NUM_PER_FRACTAL);
        return zN(rows, cols, FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            rowsRound / FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            colsRound / FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            FTSelf::Gemv::C0_NUM_PER_FRACTAL * FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            1, rowsRound * FTSelf::Gemv::C0_NUM_PER_FRACTAL);
    }
    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE constexpr zN GetTileLayout(Coordinate const &tileShape) const {
        return zN(tileShape[0], tileShape[1], shape_[0],
            FTSelf::helper::CeilDiv(tileShape[0], shape_[0]), shape_[2],
            FTSelf::helper::CeilDiv(tileShape[1], shape_[2]),
            stride_[0], stride_[1], stride_[2], stride_[3]);
    }
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex Capacity() const { return stride_[3] * shape_[3]; }
};

struct zZ : FractalLayoutBase<zZ> {
    using Base = FractalLayoutBase<zZ>;
    using Base::Base;
    template <class Element>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr zZ MakeLayout(Index rows, Index cols) {
        constexpr Index c0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);
        constexpr Index fractal = FTSelf::Gemv::BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = FTSelf::helper::RoundUp(rows, FTSelf::Gemv::C0_NUM_PER_FRACTAL);
        Index colsRound = FTSelf::helper::RoundUp(cols, c0);
        return zZ(rows, cols, FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            rowsRound / FTSelf::Gemv::C0_NUM_PER_FRACTAL, c0, colsRound / c0,
            c0, colsRound * FTSelf::Gemv::C0_NUM_PER_FRACTAL, 1, fractal);
    }
    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex GetOffset(Coordinate const &coord) const {
        return LongIndex(coord[0]) / shape_[0] * stride_[1] +
            LongIndex(coord[1]) / shape_[2] * stride_[3];
    }
    FTSELF_GEMV_HOST_DEVICE
    constexpr LongIndex Capacity() const { return stride_[1] * shape_[1]; }
};

struct nN : FractalLayoutBase<nN> {
    using Base = FractalLayoutBase<nN>;
    using Base::Base;
    template <class Element>
    FTSELF_GEMV_HOST_DEVICE
    static constexpr nN MakeLayout(Index rows, Index cols) {
        constexpr Index c0 = FTSelf::Gemv::BYTE_PER_C0 / sizeof(Element);
        constexpr Index fractal = FTSelf::Gemv::BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = FTSelf::helper::RoundUp(rows, c0);
        Index colsRound = FTSelf::helper::RoundUp(cols, FTSelf::Gemv::C0_NUM_PER_FRACTAL);
        return nN(rows, cols, c0, rowsRound / c0, FTSelf::Gemv::C0_NUM_PER_FRACTAL,
            colsRound / FTSelf::Gemv::C0_NUM_PER_FRACTAL, 1, fractal, c0,
            rowsRound * FTSelf::Gemv::C0_NUM_PER_FRACTAL);
    }
    constexpr LongIndex Capacity() const { return stride_[3] * shape_[3]; }
};

struct PaddingRowMajor : RowMajor { using RowMajor::RowMajor; };
struct PaddingColumnMajor : ColumnMajor { using ColumnMajor::ColumnMajor; };

} // namespace FTSelf::Gemv::layout

namespace FTSelf::helper {

namespace detail {

struct Coordinate1D {
    uint32_t length;

    FTSELF_GEMV_HOST_DEVICE uint32_t operator[](int) const { return length; }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE operator Coordinate() const
    {
        return Coordinate(length);
    }
};

struct Coordinate2D {
    uint32_t row;
    uint32_t column;

    FTSELF_GEMV_HOST_DEVICE uint32_t operator[](int index) const { return index == 0 ? row : column; }

    template <class Coordinate>
    FTSELF_GEMV_HOST_DEVICE operator Coordinate() const
    {
        return Coordinate(row, column);
    }
};

template <class Layout>
struct LayoutStorage {
    typename Layout::Index shape[Layout::RANK];
    typename Layout::LongIndex stride[Layout::RANK];
};

template <class Layout>
FTSELF_GEMV_HOST_DEVICE LayoutStorage<Layout> const &GetLayoutStorage(Layout const &layout)
{
    static_assert(sizeof(LayoutStorage<Layout>) == sizeof(Layout), "Unsupported layout storage");
    static_assert(alignof(LayoutStorage<Layout>) == alignof(Layout), "Unsupported layout alignment");
    return reinterpret_cast<LayoutStorage<Layout> const &>(layout);
}

} // namespace detail

template <class Layout>
FTSELF_GEMV_HOST_DEVICE typename Layout::Index GetShape(Layout const &layout, int index)
{
    return detail::GetLayoutStorage(layout).shape[index];
}

template <class Layout>
FTSELF_GEMV_HOST_DEVICE typename Layout::LongIndex GetStride(Layout const &layout, int index)
{
    return detail::GetLayoutStorage(layout).stride[index];
}

template <class Layout>
FTSELF_GEMV_HOST_DEVICE auto MakeTileLayout2D(Layout const &layout, uint32_t rows, uint32_t columns)
{
    return layout.GetTileLayout(detail::Coordinate2D{rows, columns});
}

template <class Layout>
FTSELF_GEMV_HOST_DEVICE auto MakeTileLayout1D(Layout const &layout, uint32_t length)
{
    return layout.GetTileLayout(detail::Coordinate1D{length});
}

template <class Layout>
FTSELF_GEMV_HOST_DEVICE auto MakeL0CLayout(uint32_t rows, uint32_t columns)
{
    return Layout::MakeLayoutInL0C(detail::Coordinate2D{rows, columns});
}

template <class Layout>
FTSELF_GEMV_HOST_DEVICE int64_t GetOffset2D(Layout const &layout, uint32_t row, uint32_t column)
{
    return layout.GetOffset(detail::Coordinate2D{row, column});
}

} // namespace FTSelf::helper

#endif // FTSELF_HELPER_LAYOUT_HELPER_HPP
