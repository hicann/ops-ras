/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_CORE_COORD_HPP
#define FTSELF_CORE_COORD_HPP

#include <cstdint>
#include "../core/macros.hpp"

namespace FTSelf {

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;
constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

enum class Status { kSuccess, kInvalid };

template <int Rank, class IndexType = uint32_t>
struct Coord {
    using Index = IndexType;
    static constexpr int RANK = Rank;
    Index data[Rank];

    FTSELF_HOST_DEVICE constexpr Coord() : data{} {}
    template <class... Values>
    FTSELF_HOST_DEVICE constexpr explicit Coord(Values... values) : data{Index(values)...} {}
    FTSELF_HOST_DEVICE constexpr Index const &operator[](int index) const { return data[index]; }
    FTSELF_HOST_DEVICE constexpr Index &operator[](int index) { return data[index]; }
    FTSELF_HOST_DEVICE constexpr Index const &At(int index) const { return data[index]; }
    FTSELF_HOST_DEVICE constexpr Index &At(int index) { return data[index]; }
};

template <class... Values>
FTSELF_HOST_DEVICE constexpr auto MakeCoord(Values... values)
{
    return Coord<sizeof...(Values)>{values...};
}

struct MatrixCoord : Coord<2> {
    using Base = Coord<2>;
    FTSELF_HOST_DEVICE constexpr MatrixCoord(uint32_t row = 0, uint32_t column = 0) : Base(row, column) {}
    FTSELF_HOST_DEVICE constexpr MatrixCoord(Base const &coord) : Base(coord) {}
    FTSELF_HOST_DEVICE constexpr uint32_t const &row() const { return data[0]; }
    FTSELF_HOST_DEVICE constexpr uint32_t &row() { return data[0]; }
    FTSELF_HOST_DEVICE constexpr uint32_t const &column() const { return data[1]; }
    FTSELF_HOST_DEVICE constexpr uint32_t &column() { return data[1]; }
};

struct GemvCoord : Coord<2> {
    using Base = Coord<2>;
    FTSELF_HOST_DEVICE constexpr GemvCoord(uint32_t m = 0, uint32_t n = 0) : Base(m, n) {}
    FTSELF_HOST_DEVICE constexpr GemvCoord(Base const &coord) : Base(coord) {}
    FTSELF_HOST_DEVICE constexpr uint32_t const &m() const { return data[0]; }
    FTSELF_HOST_DEVICE constexpr uint32_t &m() { return data[0]; }
    FTSELF_HOST_DEVICE constexpr uint32_t const &n() const { return data[1]; }
    FTSELF_HOST_DEVICE constexpr uint32_t &n() { return data[1]; }
};

struct GemmCoord : Coord<3> {
    using Base = Coord<3>;
    FTSELF_HOST_DEVICE constexpr GemmCoord(uint32_t m = 0, uint32_t n = 0, uint32_t k = 0) : Base(m, n, k) {}
    FTSELF_HOST_DEVICE constexpr GemmCoord(Base const &coord) : Base(coord) {}
    FTSELF_HOST_DEVICE constexpr uint32_t const &m() const { return data[0]; }
    FTSELF_HOST_DEVICE constexpr uint32_t &m() { return data[0]; }
    FTSELF_HOST_DEVICE constexpr uint32_t const &n() const { return data[1]; }
    FTSELF_HOST_DEVICE constexpr uint32_t &n() { return data[1]; }
    FTSELF_HOST_DEVICE constexpr uint32_t const &k() const { return data[2]; }
    FTSELF_HOST_DEVICE constexpr uint32_t &k() { return data[2]; }
};

template <uint32_t M_, uint32_t N_, uint32_t K_>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;
    static constexpr uint64_t MN = uint64_t(M) * N;
    static constexpr uint64_t MK = uint64_t(M) * K;
    static constexpr uint64_t NK = uint64_t(N) * K;
    static constexpr uint64_t MNK = MN * K;
    static constexpr uint64_t COUNT = MNK;
    FTSELF_HOST_DEVICE static constexpr GemmCoord ToCoord() { return GemmCoord(M, N, K); }
};

template <uint32_t M_, uint32_t N_>
struct GemvShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint64_t MN = uint64_t(M) * N;
    static constexpr uint64_t COUNT = MN;
    FTSELF_HOST_DEVICE static constexpr GemvCoord ToCoord() { return GemvCoord(M, N); }
};

template <uint32_t Rows, uint32_t Columns>
struct MatrixShape {
    static constexpr uint32_t ROW = Rows;
    static constexpr uint32_t COLUMN = Columns;
    static constexpr uint64_t COUNT = uint64_t(Rows) * Columns;
    FTSELF_HOST_DEVICE static constexpr MatrixCoord ToCoord() { return MatrixCoord(Rows, Columns); }
};

} // namespace FTSelf

#endif // FTSELF_CORE_COORD_HPP
