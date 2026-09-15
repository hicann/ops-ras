/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_FTSELF_GEMM_TILE_TILE_MMAD_HPP
#define FTSELF_FTSELF_GEMM_TILE_TILE_MMAD_HPP

#include <cstdint>
#include <type_traits>

#include "../../core/coord.hpp"
#include "../../core/gemm_type.hpp"
#include "../../core/layout.hpp"
#include "../../core/macros.hpp"

namespace FTSelf::Gemm::Tile {

template <class ArchTag, class AType, class BType, class BiasType = void>
struct TileMmad {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename FTSelf::core::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    FTSELF_DEVICE
    TileMmad() {}

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
        AscendC::LocalTensor<ElementA> const &l0ATensor,
        AscendC::LocalTensor<ElementB> const &l0BTensor,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        bool initC = true,
        uint8_t unitFlag = 0)
    {
        AscendC::MmadParams params = MakeMmadParams(m, n, k, unitFlag);
        params.cmatrixInitVal = initC;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        if constexpr (std::is_same_v<ElementA, float> &&
            (std::is_same_v<typename AType::Layout, FTSelf::layout::ColumnMajor> ||
             std::is_same_v<typename AType::Layout, FTSelf::layout::nZ>)) {
            params.kDirectionAlign = true;
        }
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if constexpr (std::is_same_v<typename AType::Layout, FTSelf::layout::VectorLayout>) {
            params.disableGemv = false;
        } else {
            params.disableGemv = true;
        }
#endif
        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, params);
        BarrierIfNeeded(m, n);
    }

    FTSELF_DEVICE
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
        AscendC::LocalTensor<ElementA> const &l0ATensor,
        AscendC::LocalTensor<ElementB> const &l0BTensor,
        AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        bool initC = true,
        uint8_t unitFlag = 0)
    {
        AscendC::MmadParams params = MakeMmadParams(m, n, k, unitFlag);
        params.cmatrixInitVal = false;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        if constexpr (std::is_same_v<ElementA, float> &&
            (std::is_same_v<typename AType::Layout, FTSelf::layout::ColumnMajor> ||
             std::is_same_v<typename AType::Layout, FTSelf::layout::nZ> ||
             std::is_same_v<typename AType::Layout, FTSelf::layout::PaddingColumnMajor>)) {
            params.kDirectionAlign = true;
        }
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        params.disableGemv = true;
#endif
        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, params);
        BarrierIfNeeded(m, n);
    }

private:
    FTSELF_DEVICE
    static AscendC::MmadParams MakeMmadParams(uint32_t m, uint32_t n, uint32_t k,
        uint8_t unitFlag)
    {
        AscendC::MmadParams params;
        params.m = m;
        params.n = n;
        params.k = k;
        params.unitFlag = unitFlag;
        return params;
    }

    FTSELF_DEVICE
    static void BarrierIfNeeded(uint32_t m, uint32_t n)
    {
        constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / FTSelf::C0_NUM_PER_FRACTAL) *
                (n / FTSelf::C0_NUM_PER_FRACTAL) <
            PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_FTSELF_GEMM_TILE_TILE_MMAD_HPP
