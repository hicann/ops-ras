/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_CORE_GEMM_TYPE_HPP
#define FTSELF_CORE_GEMM_TYPE_HPP

#include "../core/arch.hpp"
#include "../helper/type_helper.hpp"

namespace FTSelf {

namespace Epilogue {}

// FTSelf extends both Gemm and Gemv with its own Block/Tile/helper namespaces,
// so these must be real namespaces rather than namespace aliases.
namespace Gemm {
template <class ArchTag_, bool Async_ = false>
struct MmadBase {
    using ArchTag = ArchTag_;
    static constexpr bool ASYNC = Async_;
};

using MmadAtlasA2 = MmadBase<Arch::AtlasA2, false>;

template <bool EnableUnitFlag = false>
struct MmadAtlasA2Pingpong : MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = EnableUnitFlag;
};
} // namespace Gemm

template <class Element_, class Layout_, AscendC::TPosition Position_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition POSITION = Position_;
};

namespace Gemm {
using FTSelf::GemmType;
} // namespace Gemm

namespace Gemv {
using FTSelf::GemmType;
} // namespace Gemv

namespace core {
template <class ElementA, class ElementB>
using ElementAccumulatorSelector = FTSelf::helper::ElementAccumulatorSelector<ElementA, ElementB>;
} // namespace core

} // namespace FTSelf

#endif // FTSELF_CORE_GEMM_TYPE_HPP
