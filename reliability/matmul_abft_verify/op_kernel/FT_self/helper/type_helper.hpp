/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_HELPER_TYPE_HELPER_HPP
#define FTSELF_HELPER_TYPE_HELPER_HPP

#include "../helper/base_helper.hpp"

namespace FTSelf::helper {

template <class...>
inline constexpr bool DEPENDENT_FALSE = false;

template <class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>,
        "Unsupported element accumulator selector, can not find the specialization.");
};

template <>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<float, float> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template <>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<int32_t, int32_t> {
    using ElementAccumulator = int32_t;
};

} // namespace FTSelf::helper

#endif // FTSELF_HELPER_TYPE_HELPER_HPP
