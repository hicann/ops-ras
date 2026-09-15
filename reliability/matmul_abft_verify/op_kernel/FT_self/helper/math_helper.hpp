/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_HELPER_MATH_HELPER_HPP
#define FTSELF_HELPER_MATH_HELPER_HPP

#include <cstdint>

#include "../helper/base_helper.hpp"

namespace FTSelf::helper {

template <uint32_t Alignment, class T>
FTSELF_GEMV_HOST_DEVICE constexpr T RoundUp(T const &value)
{
    static_assert(Alignment != 0, "Alignment must not be zero");
    return (value + Alignment - 1) / Alignment * Alignment;
}

template <class T, class U>
FTSELF_GEMV_HOST_DEVICE constexpr auto RoundUp(T const &value, U const &alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

template <uint32_t Alignment, class T>
FTSELF_GEMV_HOST_DEVICE constexpr T RoundDown(T const &value)
{
    static_assert(Alignment != 0, "Alignment must not be zero");
    return value / Alignment * Alignment;
}

template <class T, class U>
FTSELF_GEMV_HOST_DEVICE constexpr auto RoundDown(T const &value, U const &alignment)
{
    if (alignment == 0) {
        return value;
    }
    return value / alignment * alignment;
}

template <uint32_t Divisor, class T>
FTSELF_GEMV_HOST_DEVICE constexpr T CeilDiv(T const &dividend)
{
    static_assert(Divisor != 0, "Divisor must not be zero");
    return (dividend + Divisor - 1) / Divisor;
}

template <class T, class U>
FTSELF_GEMV_HOST_DEVICE constexpr auto CeilDiv(T const &dividend, U const &divisor)
{
    if (divisor == 0) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

} // namespace FTSelf::helper

#endif // FTSELF_HELPER_MATH_HELPER_HPP
