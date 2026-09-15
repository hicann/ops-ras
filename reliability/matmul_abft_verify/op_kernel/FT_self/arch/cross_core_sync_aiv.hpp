/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_FT_FTSELF_ARCH_CROSS_CORE_SYNC_AIV_HPP
#define MATMUL_FT_FTSELF_ARCH_CROSS_CORE_SYNC_AIV_HPP

#include "../core/macros.hpp"

namespace FTSelf {

namespace Arch {

constexpr uint32_t MAX_REVERSE_DEPTH = 15;
using FlagID = uint16_t;

template <uint32_t ReverseDepth = MAX_REVERSE_DEPTH>
struct CrossCoreFlagWithReverse {
    FlagID id;
    FlagID reverseId;
    uint32_t count{0};

    FTSELF_DEVICE CrossCoreFlagWithReverse(FlagID id = 0, FlagID reverseId = 0)
        : id(id), reverseId(reverseId) {}
};

template <uint8_t Mode, pipe_t Pipe, uint32_t ReverseDepth>
FTSELF_DEVICE void CrossCoreSetFlagWithReverse(CrossCoreFlagWithReverse<ReverseDepth> &flag)
{
    AscendC::CrossCoreSetFlag<Mode, Pipe>(flag.id);
    if (++flag.count >= ReverseDepth) {
        AscendC::CrossCoreWaitFlag(flag.reverseId);
        flag.count = 0;
    }
}

template <uint8_t Mode, pipe_t Pipe, uint32_t ReverseDepth>
FTSELF_DEVICE void CrossCoreWaitFlagWithReverse(CrossCoreFlagWithReverse<ReverseDepth> &flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
    if (++flag.count >= ReverseDepth) {
        AscendC::CrossCoreSetFlag<Mode, Pipe>(flag.reverseId);
        flag.count = 0;
    }
}

} // namespace Arch

template <uint8_t MODE, pipe_t PIPE>
FTSELF_DEVICE
void CrossCoreBarrierAIC()
{
    constexpr Arch::FlagID flagId = 9;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
FTSELF_DEVICE
void CrossCoreBarrierAIV()
{
    constexpr Arch::FlagID flagId = MODE == 0x1 ? 10 : 8;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

}  // namespace FTSelf

#endif  // MATMUL_FT_FTSELF_ARCH_CROSS_CORE_SYNC_AIV_HPP
