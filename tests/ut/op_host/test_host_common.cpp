/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "op_host/hash.h"

namespace {
using Ops::Ras::HostTiling::MurmurHash;

TEST(RasHostHash, IsDeterministicForTheSameInput)
{
    constexpr std::array<uint32_t, 2> input = {0x01234567U, 0x89abcdefU};

    EXPECT_EQ(MurmurHash(input.data(), sizeof(input)), MurmurHash(input.data(), sizeof(input)));
}

TEST(RasHostHash, ChangesWhenThePayloadChanges)
{
    constexpr std::array<uint32_t, 2> first = {0x01234567U, 0x89abcdefU};
    constexpr std::array<uint32_t, 2> second = {0x01234567U, 0x89abcdeeU};

    EXPECT_NE(MurmurHash(first.data(), sizeof(first)), MurmurHash(second.data(), sizeof(second)));
}

TEST(RasHostHash, ChangesWhenTheSeedChanges)
{
    constexpr std::array<uint32_t, 2> input = {0x01234567U, 0x89abcdefU};

    EXPECT_NE(MurmurHash(input.data(), sizeof(input), 1U), MurmurHash(input.data(), sizeof(input), 2U));
}
}  // namespace
