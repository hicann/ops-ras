/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_DEVICE_KERNEL_ADAPTER_HPP
#define FTSELF_DEVICE_KERNEL_ADAPTER_HPP

#if defined(ENABLE_ASCENDC_DUMP)
#include "../debug.hpp"
#endif

namespace FTSelf {

template <class Operator>
FTSELF_KERNEL void KernelAdapter(typename Operator::Params params, GM_ADDR ptrDump = nullptr)
{
    Operator op;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}

template <class Operator>
FTSELF_KERNEL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr,
    GM_ADDR ptrDump = nullptr)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}

} // namespace FTSelf

#endif // FTSELF_DEVICE_KERNEL_ADAPTER_HPP
