/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_CORE_MACROS_HPP
#define FTSELF_CORE_MACROS_HPP

#if defined(__CCE__)
#include <kernel_operator.h>
#endif

#define FTSELF_DEVICE __forceinline__ __aicore__
#ifdef __CCE__
#define FTSELF_HOST_DEVICE __forceinline__ [host, aicore]
#else
#define FTSELF_HOST_DEVICE
#endif
#define FTSELF_KERNEL __global__ __aicore__

#endif // FTSELF_CORE_MACROS_HPP
