/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"

namespace l0op {
const aclTensor* Cast(const aclTensor* self, op::DataType /*dstDtype*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* /*y*/, aclOpExecutor* /*executor*/)
{
    return x;
}
}  // namespace l0op
