/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <iostream>

#include "tiling_case_executor.h"

class AddExampleTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddExampleTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddExampleTiling TearDown" << std::endl;
    }
};

TEST_F(AddExampleTiling, add_example_float32_success)
{
    struct AddExampleCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AddExample",
        {
            {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo, 48, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "256 8 ", {16U * 1024U * 1024U});
}

TEST_F(AddExampleTiling, add_example_int32_success)
{
    struct AddExampleCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AddExample",
        {
            {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {}, &compileInfo, 48, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, "256 8 ", {16U * 1024U * 1024U});
}
