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
#include "platform/platform_info.h"

class OpGraphUtEnvironment : public testing::Environment {
public:
    void SetUp() override
    {
        fe::OptionalInfos compilationInfo;
        compilationInfo.Init();
        compilationInfo.SetSocVersion("soc_version");
        fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(compilationInfo);
    }
};

TEST(OpGraphInfrastructure, DiscoversAtLeastOneTest)
{
    SUCCEED();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpGraphUtEnvironment());
    return RUN_ALL_TESTS();
}
