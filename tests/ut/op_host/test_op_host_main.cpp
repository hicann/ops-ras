/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>

#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_info.h"

class OpHostUtEnvironment : public testing::Environment {
public:
    void SetUp() override
    {
        fe::OptionalInfos compilationInfo;
        compilationInfo.Init();
        compilationInfo.SetSocVersion("soc_version");
        fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(compilationInfo);

        auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(registry);
    }

    void TearDown() override
    {
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
    }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if (testing::UnitTest::GetInstance()->total_test_count() == 0) {
        std::cerr << "No RAS OpHost tests were discovered." << std::endl;
        return 1;
    }
    testing::AddGlobalTestEnvironment(new OpHostUtEnvironment());
    return RUN_ALL_TESTS();
}
