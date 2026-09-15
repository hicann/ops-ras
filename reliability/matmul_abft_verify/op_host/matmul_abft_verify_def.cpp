/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_abft_verify_def.cpp
 * \brief MatmulAbftVerify operator definition.
 */
#include "register/op_def_registry.h"

namespace ops {
namespace {
const std::vector<ge::DataType> kPrecisionDtype = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
const std::vector<ge::DataType> kFloatDtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
const std::vector<ge::DataType> kUint8Dtype = {ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8};
const std::vector<ge::Format> kNdFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
} // namespace

class MatmulAbftVerify : public OpDef {
public:
    explicit MatmulAbftVerify(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType(kPrecisionDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType(kPrecisionDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
        this->Input("c")
            .ParamType(REQUIRED)
            .DataType(kFloatDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
        this->Input("checksum_weight")
            .ParamType(REQUIRED)
            .DataType(kPrecisionDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();

        this->Output("comp_row")
            .ParamType(REQUIRED)
            .DataType(kUint8Dtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();

        this->Attr("e_max").AttrType(OPTIONAL).Float(0.001);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "matmul_abft_verify");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(MatmulAbftVerify);
} // namespace ops
