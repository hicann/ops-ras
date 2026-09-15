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
 * \file matmul_abft_verify_infershape.cpp
 * \brief MatmulAbftVerify shape and dtype inference.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr int64_t L1_TILE_N = 256;
constexpr int64_t COMP_PACK = 8;

ge::graphStatus SetVectorShape(gert::InferShapeContext* context, size_t index, int64_t length)
{
    gert::Shape* shape = context->GetOutputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    shape->SetDimNum(1);
    shape->SetDim(0, length);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShapeMatmulAbftVerify(gert::InferShapeContext* context)
{
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bShape);
    OP_CHECK_IF(aShape->GetDimNum() != 2 || bShape->GetDimNum() != 2,
        OP_LOGE(context->GetNodeName(), "MatmulAbftVerify expects rank-2 A and B"), return ge::GRAPH_FAILED);

    const int64_t m = aShape->GetDim(0);
    const int64_t k = aShape->GetDim(1);
    const int64_t bK = bShape->GetDim(0);
    const int64_t n = bShape->GetDim(1);
    OP_CHECK_IF(k != bK, OP_LOGE(context->GetNodeName(), "A.K must equal B.K"), return ge::GRAPH_FAILED);

    const int64_t splitN = (n + L1_TILE_N - 1) / L1_TILE_N;
    OP_CHECK_IF(SetVectorShape(context, 0, ((m + COMP_PACK - 1) / COMP_PACK) * splitN) !=
                    ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "failed to set comp_row shape"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMatmulAbftVerify(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulAbftVerify).InferShape(InferShapeMatmulAbftVerify).InferDataType(InferDataTypeMatmulAbftVerify);
} // namespace ops
