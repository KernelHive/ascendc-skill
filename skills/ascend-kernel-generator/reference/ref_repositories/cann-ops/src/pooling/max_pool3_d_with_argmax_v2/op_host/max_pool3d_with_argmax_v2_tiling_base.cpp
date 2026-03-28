/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_with_argmax_v2_tiling_base.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "max_pool3_d_with_argmax_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

const int INPUT_IDX_X = 0;
const int NCDHW_DIMS = 5;
const int KERNEL_POS = 0;
const int STRIDE_POS = 1;
const int PADDING_POS = 2;
const int DILATION_POS = 3;
const int CEIL_POS = 4;
const int WS_SYS_SIZE = 16 * 1024 * 1024;

ge::graphStatus MaxPool3DWithArgmaxV2BaseTiling::GetPlatformInfo() {
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MaxPool3DWithArgmaxV2CompileInfo*>(context_->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;

        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNum();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    }
    OP_TILING_CHECK(coreNum == 0, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "coreNum is 0"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseTiling::GetShapeAttrsInfo() {
    auto inputX = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = ops::EnsureNotScalar(inputX->GetStorageShape());
    if (inputShape.GetDimNum() != NCDHW_DIMS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: input shape dim = %zu, should be equal 5",
                                        inputShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3DWithArgmaxV2: invalid dtype");
        return ge::GRAPH_FAILED;
    }

    auto outX = context_->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = ops::EnsureNotScalar(outX->GetStorageShape());
    
    auto indicesX = context_->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesX);        
    auto indicesShape = ops::EnsureNotScalar(indicesX->GetStorageShape());
    if (indicesShape != outShape) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3DWithArgmaxV2: indices shape and values shape are different");
        return ge::GRAPH_FAILED;
    }

    // Calculate number of batches
    int d_dim = 1;
    int h_dim = 2;
    int w_dim = 3;
    inputData.batches = inputShape.GetDim(0);
    if (inputShape.GetDimNum() == NCDHW_DIMS) {
        inputData.batches *= inputShape.GetDim(1);
        d_dim++;
        h_dim++;
        w_dim++;
    }
    inputData.inputShape = {uint64_t(inputShape.GetDim(d_dim)), uint64_t(inputShape.GetDim(h_dim)), uint64_t(inputShape.GetDim(w_dim))};
    inputData.outShape = {uint64_t(outShape.GetDim(d_dim)), uint64_t(outShape.GetDim(h_dim)), uint64_t(outShape.GetDim(w_dim))};

    auto runtimeAttrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    const gert::TypedContinuousVector<int64_t> *kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    inputData.kernelSize = {static_cast<uint64_t>(kernelSize->GetData()[D_DIM]),
                            static_cast<uint64_t>(kernelSize->GetData()[H_DIM]),
                            static_cast<uint64_t>(kernelSize->GetData()[W_DIM])};

    const gert::TypedContinuousVector<int64_t> *stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stride);
    inputData.stride = {static_cast<uint64_t>(stride->GetData()[D_DIM]),
                        static_cast<uint64_t>(stride->GetData()[H_DIM]),
                        static_cast<uint64_t>(stride->GetData()[W_DIM])};

    const gert::TypedContinuousVector<int64_t> *padding = runtimeAttrs->GetListInt(PADDING_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padding);
    inputData.pad = {static_cast<uint64_t>(padding->GetData()[D_DIM]),
                     static_cast<uint64_t>(padding->GetData()[H_DIM]),
                     static_cast<uint64_t>(padding->GetData()[W_DIM])};

    const gert::TypedContinuousVector<int64_t> *dilation = runtimeAttrs->GetListInt(DILATION_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dilation);
    inputData.dilation = {static_cast<uint64_t>(dilation->GetData()[D_DIM]),
                          static_cast<uint64_t>(dilation->GetData()[H_DIM]),
                          static_cast<uint64_t>(dilation->GetData()[W_DIM])};

    inputData.ceilMode = *runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DWithArgmaxV2BaseTiling::IsCapable() {
    return true;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseTiling::DoOpTiling() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseTiling::DoLibApiTiling() {
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DWithArgmaxV2BaseTiling::GetTilingKey() const {
    return 0;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseTiling::GetWorkspaceSize() {
    auto sys_workspace = WS_SYS_SIZE;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseTiling::PostTiling() {
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling