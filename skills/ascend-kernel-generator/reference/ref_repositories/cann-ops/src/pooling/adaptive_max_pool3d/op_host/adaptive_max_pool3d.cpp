/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file adaptive_max_pool3d.cpp
 */
#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_ADAPTIVEMAXPOOL3D"
#define OPS_UTILS_LOG_PACKAGE_TYPE "OP_ADAPTIVEMAXPOOL3D"
#include "tiling/tiling_templates_registry.h"
#include "adaptive_max_pool3d_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
using namespace AscendC;

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)

namespace optiling {
constexpr uint64_t NCDHW_DIM_N = 0;
constexpr uint64_t NCDHW_DIM_C = 1;
constexpr uint64_t NCDHW_DIM_D = 2;
constexpr uint64_t NCDHW_DIM_H = 3;
constexpr uint64_t NCDHW_DIM_W = 4;
constexpr uint64_t OUTPUTSIZE_DIMW = 2;
constexpr uint64_t OUTPUTSIZE_DIM_MAX = 3;
constexpr uint64_t DIM_NUM_FIVE = 5;
constexpr uint64_t ASCENDC_WORKSPACE = 16 * 1024 * 1024;
bool AdaptiveMaxPool3dTilingBase::IsCapable() {
    return true;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::DoOpTiling() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::DoLibApiTiling() {
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveMaxPool3dTilingBase::GetTilingKey() const {
    return 0;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::GetPlatformInfo() {
    auto platformInfo = context_->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    input_.coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, input_.ubSizePlatForm);
    OP_TILING_CHECK(
        input_.coreNum <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "GetPlatformInfo get corenum <= 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::GetShapeAttrsInfo() {
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin.");

    auto inputX = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputXDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto xDtype = inputXDesc->GetDataType();
    OP_TILING_CHECK((xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "x datatype only support float, float16, bfloat16"),
                    return ge::GRAPH_FAILED);
    input_.xDtype = xDtype;
    gert::Shape xShape = ops::EnsureNotScalar(inputX->GetStorageShape());
    if (xShape.GetDimNum() == DIM_NUM_FIVE) {
    input_.N = xShape.GetDim(NCDHW_DIM_N);
    input_.C = xShape.GetDim(NCDHW_DIM_C);
    input_.Di = xShape.GetDim(NCDHW_DIM_D);
    input_.Hi = xShape.GetDim(NCDHW_DIM_H);
    input_.Wi = xShape.GetDim(NCDHW_DIM_W);
    } else {
    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "xShape dim number should be 5");
    return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(input_.N < 1 || input_.C < 1 || input_.Di < 1 || input_.Hi < 1 || input_.Wi < 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Invalid shape. Maybe empty tensor."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        input_.Di * input_.Hi * input_.Wi > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "no support for D*H*W of input greater than int32 max value"),
        return ge::GRAPH_FAILED);

    auto attrPtr = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    auto outputSizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputSizePtr);
    OP_TILING_CHECK(outputSizePtr->GetSize() != OUTPUTSIZE_DIM_MAX,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "the size of outputsize only support 3"),
                    return ge::GRAPH_FAILED);
    const int64_t* outputSize = reinterpret_cast<const int64_t*>(outputSizePtr->GetData());
    OP_TILING_CHECK(outputSize[0] <= 0 || outputSize[1] <= 0 || outputSize[OUTPUTSIZE_DIMW] <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "the value of outputsize should > 0"),
                    return ge::GRAPH_FAILED);
    input_.Do = outputSize[0];
    input_.Ho = outputSize[1];
    input_.Wo = outputSize[OUTPUTSIZE_DIMW];

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::GetWorkspaceSize() {
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = ASCENDC_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::PostTiling() {
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AdaptiveMaxPool3d(gert::TilingContext* context) {
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4AdaptiveMaxPool3d(gert::TilingParseContext* context) {
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AdaptiveMaxPool3d enter.");

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AdaptiveMaxPool3d end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaptiveMaxPool3d)
    .Tiling(Tiling4AdaptiveMaxPool3d);
}  // namespace optiling
    
namespace ops {
class AdaptiveMaxPool3d : public OpDef {
public:
    explicit AdaptiveMaxPool3d(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("output_size")
        .AttrType(REQUIRED)
        .ListInt();
    
    this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AdaptiveMaxPool3d);

} // namespace ops

