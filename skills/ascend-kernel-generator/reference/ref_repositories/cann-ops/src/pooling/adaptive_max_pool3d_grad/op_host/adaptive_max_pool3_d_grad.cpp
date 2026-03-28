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
 * @file adaptive_max_pool3d_grad.cpp
 */
#define OPS_UTILS_LOG_SUB_MOD_NAME "OP_ADAPTIVEMAXPOOL3D"
#define OPS_UTILS_LOG_PACKAGE_TYPE "OP_ADAPTIVEMAXPOOL3D"

#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "adaptive_max_pool3d_grad_tiling.h"

using namespace AscendC;

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)

namespace optiling {

    static ge::graphStatus Tiling4AdaptiveMaxPool3DGrad(gert::TilingContext *context)
    {
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    
    static ge::graphStatus TilingPrepare4AdaptiveMaxPool3DGrad(gert::TilingParseContext *context)
    {
        OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4AdaptiveMaxPool3DGrad.");
        fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
        auto compileInfoPtr = context->GetCompiledInfo<Tiling4AdaptiveMaxPool3DGradCompileInfo>();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
        compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
        return ge::GRAPH_SUCCESS;
    }
    
IMPL_OP_OPTILING(AdaptiveMaxPool3DGrad)
    .Tiling(Tiling4AdaptiveMaxPool3DGrad);
} // namespace optiling
    
    
namespace ops {
    class AdaptiveMaxPool3DGrad : public OpDef {
     public:
      explicit AdaptiveMaxPool3DGrad(const char* name) : OpDef(name)
      {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Input("argmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});
    
        this->AICore().AddConfig("ascend910b");
      }
    };
    
OP_ADD(AdaptiveMaxPool3DGrad);
}

