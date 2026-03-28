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
 * \file conv2d_backprop_input_v2_tiling.cc
 * \brief
 */

#include "conv2d_backprop_input_v2_tiling.h"
#include "op_log.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "cube_tiling_runtime.h"

namespace optiling {
REGISTER_TILING_TEMPLATE("Conv2DBackpropInputV2", Conv2DBackpropInputV2Tiling, 0);

static ge::graphStatus Conv2DBackpropInputV2TilingFunc(gert::TilingContext *context)
{
    return AdaptTilingToConv3DBp(context, "Conv2DBackpropInputV2");
}

static ge::graphStatus TilingParseForConv2DBackpropInputV2(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto compileInfoPtr = context->GetCompiledInfo<Conv3DBackPropInputCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->shortSocVersion = ascendcPlatform.GetSocVersion();
    optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
    plaformInstance.SetInstance(*compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv2DBackpropInputV2)
    .Tiling(Conv2DBackpropInputV2TilingFunc)
    .TilingParse<Conv3DBackPropInputCompileInfo>(TilingParseForConv2DBackpropInputV2);
}