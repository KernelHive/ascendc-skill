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
 * \file conv2_d_backprop_filter_v3_tiling.cpp
 * \brief
 */
#include "conv2d_backprop_filter_v3_tiling.h"
#include "op_log.h"
#include "tiling/tiling_templates_registry.h"
#include "conv2dbp_adapt_to_conv3dbp.h"

namespace {
using Conv3DBackpropFilterV2CompileInfo = optiling::Conv3DCompileInfo;
constexpr int32_t KERNEL_HW_4 = 4;
constexpr int32_t KERNEL_HW_9 = 9;
constexpr int32_t KERNEL_HW_16 = 16;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
uint32_t WORKSPACE_NUM = 0U;
}  // namespace

namespace optiling {
ge::graphStatus Conv2DBackpropFilterV3Tiling::DoLibApiTiling()
{
    enableDeterministic_ = false;
    SetDwTilingFromTbeTiling();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Conv2DBackpropFilterV3", Conv2DBackpropFilterV3Tiling, 1);

static ge::graphStatus Conv2DBackpropFilterV3TilingFunc(gert::TilingContext *context)
{
    return AdaptTilingToConv3DBackprop(context, "Conv2DBackpropFilterV3");
}

static ge::graphStatus TilingParseForConv2DBackpropFilterV3(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto compileInfoPtr = context->GetCompiledInfo<Conv3DBackpropFilterV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
    plaformInstance.SetInstance(*compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv2DBackpropFilterV3)
    .Tiling(Conv2DBackpropFilterV3TilingFunc)
    .TilingParse<Conv3DBackpropFilterV2CompileInfo>(TilingParseForConv2DBackpropFilterV3);
} // namespace optiling