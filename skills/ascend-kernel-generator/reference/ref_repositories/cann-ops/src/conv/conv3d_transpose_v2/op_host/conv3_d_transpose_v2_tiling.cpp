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
 * \file conv3d_transpose_v2_tiling.cpp
 * \brief
 */

#include "../../conv3d_backprop_input_v2/op_host/conv3d_backprop_input_v2_tiling.h"

#include "op_log.h"
#include "tiling/tiling_templates_registry.h"

namespace {
using Conv3DTransposeV2CompileInfo = optiling::Conv3DBackPropInputCompileInfo;
}  // namespace

namespace optiling {
REGISTER_TILING_TEMPLATE("Conv3DTransposeV2", Conv3DTransposeV2Tiling, 0);

static ge::graphStatus Conv3DTransposeV2TilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForConv3DTransposeV2(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<Conv3DTransposeV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();

    optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
    plaformInstance.SetInstance(*compileInfoPtr);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv3DTransposeV2)
    .Tiling(Conv3DTransposeV2TilingFunc)
    .TilingParse<Conv3DTransposeV2CompileInfo>(TilingParseForConv3DTransposeV2);
}  // namespace optiling
