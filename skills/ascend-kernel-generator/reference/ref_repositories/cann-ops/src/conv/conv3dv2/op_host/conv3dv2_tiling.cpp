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
 * \file conv3dv2_tiling.cpp
 * \brief
 */

#include "conv3dv2_tiling.h"
#include "conv3d_base_tiling.h"
 
#include "tiling/tiling_templates_registry.h"
#include "register/op_def_registry.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr)   \
  do {                                          \
    if (cond) {                                 \
        std::printf(log_func);                  \
        expr;                                   \
    }                                           \
  } while (0)
 
using namespace optiling::conv3d_ops_tiling;
 
namespace optiling {
 
    REGISTER_TILING_TEMPLATE("Conv3DV2", Conv3dBaseTiling, 0);
 
    static ge::graphStatus Conv3DV2TilingFunc(gert::TilingContext* context)
    {
        OP_TILING_CHECK(context == nullptr,
                        "Conv3DV2, context is null",
        return ge::GRAPH_FAILED);
        OP_LOGD(context->GetNodeType(), "Begin process Conv3DV2TilingFunc");
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }
 
    static ge::graphStatus TilingPrepareForConv3DV2(gert::TilingParseContext *context) {
        OP_TILING_CHECK(context == nullptr,
                        "Conv3DV2, context is null",
        return ge::GRAPH_FAILED);
        fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
        OP_TILING_CHECK(platformInfo == nullptr,
                        "platformInfoPtr is null",
        return ge::GRAPH_FAILED);
 
        auto compileInfoPtr = context->GetCompiledInfo<optiling::Conv3DTilingParseInfo>();
        OP_TILING_CHECK(compileInfoPtr == nullptr,
                        "compileInfoPtr is null",
        return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfoPtr->aicoreNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
        ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Rate);
 
        OP_LOGD(context->GetNodeName(),
                " l1Size:%lu, l2Size:%lu, coreNum:%u"
                "%lu, %lu, %lu, %lu",
                compileInfoPtr->l1Size,
                compileInfoPtr->l2Size,
                compileInfoPtr->aicoreNum,
                compileInfoPtr->l0aSize,
                compileInfoPtr->l0bSize,
                compileInfoPtr->l0cSize,
                compileInfoPtr->l2Rate);
        return ge::GRAPH_SUCCESS;
    }
 
    IMPL_OP_OPTILING(Conv3DV2)
    .Tiling(Conv3DV2TilingFunc)
    .TilingParse<optiling::Conv3DTilingParseInfo>(TilingPrepareForConv3DV2);
}