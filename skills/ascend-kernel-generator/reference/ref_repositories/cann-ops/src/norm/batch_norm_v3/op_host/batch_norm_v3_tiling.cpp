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
 * \file batch_norm_v3_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "batch_norm_v3_tiling.h"
using namespace ge;

namespace optiling {

static ge::graphStatus Tiling4BatchNormV3(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4BatchNormV3(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4BatchNormV3 enter.");

    auto compileInfo = GetCompileInfoPtr<BatchNormV3CompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->coreNum <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
        "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    OP_TILING_CHECK((compileInfo->ubSize <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
        "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepare4BatchNormV3 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchNormV3)
    .Tiling(Tiling4BatchNormV3)
    .TilingParse<BatchNormV3CompileInfo>(TilingPrepare4BatchNormV3);

}  // namespace optiling
