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
 * \file layer_norm_v4_tiling.cc
 * \brief
 */

#include "layer_norm_v4_tiling.h"

namespace optiling {

static ge::graphStatus Tiling4LayerNormV4(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4LayerNormV4(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4LayerNormV4 enter.");

    auto compileInfo = GetCompileInfoPtr<LayerNormV4CompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->isAscend310P = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P;
    OP_TILING_CHECK((compileInfo->coreNum <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_TILING_CHECK((compileInfo->ubSizePlatForm <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
            "Get ub size failed, ub size: %u",
            static_cast<uint32_t>(compileInfo->ubSizePlatForm)),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepare4LayerNormV4 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LayerNormV4).Tiling(Tiling4LayerNormV4).TilingParse<LayerNormV4CompileInfo>(TilingPrepare4LayerNormV4);

}  // namespace optiling
