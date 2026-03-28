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
 * \file foreach_custom_tiling_func.cpp
 * \brief
 */
 
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_CUSTOM_FUNC_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_CUSTOM_FUNC_H_

#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "common_dtype.h"
#include "foreach_tiling_func.h"
#include "foreach_custom_tiling_def.h"

namespace optiling {

static ge::graphStatus TilingPrepare4ForeachCustomTiling(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ForeachCustomTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ABS_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(ForeachCustom)
.Tiling(Tiling4ForeachCustomTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachCustomTiling);
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_CUSTOM_FUNC_H_