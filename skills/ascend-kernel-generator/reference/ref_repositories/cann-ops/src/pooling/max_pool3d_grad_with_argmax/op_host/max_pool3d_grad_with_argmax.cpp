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
 * @file max_pool3d_grad_with_argmax.cpp
 */
#include <cstdint>
#include <map>
#include <unordered_map>
#include "max_pool3d_grad_with_argmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

// optiling
namespace optiling {

static ge::graphStatus Tiling4MaxPool3DGradWithArgmax(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MaxPool3DGradWithArgmax(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4MaxPool3DGradWithArgmax.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<Tiling4MaxPool3DGradWithArgmaxCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPool3DGradWithArgmax)
    .Tiling(Tiling4MaxPool3DGradWithArgmax)
    .TilingParse<Tiling4MaxPool3DGradWithArgmaxCompileInfo>(TilingPrepare4MaxPool3DGradWithArgmax);
} // namespace optiling

namespace optiling {
namespace vectorutil {
namespace {
const std::map<ge::DataType, int64_t> DTYPE_LEN_MAP {
    {ge::DataType::DT_INT64, 8},
    {ge::DataType::DT_UINT64, 8},
    {ge::DataType::DT_FLOAT, 4},
    {ge::DataType::DT_INT32, 4},
    {ge::DataType::DT_UINT32, 4},
    {ge::DataType::DT_FLOAT16, 2},
    {ge::DataType::DT_INT16, 2},
    {ge::DataType::DT_UINT16, 2},
    {ge::DataType::DT_INT8, 1},
    {ge::DataType::DT_UINT8, 1},
    {ge::DataType::DT_BOOL, 1},
};

constexpr int64_t BLOCK_SIZE_BYTES = 32;
constexpr int64_t ELEMENT_IN_BLOCK_DEFAULT = 16;
}

int64_t GetElementByType(const ge::DataType& dtype) {
  if (DTYPE_LEN_MAP.find(dtype) != DTYPE_LEN_MAP.end()) {
    return BLOCK_SIZE_BYTES / DTYPE_LEN_MAP.at(dtype);
  }
  return ELEMENT_IN_BLOCK_DEFAULT;
}

int64_t CeilDiv(const int64_t dividend, const int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (dividend + divisor - 1) / divisor;
}

int64_t FloorAlign(const int64_t dividend, const int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return dividend / divisor * divisor;
}
int64_t CeilAlign(const int64_t dividend, const int64_t divisor) {
  return CeilDiv(dividend, divisor) * divisor;
}
} // namespace vectorutil
} // namespace optiling