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
 * \file stft_tiling.cc
 * \brief
 */

#include "platform/platform_info.h"
#include "stft_tiling.h"
#include "tiling/tiling_templates_registry.h"

#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)

using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

static ge::graphStatus Tiling4STFT(gert::TilingContext* context) {
  // 初始化算子Tiling类
  return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4STFT(gert::TilingParseContext* context) {
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

  auto compileInfoPtr = context->GetCompiledInfo<STFTCompileInfo>();
  OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
  compileInfoPtr->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
  compileInfoPtr->aicCoreNum = ascendcPlatform.GetCoreNumAic();
  compileInfoPtr->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(STFT).Tiling(Tiling4STFT).TilingParse<STFTCompileInfo>(TilingPrepare4STFT);

} // namespace optiling
