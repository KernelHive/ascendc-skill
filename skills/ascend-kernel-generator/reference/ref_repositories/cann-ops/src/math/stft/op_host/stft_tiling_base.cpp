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
 * \file stft_tiling_base.cpp
 * \brief
 */
#include "tiling/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "stft_tiling_base.h"
#include "error_log.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr size_t INPUT_MAX_DIM_NUM = 2;

ge::graphStatus STFTBaseTiling::GetPlatformInfo() {
  auto platformPtr = context_->GetPlatformInfo();
  if (platformPtr == nullptr) {
    auto compileInfoPtr = reinterpret_cast<const STFTCompileInfo*>(context_->GetCompileInfo());
    coreNum = compileInfoPtr->coreNum;
    aivCoreNum = compileInfoPtr->aivCoreNum;
    aicCoreNum = compileInfoPtr->aicCoreNum;
    sysWorkspaceSize = compileInfoPtr->sysWorkspaceSize;
    ubSize = compileInfoPtr->ubSize;
    l1Size = compileInfoPtr->l1Size;
    l0ASize = compileInfoPtr->l0ASize;
    l0BSize = compileInfoPtr->l0BSize;
    l0CSize = compileInfoPtr->l0CSize;
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
    coreNum = ascendcPlatform.GetCoreNum();
    aicCoreNum = ascendcPlatform.GetCoreNumAic();
    aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    uint64_t ubSizePlatform;
    uint64_t l1SizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatform);
    ubSize = static_cast<int64_t>(ubSizePlatform);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatform);
    l1Size = static_cast<int64_t>(l1SizePlatform);

    uint64_t l0ASizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, l0ASizePlatform);
    l0ASize = static_cast<int64_t>(l0ASizePlatform);

    uint64_t l0BSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, l0BSizePlatform);
    l0BSize = static_cast<int64_t>(l0BSizePlatform);

    uint64_t l0CSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, l0CSizePlatform);
    l0CSize = static_cast<int64_t>(l0CSizePlatform);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus STFTBaseTiling::GetShapeAttrsInfo() {
  auto input = context_->GetInputDesc(0);
  dtype = input->GetDataType();
  if (dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT && dtype != ge::DataType::DT_COMPLEX64) {
    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "STFT: invalid dtype.");
    return ge::GRAPH_FAILED;
  }

  auto inputShape = context_->GetInputShape(0)->GetStorageShape();
  if (inputShape.GetDimNum() > INPUT_MAX_DIM_NUM) {
    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                    "STFT: input shape dim(=%zu) > 2, please check.",
                                    inputShape.GetDimNum());
    return ge::GRAPH_FAILED;
  }

  // 当前迭代window输入为权重矩阵与窗函数乘积之后的结果，先不做任何校验
  auto attrs = context_->GetAttrs();

  const int64_t* hopPtr = attrs->GetAttrPointer<int64_t>(0);
  hop = *hopPtr;
  if (hop <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "STFT: invalid hop attr.");
    return ge::GRAPH_FAILED;
  }

  const int64_t* winLengthPtr = attrs->GetAttrPointer<int64_t>(1);
  winLength = *winLengthPtr;
  if (winLength <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "STFT: invalid winLength attr.");
    return ge::GRAPH_FAILED;
  }

  const bool* normalizedPtr = attrs->GetAttrPointer<bool>(2);
  normalized = *normalizedPtr;

  const bool* onesidedPtr = attrs->GetAttrPointer<bool>(3);
  onesided = *onesidedPtr;

  const bool* returnComplexPtr = attrs->GetAttrPointer<bool>(4);
  returnComplex = *returnComplexPtr;

  const int64_t* nfftPtr = attrs->GetAttrPointer<int64_t>(5);
  nfft = *nfftPtr;
  if (nfft <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "STFT: invalid nfft attr.");
    return ge::GRAPH_FAILED;
  }

  if (inputShape.GetDimNum() == INPUT_MAX_DIM_NUM) {
    batch = inputShape.GetDim(0);
    inputSize = inputShape.GetDim(1) - nfft;  // 调用到kernel时，已经对尾轴补pad，因此需要去掉pad的长度
  } else {
    batch = 1;
    inputSize = inputShape.GetDim(0) - nfft;
  }

  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
