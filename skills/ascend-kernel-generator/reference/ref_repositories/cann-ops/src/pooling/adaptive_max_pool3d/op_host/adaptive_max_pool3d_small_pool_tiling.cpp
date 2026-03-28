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
 * \file adaptive_max_pool3d_small_pool_tiling.cc
 * \brief
 */

#include "adaptive_max_pool3d_tiling.h"

constexpr uint64_t TILING_KEY_SMALL_KERNEL_FLOAT = 320000;
constexpr uint64_t TILING_KEY_SMALL_KERNEL_HALF = 321000;
constexpr uint64_t TILING_KEY_SMALL_KERNEL_BF16 = 322000;
constexpr uint64_t KERNEL_SIZE_LIMIT = 128;
constexpr uint64_t BLOCK_LEN = 256;
constexpr uint64_t KERNEL_W_LIMIT = 16;
constexpr uint64_t KERNEL_OUT_SIZE_LIMIT = 32;
constexpr uint64_t CAL_KER_THRESHOLD = 10000;

using namespace AscendC;

namespace optiling {
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
bool AdaptiveMaxPool3dSmallPoolTiling::IsCapable() {
  // 按照搬运对齐的大小全载UB 和 kernelW<=16, 判断是否走当前模板
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dSmallPoolTiling IsCapable check.");
  calInfo_.kernelDMax = CalKernelMax(input_.Di, input_.Do);
  calInfo_.kernelHMax = CalKernelMax(input_.Hi, input_.Ho);
  calInfo_.kernelWMax = CalKernelMax(input_.Wi, input_.Wo);
  auto kernelWMaxAlign = (calInfo_.kernelWMax + 7) / 8 * 8;
  bool limitSizeMax = (calInfo_.kernelDMax * calInfo_.kernelHMax * kernelWMaxAlign <= KERNEL_SIZE_LIMIT);
  bool limitWMax = (calInfo_.kernelWMax <= KERNEL_W_LIMIT);
  bool isCapable = limitSizeMax && limitWMax;
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dSmallPoolTiling IsCapable check: %s",
          isCapable ? "true" : "false");
  return isCapable;
}

uint64_t AdaptiveMaxPool3dSmallPoolTiling::CalKernelMax(uint64_t inputSize, uint64_t outputSize) {
  uint64_t kernelSize = 1;
  if (outputSize > CAL_KER_THRESHOLD) {
    return (inputSize + outputSize - 1) / outputSize+1; // 防止计算时间过长
  }
  for (uint64_t i = 0; i < outputSize; i++) {
    auto kernelLeft = (i * inputSize) / outputSize;
    auto kernelRight = ((i + 1) * inputSize + outputSize - 1) / outputSize;
    auto kernelCurrent = kernelRight - kernelLeft;
    kernelSize = kernelCurrent > kernelSize ? kernelCurrent : kernelSize;
  }
  return kernelSize;
}

void AdaptiveMaxPool3dSmallPoolTiling::SetTilingData() {
  tilingdata_.set_N(input_.N);
  tilingdata_.set_C(input_.C);
  tilingdata_.set_Di(input_.Di);
  tilingdata_.set_Hi(input_.Hi);
  tilingdata_.set_Wi(input_.Wi);
  tilingdata_.set_Do(input_.Do);
  tilingdata_.set_Ho(input_.Ho);
  tilingdata_.set_Wo(input_.Wo);
  tilingdata_.set_coreNums(input_.coreNum);
  tilingdata_.set_useCoreNum(calInfo_.useCoreNum);
  tilingdata_.set_totalIdx(calInfo_.totalIdx);
  tilingdata_.set_blockFactor(calInfo_.blockFactor);
  tilingdata_.set_blockTail(calInfo_.blockTail);
  tilingdata_.set_ncFactor(calInfo_.ncFactor);
  tilingdata_.set_doFactor(calInfo_.doFactor);
  tilingdata_.set_hoFactor(calInfo_.hoFactor);
  tilingdata_.set_woFactor(calInfo_.woFactor);
  tilingdata_.set_ncOuter(calInfo_.ncOuter);
  tilingdata_.set_doOuter(calInfo_.doOuter);
  tilingdata_.set_hoOuter(calInfo_.hoOuter);
  tilingdata_.set_woOuter(calInfo_.woOuter);
  tilingdata_.set_ncTail(calInfo_.ncTail);
  tilingdata_.set_doTail(calInfo_.doTail);
  tilingdata_.set_hoTail(calInfo_.hoTail);
  tilingdata_.set_woTail(calInfo_.woTail);
}

ge::graphStatus AdaptiveMaxPool3dSmallPoolTiling::DoOpTiling() {
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dSmallPoolTiling DoOpTiling start.");
  auto kernelD = calInfo_.kernelDMax;
  auto kernelH = calInfo_.kernelHMax;
  auto kernelW = calInfo_.kernelWMax;
  OP_TILING_CHECK((kernelW <= 0 || kernelH <= 0 || kernelD <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Kernel size <= 0, not support"),
                  return ge::GRAPH_FAILED);

  calInfo_.ncFactor = BLOCK_LEN / sizeof(float);

  auto kernelWFactorMax1 = KERNEL_W_LIMIT / kernelW > input_.Wo ? input_.Wo : KERNEL_W_LIMIT / kernelW;
  auto kernelWFactorMax2 = KERNEL_SIZE_LIMIT / (kernelD * kernelH) / 8 * 8 / kernelW;
  calInfo_.woFactor = kernelWFactorMax1 > kernelWFactorMax2 ? kernelWFactorMax2 : kernelWFactorMax1;

  auto kernelWprodFactorAlign = (kernelW * calInfo_.woFactor + 8 - 1) / 8 * 8;
  auto kernelHFactorMax1 = input_.Ho > KERNEL_OUT_SIZE_LIMIT / kernelWprodFactorAlign
                               ? KERNEL_OUT_SIZE_LIMIT / kernelWprodFactorAlign
                               : input_.Ho;
  auto kernelHFactorMax2 = KERNEL_SIZE_LIMIT / (kernelD * kernelH * kernelWprodFactorAlign);
  calInfo_.hoFactor = kernelHFactorMax1 > kernelHFactorMax2 ? kernelHFactorMax2 : kernelHFactorMax1;

  auto kernelDFactorMax1 = input_.Do > KERNEL_OUT_SIZE_LIMIT / (kernelWprodFactorAlign * calInfo_.hoFactor)
                               ? KERNEL_OUT_SIZE_LIMIT / (kernelWprodFactorAlign * calInfo_.hoFactor)
                               : input_.Do;
  auto kernelDFactorMax2 = KERNEL_SIZE_LIMIT / (kernelD * calInfo_.hoFactor * kernelH * kernelWprodFactorAlign);
  calInfo_.doFactor = kernelDFactorMax1 > kernelDFactorMax2 ? kernelDFactorMax2 : kernelDFactorMax1;
  OP_LOGD(context_->GetNodeName(), "Tiling result: ub factor %lu, %lu, %lu, %lu", calInfo_.ncFactor, calInfo_.doFactor,
          calInfo_.hoFactor, calInfo_.woFactor);
  OP_TILING_CHECK((calInfo_.doFactor <= 0 || calInfo_.hoFactor <= 0 || calInfo_.woFactor <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Kernel multiply <= 0, not support"),
                  return ge::GRAPH_FAILED);

  calInfo_.doOuter = (input_.Do + calInfo_.doFactor - 1) / calInfo_.doFactor;
  calInfo_.doTail = input_.Do - (calInfo_.doOuter - 1) * calInfo_.doFactor;
  calInfo_.hoOuter = (input_.Ho + calInfo_.hoFactor - 1) / calInfo_.hoFactor;
  calInfo_.hoTail = input_.Ho - (calInfo_.hoOuter - 1) * calInfo_.hoFactor;
  calInfo_.woOuter = (input_.Wo + calInfo_.woFactor - 1) / calInfo_.woFactor;
  calInfo_.woTail = input_.Wo - (calInfo_.woOuter - 1) * calInfo_.woFactor;
  calInfo_.ncOuter = (input_.N * input_.C + calInfo_.ncFactor - 1) / calInfo_.ncFactor;
  calInfo_.ncTail = input_.N * input_.C - (calInfo_.ncOuter - 1) * calInfo_.ncFactor;

  calInfo_.totalIdx = calInfo_.ncOuter * calInfo_.woOuter * calInfo_.hoOuter * calInfo_.doOuter;  // 总共的UB块
  calInfo_.blockFactor = (calInfo_.totalIdx + input_.coreNum - 1) / input_.coreNum;
  calInfo_.useCoreNum = (calInfo_.totalIdx + calInfo_.blockFactor - 1) / calInfo_.blockFactor;
  calInfo_.blockTail = calInfo_.totalIdx - (calInfo_.useCoreNum - 1) * calInfo_.blockFactor;

  OP_LOGD(context_->GetNodeName(), "Tiling result: multi core factor %lu, %lu, %lu, %lu", calInfo_.ncOuter,
          calInfo_.doOuter, calInfo_.hoOuter, calInfo_.woOuter);
  SetTilingData();
  return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveMaxPool3dSmallPoolTiling::GetTilingKey() const {
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dSmallPoolTiling GetTilingKey start.");
  uint64_t tilingKey = TILING_KEY_SMALL_KERNEL_FLOAT;
  if (input_.xDtype == ge::DT_FLOAT) {
    tilingKey = TILING_KEY_SMALL_KERNEL_FLOAT;
  } else if (input_.xDtype == ge::DT_FLOAT16) {
    tilingKey = TILING_KEY_SMALL_KERNEL_HALF;
  } else {
    tilingKey = TILING_KEY_SMALL_KERNEL_BF16;
  }
  OP_LOGD(context_->GetNodeName(), "Tiling key is %lu", tilingKey);
  return tilingKey;
}

ge::graphStatus AdaptiveMaxPool3dSmallPoolTiling::PostTiling() {
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dSmallPoolTiling PostTiling start.");
  context_->SetBlockDim(tilingdata_.get_useCoreNum());
  OP_TILING_CHECK(tilingdata_.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                      tilingdata_.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
                  return ge::GRAPH_FAILED);
  tilingdata_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingdata_.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("AdaptiveMaxPool3d", AdaptiveMaxPool3dSmallPoolTiling, 2);
}  // namespace optiling
