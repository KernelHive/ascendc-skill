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
 * \file stft_tiling_align.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_templates_registry.h"
#include "stft_tiling_base.h"
#include "error_log.h"

using namespace AscendC;
using namespace matmul_tiling;

namespace optiling {

constexpr size_t EXTRA_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BLK_FRAME_SETTING = 32;
constexpr int64_t NFFT_DIVISOR = 2;
constexpr int32_t GATHER_MEMORY_PART = 10;
constexpr int32_t ONE_REPEAT_UB_SIZE = 256;
constexpr int32_t WORKSPACE_ALIGN_SIZE = 512;
constexpr int BLOCK_ALIGN_NUM = 8;
constexpr int REAL_IMAG = 2;
constexpr int C_V_DOUBLE = 2;
constexpr int TWO_NUM_AVERAGE = 2;
constexpr int MIN_FACTOR = 1;

class STFTTiling : public STFTBaseTiling {
 public:
  explicit STFTTiling(gert::TilingContext* context) : STFTBaseTiling(context) {
  }

 protected:
  bool IsCapable() override;

  // 1、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;

  // 2、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override;

  // 3、计算TilingKey
  uint64_t GetTilingKey() const override;

  // 4、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;

  // 5、保存Tiling数据
  ge::graphStatus PostTiling() override;

 private:
  const int32_t CalculateMaxGatherSize();
  void GetVectorSplitFactor(int32_t windowCount);
  void GetCubeSplitFactor(int32_t batch, int32_t windowCount);
  int32_t frameCount{0};
  int32_t frameCountAlign{0};
  int32_t blkFrame{0};
  int32_t matmulM{0};
  int32_t batchFactor{0};
  int32_t batchLoop{0};
  int32_t batchReminder{0};
  STFTTilingData tilingData;
};

static std::vector<int> GetIntegerFactor(int32_t value) {
  std::vector<int> factors;
  for (int32_t i = value; i > 1; i--) {
    if (value % i == 0) {
      factors.push_back(i);
    }
  }
  return factors;
}

static int GetAnyBatchSplitFactor(int target, std::vector<int> factors) {
  for (size_t i = 0; i < factors.size(); i++) {
    int32_t batchFactor = (target + factors[i] - 1) / factors[i];
    int32_t total = batchFactor * factors[i];
    int32_t diff = total - target;
    if (diff >= batchFactor) {
      continue;
    }
    return factors[i];
  }
  return MIN_FACTOR;
}

static int32_t GetBaseFactor(int32_t value, int32_t num) {
  if (num == MIN_FACTOR) {
    return MIN_FACTOR;
  }
  auto factors = GetIntegerFactor(value);
  for (auto factor : factors) {
    if (num % factor == 0) {
      return factor;
    }
  }
  auto factor = GetAnyBatchSplitFactor(num, factors);
  return factor;
}

static int32_t CeilDiv(int a, int b) {
  return (a + b - 1) / b;
}

void STFTTiling::GetVectorSplitFactor(int32_t windowCount) {
  int32_t totalLen = tilingData.get_aicTotalLen();
  int32_t aivTotalEvenRow = ((totalLen / REAL_IMAG + REAL_IMAG - 1) / REAL_IMAG) * REAL_IMAG;
  int32_t aivTotalOddRow = totalLen - aivTotalEvenRow;

  int32_t tailLen = tilingData.get_aicTailLen();
  int32_t aivTailEvenRow = ((tailLen / REAL_IMAG + REAL_IMAG - 1) / REAL_IMAG) * REAL_IMAG;
  int32_t aivTailOddRow = tailLen - aivTailEvenRow;

  tilingData.set_aivBatchLoop(tilingData.get_aicBatchLoop());
  tilingData.set_aivTailLoop(tilingData.get_aicTailLoop());
  tilingData.set_aivTotalEvenRow(aivTotalEvenRow / REAL_IMAG);
  tilingData.set_aivTotalOddRow(aivTotalOddRow / REAL_IMAG);
  tilingData.set_aivTailEvenRow(aivTailEvenRow / REAL_IMAG);
  tilingData.set_aivTailOddRow(aivTailOddRow / REAL_IMAG);
  tilingData.set_aivWindowLoop(tilingData.get_aicMatmulMCore() * C_V_DOUBLE);
  tilingData.set_aivBatchTailIdx(tilingData.get_aicBatchTailIdx() * C_V_DOUBLE);
  tilingData.set_aivMTailIdx(tilingData.get_aicMTailIdx() * C_V_DOUBLE);
}

void STFTTiling::GetCubeSplitFactor(int32_t batch, int32_t windowCount) {
  batchFactor = GetBaseFactor(aicCoreNum, batch);
  batchLoop = CeilDiv(batch, batchFactor);
  batchReminder = batch - (batchFactor - 1) * batchLoop;

  int32_t matmulMCore = aicCoreNum / batchFactor;
  int32_t matmulMFactor = CeilDiv(windowCount, matmulMCore);
  int32_t prevCnt = (matmulMFactor - 1) * matmulMCore;
  int32_t remain = windowCount - prevCnt;
  int32_t matmulMReminder = matmulMFactor;
  int32_t matmulLastTotalIdx = matmulMCore;

  if (remain != 0) {
    matmulMReminder = matmulMFactor - 1;
    matmulLastTotalIdx = remain;
  }

  int32_t aicBatchTailIdx = aicCoreNum;
  if (batchReminder < batchLoop) {
    aicBatchTailIdx = matmulMCore * (batchFactor - 1);
  }

  tilingData.set_aicBatchLoop(batchLoop);
  tilingData.set_aicTailLoop(batchReminder);
  tilingData.set_aicBatchTailIdx(aicBatchTailIdx);
  tilingData.set_aicBatchFactor(batchFactor);
  tilingData.set_aicMatmulMCore(matmulMCore);
  tilingData.set_aicTotalLen(matmulMFactor * REAL_IMAG);
  tilingData.set_aicTailLen(matmulMReminder * REAL_IMAG);
  tilingData.set_aicMTailIdx(matmulLastTotalIdx);
}

const int32_t STFTTiling::CalculateMaxGatherSize() {
  int32_t ubCanUse = ubSize - (BLK_FRAME_SETTING * hop + nfft - hop) * 4 - (BLK_FRAME_SETTING * nfft) * 4;
  int32_t maxSize = ubCanUse / GATHER_MEMORY_PART;
  int32_t base = frameCountAlign * sizeof(float);
  maxSize = maxSize / base * base;
  return maxSize;
}

bool STFTTiling::IsCapable() {
  if (nfft == 400 && hop == 160 && normalized == false && onesided == true && returnComplex == false) {
    return true;
  }
  return false;
}

ge::graphStatus STFTTiling::DoOpTiling() {
  tilingData.set_batch(batch);
  tilingData.set_inputSize(inputSize);

  frameCount = inputSize / hop + 1;
  tilingData.set_frameCount(frameCount);

  frameCountAlign = ((frameCount + BLOCK_ALIGN_NUM - 1) / BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
  tilingData.set_frameCountAlign(frameCountAlign);

  int32_t dummyData = REAL_IMAG * (frameCountAlign - frameCount);
  int32_t ubGap = dummyData >= BLOCK_ALIGN_NUM ? (dummyData / BLOCK_ALIGN_NUM) : 0;
  tilingData.set_aivGatherUbGap(ubGap);

  tilingData.set_nfft(nfft);
  tilingData.set_hop(hop);

  tilingData.set_blkFrame(BLK_FRAME_SETTING);

  matmulM = nfft / NFFT_DIVISOR + 1;
  tilingData.set_matmulM(matmulM);

  GetCubeSplitFactor(batch, matmulM);
  GetVectorSplitFactor(frameCount);
  auto maxSize = CalculateMaxGatherSize();
  tilingData.set_sizePerRepeat(maxSize);
  tilingData.set_blockNum(coreNum);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus STFTTiling::DoLibApiTiling() {
  // 调用MatMul高阶API tiling，后续重点优化此逻辑
  TPosition leftPos = TPosition::GM;
  CubeFormat leftFormat = CubeFormat::ND;
  matmul_tiling::DataType leftDtype = matmul_tiling::DataType::DT_FLOAT;
  int transposeA = 0;

  TPosition rightPos = TPosition::GM;
  CubeFormat rightFormat = CubeFormat::ND;
  matmul_tiling::DataType rightDtype = matmul_tiling::DataType::DT_FLOAT;
  int transposeB = 1;

  TPosition resPos = TPosition::GM;
  CubeFormat resFormat = CubeFormat::ND;
  matmul_tiling::DataType resDtype = matmul_tiling::DataType::DT_FLOAT;

  TPosition biasPos = TPosition::GM;
  CubeFormat biasFormat = CubeFormat::ND;
  matmul_tiling::DataType biasDtype = matmul_tiling::DataType::DT_FLOAT;
  int isBias = 0;

  int M = (nfft / 2 + 1) * 2;
  int N = (inputSize / hop + 1);
  int K = nfft;

  tilingData.mmTilingData.set_usedCoreNum(1);
  MatmulApiTiling tilingApi;
  tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
  tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
  tilingApi.SetCType(resPos, resFormat, resDtype);
  tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);

  tilingApi.SetShape(tilingData.get_aicTotalLen(), N, K);
  tilingApi.SetOrgShape(M, N, K);
  tilingApi.SetBias(bool(isBias));

  tilingApi.SetBufferSpace(-1, -1, -1);
  int64_t status = tilingApi.GetTiling(tilingData.mmTilingData);
  if (status == -1) {
    return ge::GRAPH_FAILED;
  }

  tilingData.mmTilingData.set_iterateOrder(1);
  tilingApi.SetFixSplit(-1, -1, -1);

  return ge::GRAPH_SUCCESS;
}

uint64_t STFTTiling::GetTilingKey() const {
  return 0;
}

ge::graphStatus STFTTiling::GetWorkspaceSize() {
  // 每块workspace地址需要512B对齐
  // 第0块workspace用于存储按照窗口拆分之后的input data
  size_t windowSplitWorkspaceSize =
      batchLoop * ((aicCoreNum * frameCount * nfft * sizeof(float) + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE) *
      WORKSPACE_ALIGN_SIZE;

  // 第一块workspace用于存储input data和plan mm运算之后的结果
  size_t matmulWorkspaceSize =
      ((batch * frameCount * matmulM * 2 * sizeof(float) + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE) *
      WORKSPACE_ALIGN_SIZE;
  workspaceSize_ = windowSplitWorkspaceSize + matmulWorkspaceSize + EXTRA_WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus STFTTiling::PostTiling() {
  tilingData.set_tilingKey(GetTilingKey());
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(coreNum);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("STFT", STFTTiling, 10000);

}  // namespace optiling
