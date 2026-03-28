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
 * \file stft_tiling_generalized.cpp
 * \brief
 */
#include <vector>
#include <map>
#include <cmath>
#include <climits>
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_templates_registry.h"
#include "stft_tiling_base.h"
#include "error_log.h"

using namespace AscendC;
using namespace matmul_tiling;

namespace optiling {

constexpr size_t EXTRA_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t GATHER_MASK_UB_SIZE = 10240; // in bytes
constexpr uint32_t GATHER_MASK_COMPLEX_UB_SIZE = 20480; // in bytes
constexpr uint32_t WORKSPACE_ALIGN_SIZE = 512;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t PACKAGE_SIZE = 128;
constexpr uint32_t DWORD_SIZE = 4; // float32
constexpr uint32_t MAX_LOOP = 100000;
constexpr uint32_t GATHER_MASK = 64;
constexpr uint32_t WORKSPACE_NUM = 3;
constexpr uint32_t SPLIT_WINDOW_VEC_SKIP_THRESHOLD = 8;
constexpr uint32_t SPLIT_WINDOW_MTE_SKIP_BUFFER_NUM = 2;
constexpr uint32_t IMAG_AND_REAL = 2;
constexpr uint32_t TILING_KEY_FLOAT32 = 1;
constexpr uint32_t TILING_KEY_COMPLEX64 = 2;
constexpr uint32_t TILING_KEY_FLOAT16 = 3;
constexpr uint32_t CORE_COEFF = 2;
constexpr uint32_t INVALID_CORES_NUM = 0x5A5A5A5A;
std::map<int, std::vector<int>> g_factorsMap = {
  {50, {50, 25, 10, 5, 2, 1}},
  {48, {48, 24, 12, 8, 6, 4, 2, 1}},
  {40, {40, 20, 10, 8, 5, 4, 2, 1}},
  {25, {25, 5, 1}},
  {24, {24, 12, 6, 4, 2, 1}},
  {20, {20, 10, 5, 4, 2, 1}},
  {16, {16, 8, 4, 2, 1}},
  {12, {12, 6, 2, 1}},
  {10, {10, 5, 2, 1}},
  {8, {8, 4, 2, 1}},
  {6, {6, 1}},
  {5, {5, 1}},
  {4, {4, 2, 1}},
  {2, {2, 1}},
  {1, {1}}
};

static inline int32_t CeilDiv(int a, int b) {
  return (a + b - 1) / b;
}

class STFTGeneralizedTiling : public STFTBaseTiling {
 public:
  explicit STFTGeneralizedTiling(gert::TilingContext* context) : STFTBaseTiling(context) {
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
  uint32_t matmulM{0};
  uint32_t matmulN{0};
  uint32_t bCoreNum{0};
  uint32_t bTailCoreNum{0};
  uint32_t bCoreFactor{0};
  uint32_t mCoreNum{0};
  uint32_t mTailCoreNum{0};
  uint32_t mCoreFactor{0};
  uint32_t nCoreNum{0};
  uint32_t nTailCoreNum{0};
  uint32_t nCoreFactor{0};
  uint32_t nfftAlign{0};
  STFTGeneralizedTilingData tilingData;

  matmul_tiling::DataType GetMatmulDataType(ge::DataType inputDtype) const;
  void SetMatmulTiling(matmul_tiling::MatmulApiTiling &tilingApi, int m, int n, int k, int kAlign, TCubeTiling &mmTiling);
  uint32_t CalcMaskUBSize(uint32_t memHasUsed) const;
  uint32_t CalcCopyUBSize() const;
  uint32_t CalcComplexUBLoop() const;
  uint32_t CalcComplexCopyUBSize() const;
  void SetN();
  uint32_t SplitCoresOnB(uint32_t coresNum);
  uint32_t SplitCoresOnM(uint32_t coresNum);
  uint32_t SplitCoresOnN(uint32_t coresNum);
  bool SplitCoresOnBMN();
  bool SplitCoresFirstOnB();
  bool SplitCoresFirstOnM();
  bool SplitCoresFirstOnN();
  uint32_t SplitCoresMNBlanced();
  bool SplitCores();
  void SplitWindowTiling();
  void GetPlanSplitStrategy();
  int64_t CalcMatmulCost(int64_t mCoreNum, int64_t nCoreNum) const;
};

void STFTGeneralizedTiling::GetPlanSplitStrategy() {
  int32_t blockDim = aivCoreNum;
  int32_t oneRowSize = nfftAlign * 4;
  int32_t halfUbSize = (ubSize - oneRowSize) / 2;
  int32_t mFactor = CeilDiv(2 * matmulM, blockDim);
  int32_t prevCnt = blockDim * (mFactor - 1);
  int32_t remainCnt = 2 * matmulM - prevCnt;
  int32_t totalLine = mFactor;
  int32_t tailBlockIdx = remainCnt;
  int32_t tailLine = mFactor;
  if (remainCnt < blockDim) {
    tailLine = mFactor - 1;
  }
  int32_t ubMaxLine = halfUbSize / oneRowSize;
  int32_t numsInOneRepeat = 64;
  int32_t totalInCol = nfftAlign / numsInOneRepeat;
  int32_t tailInCol = nfftAlign % numsInOneRepeat;

  tilingData.planTilingData.set_totalInCol(totalInCol);
  tilingData.planTilingData.set_tailInCol(tailInCol);
  tilingData.planTilingData.set_oneRowLen(nfftAlign);
  tilingData.planTilingData.set_totalLine(totalLine);
  tilingData.planTilingData.set_tailLine(tailLine);
  tilingData.planTilingData.set_tailBlockIdx(tailBlockIdx);
  tilingData.planTilingData.set_ubMaxLine(ubMaxLine);
  tilingData.planTilingData.set_batch(batch);
  tilingData.planTilingData.set_matmulM(matmulM);
  tilingData.planTilingData.set_matmulN(matmulN);
}

uint32_t STFTGeneralizedTiling::SplitCoresOnB(uint32_t coresNum) {
  uint32_t coresLeft = coresNum;
  auto iter = g_factorsMap.find(coresNum);
  OP_TILING_CHECK(iter == g_factorsMap.end(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "invalid coresNum %d", coresNum),
                  return INVALID_CORES_NUM);

  std::vector<int> factors = iter->second;
  for (size_t i = 0; i < factors.size(); i++) {
    if (batch / factors[i] >= 1) {
      bCoreNum = factors[i];
      bCoreFactor = (batch + factors[i] - 1) / factors[i];
      bTailCoreNum = bCoreNum * bCoreFactor - batch;
      coresLeft = coresNum / factors[i];
      break;
    }
  }
  return coresLeft;
}

uint32_t STFTGeneralizedTiling::SplitCoresOnM(uint32_t coresNum) {
  uint32_t coresLeft = coresNum;
  auto iter = g_factorsMap.find(coresNum);
  OP_TILING_CHECK(iter == g_factorsMap.end(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "invalid coresNum %d", coresNum),
                  return INVALID_CORES_NUM);

  std::vector<int> factors = iter->second;
  for (size_t i = 0; i < factors.size(); i++) {
    if (matmulM / factors[i] >= 1) {
      mCoreNum = factors[i];
      mCoreFactor = (matmulM + factors[i] - 1) / factors[i];
      mTailCoreNum = mCoreNum * mCoreFactor - matmulM;
      coresLeft = coresNum / factors[i];
      break;
    }
  }
  return coresLeft;
}

uint32_t STFTGeneralizedTiling::SplitCoresOnN(uint32_t coresNum) {
  uint32_t coresLeft = coresNum;
  auto iter = g_factorsMap.find(coresNum);
  OP_TILING_CHECK(iter == g_factorsMap.end(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "invalid coresNum %d", coresNum),
                  return INVALID_CORES_NUM);

  std::vector<int> factors = iter->second;
  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  nCoreNum = 0;
  for (size_t i = 0; i < factors.size(); i++) {
    int n = 1;
    while (matmulN > (factors[i] - 1) * n * BLOCK_SIZE / typeSize) {
      if (matmulN <= factors[i] * n * BLOCK_SIZE / typeSize) {
        nCoreNum = factors[i];
        nCoreFactor = n * BLOCK_SIZE / typeSize;
        if (matmulN % nCoreFactor != 0) {
          nTailCoreNum = 1;
        } else {
          nTailCoreNum = 0;
        }
        coresLeft = coresNum / factors[i];
        break;
      }
      n++;
    }
    if (nCoreNum > 0) {
      break;
    }
  }
  if (nCoreNum == 1 && nTailCoreNum == 1) {
      nCoreFactor = matmulN;
      nTailCoreNum = 0;
  }
  return coresLeft;
}

void STFTGeneralizedTiling::SetN() {
    nCoreNum = 1;
    nCoreFactor = matmulN;
    nTailCoreNum = 0;
}

bool STFTGeneralizedTiling::SplitCoresOnBMN() {
  uint32_t coresLeft = aivCoreNum;
  coresLeft = SplitCoresOnB(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split B failed"),
                  return false);

  if (coresLeft == 1) {
    mCoreNum = 1;
    mTailCoreNum = 0;
    mCoreFactor = matmulM;
    nCoreNum = 1;
    nCoreFactor = matmulN;
    nTailCoreNum = 0;
    return true;
  }
  coresLeft = SplitCoresOnM(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split M failed"),
                  return false);

  OP_TILING_CHECK(coresLeft == 1, SetN(), return true);

  coresLeft = SplitCoresOnN(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split N failed"),
                  return false);
  return true;
}

bool STFTGeneralizedTiling::SplitCoresFirstOnB() {
  bCoreFactor = batch / bCoreNum;
  bTailCoreNum = 0;
  uint32_t coresLeft = aivCoreNum / bCoreNum;
  if (coresLeft == 1) {
    mCoreNum = 1;
    mTailCoreNum = 0;
    mCoreFactor = matmulM;
    nCoreNum = 1;
    nCoreFactor = matmulN;
    nTailCoreNum = 0;
    return true;
  }
  coresLeft = SplitCoresOnM(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split M failed"),
                  return false);

  OP_TILING_CHECK(coresLeft == 1, SetN(), return true);

  coresLeft = SplitCoresOnN(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split N failed"),
                  return false);
  return true;
}

bool STFTGeneralizedTiling::SplitCoresFirstOnM() {
  mCoreFactor = matmulM / mCoreNum;
  mTailCoreNum = 0;
  uint32_t coresLeft = aivCoreNum / mCoreNum;
  if (coresLeft == 1) {
    bCoreNum = 1;
    bTailCoreNum = 0;
    bCoreFactor = batch;
    nCoreNum = 1;
    nCoreFactor = matmulN;
    nTailCoreNum = 0;
    return true;
  }
  coresLeft = SplitCoresOnB(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split B failed"),
                  return false);

  OP_TILING_CHECK(coresLeft == 1, SetN(), return true);

  coresLeft = SplitCoresOnN(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split N failed"),
                  return false);
  return true;
}

bool STFTGeneralizedTiling::SplitCoresFirstOnN() {
  nCoreFactor = matmulN / nCoreNum;
  nTailCoreNum = 0;
  uint32_t coresLeft = aivCoreNum / nCoreNum;
  if (coresLeft == 1) {
    bCoreNum = 1;
    bTailCoreNum = 0;
    bCoreFactor = batch;
    mCoreNum = 1;
    mCoreFactor = matmulM;
    mTailCoreNum = 0;
    return true;
  }
  coresLeft = SplitCoresOnB(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split B failed"),
                  return false);
  if (coresLeft == 1) {
    mCoreNum = 1;
    mCoreFactor = matmulM;
    mTailCoreNum = 0;
    return true;
  }
  coresLeft = SplitCoresOnM(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split M failed"),
                  return false);
  return true;
}

int64_t STFTGeneralizedTiling::CalcMatmulCost(int64_t mCoreNum, int64_t nCoreNum) const {
  if (mCoreNum == 0 || nCoreNum == 0) {
    return LLONG_MAX;
  }
  int64_t m = matmulM * IMAG_AND_REAL;
  int64_t n = batch * matmulN;
  if (matmulM < mCoreNum || n < nCoreNum) {
    return LLONG_MAX;
  }
  return ((m + mCoreNum - 1) / mCoreNum) * n + ((n + nCoreNum - 1) / nCoreNum) * m;
}

uint32_t STFTGeneralizedTiling::SplitCoresMNBlanced() {
  // return 1: success, 0: no result, no error, INVALID_CORES_NUM: no result, with error occured
  int64_t matmulCostMin = LLONG_MAX;
  uint32_t mCoreNumNeedSplit;
  uint32_t nCoreNumNeedSplit;

  auto iter = g_factorsMap.find(aivCoreNum);
  OP_TILING_CHECK(iter == g_factorsMap.end(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "invalid aivCoreNum %ld", aivCoreNum),
                  return INVALID_CORES_NUM);

  std::vector<int> factors = iter->second;
  for (size_t i = 0; i < factors.size(); i++) {
    int64_t curMatmulCost = CalcMatmulCost(factors[i], aivCoreNum / factors[i]);
    if (curMatmulCost < matmulCostMin) {
      matmulCostMin = curMatmulCost;
      mCoreNumNeedSplit = factors[i];
      nCoreNumNeedSplit = aivCoreNum / factors[i];
    }
  }

  if (matmulCostMin == LLONG_MAX) {
    return 0;
  }
  uint32_t coresLeft = SplitCoresOnM(mCoreNumNeedSplit);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split M failed"),
                  return INVALID_CORES_NUM);

  coresLeft = SplitCoresOnB(nCoreNumNeedSplit);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split B failed"),
                  return INVALID_CORES_NUM);

  OP_TILING_CHECK(coresLeft == 1, SetN(), return 1);

  coresLeft = SplitCoresOnN(coresLeft);
  OP_TILING_CHECK(coresLeft == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "split N failed"),
                  return INVALID_CORES_NUM);
  return 1;
}

bool STFTGeneralizedTiling::SplitCores() {
  auto ret = SplitCoresMNBlanced();
  OP_TILING_CHECK(ret == INVALID_CORES_NUM,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "SplitCoresMNBlanced failed"),
                  return false);

  if (ret == 1) {
    return true;
  }

  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  bCoreNum = 1;
  mCoreNum = 1;
  nCoreNum = 1;
  auto iter = g_factorsMap.find(aivCoreNum);
  OP_TILING_CHECK(iter == g_factorsMap.end(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "invalid aivCoreNum %ld", aivCoreNum),
                  return false);

  std::vector<int> factors = iter->second;
  for (size_t i = 0; i < factors.size(); i++) {
    if (batch % factors[i] == 0 && bCoreNum == 1) {
      bCoreNum = factors[i];
    }
    if (matmulM % factors[i] == 0 && mCoreNum == 1) {
      mCoreNum = factors[i];
    }
    if (matmulN % (factors[i] * BLOCK_SIZE / typeSize) == 0 && nCoreNum == 1) {
      nCoreNum = factors[i];
    }
  }

  // batch has max factors of aivCoreNum, split batch first
  if (bCoreNum >= mCoreNum && bCoreNum >= nCoreNum && bCoreNum > 1) {
    return SplitCoresFirstOnB();
  }

  if (mCoreNum > bCoreNum && mCoreNum >= nCoreNum) {
    return SplitCoresFirstOnM();
  }

  if (nCoreNum > bCoreNum && nCoreNum > mCoreNum) {
    return SplitCoresFirstOnN();
  }
  // todo: check performance, may need to change split order
  return SplitCoresOnBMN();
}

bool STFTGeneralizedTiling::IsCapable() {
  return true;
}

uint32_t STFTGeneralizedTiling::CalcMaskUBSize(uint32_t memHasUsed) const {
  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  uint32_t ubLeft = ubSize - memHasUsed;
  uint32_t count = ubLeft / (4 * typeSize + DWORD_SIZE * 2);
  uint32_t maskUBSize = count * DWORD_SIZE / 2 / BLOCK_SIZE * BLOCK_SIZE * 2;
  return maskUBSize;
}

uint32_t STFTGeneralizedTiling::CalcCopyUBSize() const {
  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  uint32_t ubLeft = ubSize - GATHER_MASK_UB_SIZE;
  uint32_t count = GATHER_MASK_UB_SIZE / DWORD_SIZE;
  uint32_t copyUBSize = (ubLeft - (4 * typeSize + 2 * DWORD_SIZE) * count) / BLOCK_SIZE * BLOCK_SIZE;
  return copyUBSize;
}

uint32_t STFTGeneralizedTiling::CalcComplexCopyUBSize() const {
  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  uint32_t ubLeft = ubSize - GATHER_MASK_COMPLEX_UB_SIZE;
  uint32_t count = GATHER_MASK_COMPLEX_UB_SIZE / DWORD_SIZE;
  uint32_t copyUBSize = (ubLeft - (4 * typeSize + 2 * DWORD_SIZE) * count) / BLOCK_SIZE * BLOCK_SIZE;
  return copyUBSize;
}

uint32_t STFTGeneralizedTiling::CalcComplexUBLoop() const {
  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  uint32_t copyUBSize = CalcComplexCopyUBSize() / 2;
  uint32_t ubLoop = 1;
  uint32_t ubFormer;
  uint32_t ubAlignSize;
  while(ubLoop < MAX_LOOP) {
    ubFormer = (nfftAlign + ubLoop - 1) / ubLoop;
    ubAlignSize = ((ubFormer << 1) + GATHER_MASK - 1) / GATHER_MASK * GATHER_MASK * WORKSPACE_NUM * typeSize;
    if (ubAlignSize < copyUBSize) {
      return ubLoop;
    }
    ubLoop++;
  }
  return ubLoop;
}

void STFTGeneralizedTiling::SplitWindowTiling() {
  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  tilingData.set_splitWindowSkipNum(0);
  if ((nfft != nfftAlign || (hop * typeSize) % BLOCK_SIZE != 0) && (nfftAlign + hop - 1) / hop < inputSize / hop + 1) {
    // not align, copy nfftAlign to UB once
    if (nfftAlign * typeSize + GATHER_MASK_UB_SIZE > ubSize) {
      tilingData.set_maskUBSize(GATHER_MASK_UB_SIZE);
      uint32_t copyUBSize = CalcCopyUBSize();
      tilingData.set_copyUBSize(copyUBSize);
    } else {
      uint32_t splitWindowSkipNum = (nfftAlign + hop - 1) / hop;
      uint32_t maxMem;
      if (splitWindowSkipNum <= SPLIT_WINDOW_VEC_SKIP_THRESHOLD) {
        tilingData.set_splitWindowSkipNum(splitWindowSkipNum);
        uint32_t maxN = (nCoreFactor + splitWindowSkipNum - 1) / splitWindowSkipNum;
        maxMem = ((maxN * nfftAlign * typeSize) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        // because need copy_ub_to_ub
        maxMem *= SPLIT_WINDOW_MTE_SKIP_BUFFER_NUM;
      } else {
        tilingData.set_splitWindowSkipNum(SPLIT_WINDOW_VEC_SKIP_THRESHOLD);
        uint32_t maxN = (nCoreFactor + SPLIT_WINDOW_VEC_SKIP_THRESHOLD - 1) / SPLIT_WINDOW_VEC_SKIP_THRESHOLD;
        maxMem = maxN * (SPLIT_WINDOW_VEC_SKIP_THRESHOLD * hop + nfftAlign) + nfftAlign;
        maxMem = (maxMem * typeSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
      }
      if (maxMem + GATHER_MASK_UB_SIZE > ubSize) {
        tilingData.set_maskUBSize(GATHER_MASK_UB_SIZE);
        uint32_t copyUBSize = CalcCopyUBSize();
        tilingData.set_copyUBSize(copyUBSize);
      } else {
        tilingData.set_copyUBSize(maxMem);
        uint32_t maskUBSize = CalcMaskUBSize(maxMem);
        tilingData.set_maskUBSize(maskUBSize);
      }
    }
  } else {// align, copy data to UB as much as possible once
    tilingData.set_maskUBSize(GATHER_MASK_UB_SIZE);
    uint32_t copyUBSize = CalcCopyUBSize();
    tilingData.set_copyUBSize(copyUBSize);
  }
}

ge::graphStatus STFTGeneralizedTiling::DoOpTiling() {
  matmulM = onesided ? (nfft / 2 + 1) : nfft;
  matmulN = inputSize / hop + 1;

  bool result = SplitCores();
  OP_TILING_CHECK(result == false,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "SplitCores failed"),
                  return ge::GRAPH_FAILED);

  tilingData.set_matmulM(matmulM);
  tilingData.set_matmulN(matmulN);
  tilingData.set_batchCoreNum(bCoreNum);
  tilingData.set_batchTailCoreNum(bTailCoreNum);
  tilingData.set_batchCoreFactor(bCoreFactor);
  tilingData.set_matmulMCoreNum(mCoreNum);
  tilingData.set_matmulMTailCoreNum(mTailCoreNum);
  tilingData.set_matmulMCoreFactor(mCoreFactor);
  tilingData.set_matmulNCoreNum(nCoreNum);
  tilingData.set_matmulNTailCoreNum(nTailCoreNum);
  tilingData.set_matmulNCoreFactor(nCoreFactor);

  tilingData.set_batch(batch);
  tilingData.set_inputSize(inputSize);
  tilingData.set_nfft(nfft);
  uint32_t typeSize = ge::GetSizeByDataType(dtype);
  nfftAlign = (nfft * typeSize + PACKAGE_SIZE - 1) / PACKAGE_SIZE * PACKAGE_SIZE / typeSize;
  tilingData.set_nfftAlign(nfftAlign);
  tilingData.set_hopLength(hop);
  tilingData.set_normalized(normalized);
  if (normalized) {
    float root = 1 / sqrt(nfft);
    tilingData.set_rootNfft(root);
  }

 if (dtype == ge::DataType::DT_COMPLEX64) {
    uint32_t ubLoop = CalcComplexUBLoop();
    tilingData.set_nFactorUbLoop(ubLoop);
    uint32_t ubFormer = (nfftAlign + ubLoop - 1) / ubLoop;
    tilingData.set_nFactorUbFormer(ubFormer);
    tilingData.set_nFactorUbTail(nfftAlign - (ubLoop - 1) * ubFormer);
    tilingData.set_maskUBSize(GATHER_MASK_UB_SIZE);
    uint32_t copyUBSize = CalcCopyUBSize();
    tilingData.set_copyUBSize(copyUBSize);
    return ge::GRAPH_SUCCESS;
  }

  SplitWindowTiling();
  tilingData.set_usedCoreNum((bCoreNum * mCoreNum * nCoreNum + 1) / CORE_COEFF * CORE_COEFF);

  return ge::GRAPH_SUCCESS;
}

matmul_tiling::DataType STFTGeneralizedTiling::GetMatmulDataType(ge::DataType inputDtype) const {
  matmul_tiling::DataType mtype;
  if (inputDtype == ge::DataType::DT_COMPLEX64 || inputDtype == ge::DataType::DT_FLOAT) {
    mtype = matmul_tiling::DataType::DT_FLOAT;
  } else {
    mtype = matmul_tiling::DataType::DT_FLOAT16;
  }
  return mtype;
}

void STFTGeneralizedTiling::SetMatmulTiling(matmul_tiling::MatmulApiTiling &tilingApi, int m, int n, int k,
                                            int kAlign, TCubeTiling &mmTiling) {
  m *= IMAG_AND_REAL;
  TPosition leftPos = TPosition::GM;
  CubeFormat leftFormat = CubeFormat::ND;
  matmul_tiling::DataType leftDtype = GetMatmulDataType(dtype);
  int transposeA = 0;

  TPosition rightPos = TPosition::GM;
  CubeFormat rightFormat = CubeFormat::ND;
  matmul_tiling::DataType rightDtype = leftDtype;
  int transposeB = 1;

  TPosition resPos = TPosition::GM;
  CubeFormat resFormat = CubeFormat::ND;
  matmul_tiling::DataType resDtype = leftDtype;

  TPosition biasPos = TPosition::GM;
  CubeFormat biasFormat = CubeFormat::ND;
  matmul_tiling::DataType biasDtype = leftDtype;
  int isBias = 0;

  tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
  tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
  tilingApi.SetCType(resPos, resFormat, resDtype);
  tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);
  tilingApi.SetOrgShape(m, n, kAlign);

  // singleCoreM, singleCoreN according to split core
  tilingApi.SetShape(m, n, k);
  tilingApi.SetBias(bool(isBias));
  tilingApi.SetBufferSpace(-1, -1, -1);
  mmTiling.set_usedCoreNum(1);
  tilingApi.GetTiling(mmTiling);
  mmTiling.set_iterateOrder(1);
}

ge::graphStatus STFTGeneralizedTiling::DoLibApiTiling() {
  // 调用MatMul高阶API tiling，后续重点优化此逻辑
  int m = mCoreFactor;
  int n = nCoreFactor;
  int kAlign = nfftAlign;
  int k = nfft;

  // set matmul tiling for m_head * n_head
  matmul_tiling::MatmulApiTiling mm0;
  SetMatmulTiling(mm0, m, n, k, kAlign, tilingData.mm0TilingData);

  // set matmul tiling for m_head * n_tail
  matmul_tiling::MatmulApiTiling mm1;
  if (nTailCoreNum > 0) {
    n = matmulN % nCoreFactor;
  }
  SetMatmulTiling(mm1, m, n, k, kAlign, tilingData.mm1TilingData);

  // set matmul tiling for m_tail * n_head
  matmul_tiling::MatmulApiTiling mm2;
  if (mTailCoreNum > 0) {
    m = mCoreFactor - 1;
  }
  n = nCoreFactor;
  SetMatmulTiling(mm2, m, n, k, kAlign, tilingData.mm2TilingData);

  // set matmul tiling for m_tail * n_tail
  matmul_tiling::MatmulApiTiling mm3;
  if (nTailCoreNum > 0) {
    n = matmulN % nCoreFactor;
  }
  SetMatmulTiling(mm3, m, n, k, kAlign, tilingData.mm3TilingData);

  GetPlanSplitStrategy();
  return ge::GRAPH_SUCCESS;
}

uint64_t STFTGeneralizedTiling::GetTilingKey() const {
  if (dtype == ge::DataType::DT_FLOAT) {
    return TILING_KEY_FLOAT32;
  } else if (dtype == ge::DataType::DT_COMPLEX64) {
    return TILING_KEY_COMPLEX64;
  } else { // float16
    return TILING_KEY_FLOAT16;
  }
}

ge::graphStatus STFTGeneralizedTiling::GetWorkspaceSize() {
  // 每块workspace地址需要512B对齐
  // 第0块workspace用于存储按照窗口拆分之后的input data
  // 按照nfft block对齐之后的大小
  uint64_t typeSize = ge::GetSizeByDataType(dtype);

  size_t splitWindowWorkspaceSize =
    ((typeSize * batch * matmulN * nfftAlign + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE) * WORKSPACE_ALIGN_SIZE;

  // 第一块workspace用于存储input data和plan mm运算之后的结果
  size_t matmulWorkspaceSize =
  ((typeSize * batch * matmulN * matmulM * 2 + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE) * WORKSPACE_ALIGN_SIZE;

  size_t planWorkspaceSize = ((typeSize * matmulM * nfftAlign * 2 + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE) *
                             WORKSPACE_ALIGN_SIZE;

  workspaceSize_ = splitWindowWorkspaceSize + matmulWorkspaceSize + planWorkspaceSize + sysWorkspaceSize +
                   EXTRA_WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus STFTGeneralizedTiling::PostTiling() {
  tilingData.set_tilingKey(GetTilingKey());
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(aicCoreNum);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("STFT", STFTGeneralizedTiling, 20000);

}  // namespace optiling
