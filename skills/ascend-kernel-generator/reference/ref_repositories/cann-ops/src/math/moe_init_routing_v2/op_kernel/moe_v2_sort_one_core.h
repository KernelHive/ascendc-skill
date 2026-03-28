/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_v2_sort_one_core.h
 * \brief
 */
   
#ifndef MOE_V2_SORT_ONE_CORE_H
#define MOE_V2_SORT_ONE_CORE_H

#include "moe_v2_mrgsort.h"
#include "moe_v2_sort_base.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

class MoeV2SortOneCore : public MoeV2SortBase {
 public:
  __aicore__ inline MoeV2SortOneCore(){};
  template <typename TilingData>
  __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                              GM_ADDR workspace, const TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOut();

 private:
  int64_t sortNum;
  int64_t blockIdx;
};

__aicore__ inline void MoeV2SortOneCore::CopyIn() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->totalLength * sizeof(int32_t)),
                                   0, 0, 0};
  DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);

  LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
  ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
  sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeV2SortOneCore::SortCompute() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
  LocalTensor<float> expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
  Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
  pipe_barrier(PIPE_V);
  Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, this->tileLength);
  pipe_barrier(PIPE_V);

  int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
  if (duplicateNum > 0) {
    int duplicateIndex = this->totalLength - duplicateNum;
    uint64_t mask0 = UINT64_MAX;
    mask0 = mask0 << duplicateNum;
    mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
    uint64_t mask[2] = {mask0, 0};
    Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    pipe_barrier(PIPE_V);
  }

  LocalTensor<float> concatLocal;
  LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
  Concat(concatLocal, expertForSourceRowLocalFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);

  LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
  LocalTensor<uint32_t> sourceRowLocal;
  sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
  Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);

  LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
  LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
  LocalTensor<uint32_t> expandDstToSrcRowLocal;
  expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
  Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
  pipe_barrier(PIPE_V);
  Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);
  pipe_barrier(PIPE_V);

  LocalTensor<int32_t> expertForSourceRowLocalInt32;
  expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
  Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
  sortDataCopyOutQueue.EnQue<float>(outLocal);
  sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeV2SortOneCore::CopyOut() {
  LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = this->totalLength * sizeof(int32_t);
  DataCopyPad(sortedexpertIdxGm, outLocal[0], intriParams);
  DataCopyPad(expandDstToSrcRowGm, outLocal[this->sortNum], intriParams);
  sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename TilingData>
__aicore__ inline void MoeV2SortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                              GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                              const TilingData* tilingData, TPipe* tPipe) {
  this->blockIdx = GetBlockIdx();
  this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
  this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
  this->totalLength = tilingData->n * tilingData->k;
  this->coreNum = tilingData->coreNum;
  this->pipe = tPipe;
  this->n = tilingData->n;
  this->k = tilingData->k;
  this->expertNum = tilingData->expertNum;
  this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
  this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

  expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx, this->tileLength);
  sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), this->tileLength);
  expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace) + this->tileLength,
                                      this->tileLength);

  if (this->blockIdx == this->coreNum - 1) {
    if (this->expertTokensCountOrCumsumFlag > 0) {
      expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum,
                                                  Align(this->expertNum, sizeof(int32_t)));
      InitGlobalMemory(expertTokensCountOrCumsumGm, this->expertNum, 0);
    }
    if (this->expertTokensBeforeCapacityFlag == 1) {
      expertTokensBeforeCapacityGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensBeforeCapacity,
                                                   Align(this->expertNum, sizeof(int32_t)));
      InitGlobalMemory(expertTokensBeforeCapacityGm, this->expertNum, 0);
    }
  }
  // key and value
  int64_t kvFactor = 2;
  int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
  pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
  pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
  pipe->InitBuffer(tempBuffer, buffSize);
  pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void MoeV2SortOneCore::Process() {
  if (GetBlockIdx() < 1) {
    CopyIn();
    SortCompute();
    CopyOut();
  }
  this->SyncAll();
}
}  // namespace MoeInitRoutingV2
#endif  // MOE_V2_SORT_ONE_CORE_H