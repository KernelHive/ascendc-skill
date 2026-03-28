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
 * \file moe_v2_gather_out.h
 * \brief
 */
 
#ifndef MOE_V2_GATHER_OUT_H
#define MOE_V2_GATHER_OUT_H

#include "moe_v2_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;

template <typename T>
class MoeV2GatherOut {
 public:
  __aicore__ inline MoeV2GatherOut(){};
  __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR expandedRowIdx,GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedX, GM_ADDR workspace,
                              const MoeInitRoutingV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyInIndices(int64_t progress);
  __aicore__ inline void Compute(int64_t progress);
  __aicore__ inline void CopyOut(int64_t progress);

 private:
  TPipe* pipe;
  TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inputActivationsCopyInQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> expandDstToSrcRowCopyInQueue, inQueue_total_rows;

  GlobalTensor<T> inputXGm;
  GlobalTensor<T> expandedXGm;
  GlobalTensor<int32_t> expandedRowIdxGm;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;

  const MoeV2GatherOutComputeTilingData* gatherOutTilingData;

  int64_t needCoreNum;
  int64_t blockIdx;
  int64_t cols;
  int64_t n;
  int64_t k;
  int64_t activateRows;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;
  int64_t rowLoops;
  int64_t colsTileLength;
  int64_t perLoopCols;
  int64_t lastLoopCols;
  int64_t colLoops;
  int64_t dropPadMode;
  int64_t expertNum;

  int64_t indicesOffset;
  int64_t inputOffset;
  int64_t outOffset;

  int64_t start_expertId;
  int64_t end_expertId;
  int32_t start_row_;
  int32_t end_row_;
  int64_t device_id_;
  int32_t align_cnt_total_rows;
  int32_t per_dim_total_;
  int32_t BASE_SIZE_TOTAL;
  int64_t core_id_;
};

template <typename T>
__aicore__ inline void MoeV2GatherOut<T>::CopyInIndices(int64_t progress) {
  this->indicesOffset = progress * this->perLoopRows;
  LocalTensor<int32_t> indicesLocal = expandDstToSrcRowCopyInQueue.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
  DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(indicesLocal, expandedRowIdxGm[indicesOffset], dataCopyParams, dataCopyPadParams);

  expandDstToSrcRowCopyInQueue.EnQue<int32_t>(indicesLocal);

  if(progress == 0)
  {
      LocalTensor<int32_t> total_rowsLocal = inQueue_total_rows.AllocTensor<int32_t>();
      DataCopy(total_rowsLocal, expertTokensCountOrCumsumGm[0], align_cnt_total_rows);
      inQueue_total_rows.EnQue(total_rowsLocal);
  }
}

template <typename T>
__aicore__ inline void MoeV2GatherOut<T>::Compute(int64_t progress) {
  LocalTensor<int32_t> total_rowsLocal = inQueue_total_rows.DeQue<int32_t>();
  this->start_row_ = 0;
  if(this->device_id_ != 0)
  {
    this->start_row_ = total_rowsLocal.GetValue(this->start_expertId);
  }
  this->end_row_ = total_rowsLocal.GetValue(this->end_expertId);
  inQueue_total_rows.FreeTensor(total_rowsLocal);

}

template <typename T>
__aicore__ inline void MoeV2GatherOut<T>::CopyOut(int64_t progress) {
  LocalTensor<int32_t> indicesLocal = expandDstToSrcRowCopyInQueue.DeQue<int32_t>();
  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  colsTileLength = this->perLoopCols;
  for (int64_t colsLoop = 0; colsLoop < this->colLoops; colsLoop++) {
    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;
    if (colsLoop == this->colLoops - 1) {
      colsTileLength = this->lastLoopCols;
    }
    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
      LocalTensor<T> inLocal = inputActivationsCopyInQueue.AllocTensor<T>();
      // input row position
      inputOffset = row * this->cols + colsLoop * this->perLoopCols;
      DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->colsTileLength * sizeof(T)), 0, 0, 0};
      DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
      DataCopyPad(inLocal, inputXGm[inputOffset], dataCopyParams, dataCopyPadParams);
      SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

      DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->colsTileLength * sizeof(T)), 0, 0, 0};
      while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
        int32_t outIndex = indicesLocal.GetValue(curLoopRow);
        curLoopRow++;
        initialRow++;
        if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
          continue;
        }

      if(outIndex >= 0 && outIndex < this->end_row_ - this->start_row_){
        outOffset = outIndex * cols + colsLoop * this->perLoopCols;
        DataCopyPad(expandedXGm[ outOffset], inLocal, intriParams);
      }
      }
      inputActivationsCopyInQueue.FreeTensor(inLocal);
    }
  }
  expandDstToSrcRowCopyInQueue.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOut<T>::Init(GM_ADDR inputX, GM_ADDR expandedRowIdx,GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedX,
                                               GM_ADDR workspace, const MoeInitRoutingV2TilingData* tilingData,
                                               TPipe* tPipe) {
  this->pipe = tPipe;
  this->blockIdx = GetBlockIdx();
  this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

  this->needCoreNum = this->gatherOutTilingData->needCoreNum;
  this->activateRows = this->gatherOutTilingData->activateRows;
  this->cols = tilingData->cols;
  this->n = tilingData->n;
  this->k = tilingData->k;
  this->dropPadMode = tilingData->dropPadMode;

  this->expertNum = tilingData->expertNum;
  this->start_expertId = tilingData->start_expertId;
  this->end_expertId = tilingData->end_expertId;
  this->device_id_ = tilingData->device_id;
  this->core_id_ = GetBlockIdx();
  BASE_SIZE_TOTAL = 32/sizeof(int32_t);
  per_dim_total_ = this->expertNum;// 256
  align_cnt_total_rows = DivCeil(per_dim_total_, BASE_SIZE_TOTAL)* BASE_SIZE_TOTAL;


  if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
    this->coreRows = this->gatherOutTilingData->lastCoreRows;
    this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
    this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
    this->rowLoops = this->gatherOutTilingData->lastCoreLoops;
  } else {
    this->coreRows = this->gatherOutTilingData->perCoreRows;
    this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
    this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
    this->rowLoops = this->gatherOutTilingData->perCoreLoops;
  }
  this->perLoopCols = this->gatherOutTilingData->perLoopCols;
  this->lastLoopCols = this->gatherOutTilingData->lastLoopCols;
  this->colLoops = this->gatherOutTilingData->colLoops;

  inputXGm.SetGlobalBuffer((__gm__ T*)inputX, this->coreRows * this->cols);
  expandedXGm.SetGlobalBuffer((__gm__ T*)expandedX, tilingData->n * tilingData->k * this->cols);

  expertTokensCountOrCumsumGm.SetGlobalBuffer(( __gm__ int32_t*)expertTokensCountOrCumsum, this->expertNum);
  pipe->InitBuffer(inQueue_total_rows, 1, align_cnt_total_rows*sizeof(int32_t));

  expandedRowIdxGm.SetGlobalBuffer(
      (__gm__ int32_t*)expandedRowIdx + this->blockIdx * this->gatherOutTilingData->perCoreRows,
      Align(this->coreRows, sizeof(int32_t)));

  pipe->InitBuffer(inputActivationsCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopCols, sizeof(T)));
  pipe->InitBuffer(expandDstToSrcRowCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopRows, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV2GatherOut<T>::Process() {
  if (this->blockIdx < this->needCoreNum) {
    currentLoopRows = perLoopRows;
    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
      if (loop == this->rowLoops - 1) {
        currentLoopRows = lastLoopRows;
      }
      CopyInIndices(loop);
      if(loop == 0){
        Compute(loop);
      }

      CopyOut(loop);
    }
  }
}
}  // namespace MoeInitRoutingV2
#endif  // MOE_V2_GATHER_OUT_H
