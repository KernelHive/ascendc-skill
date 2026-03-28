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
 * \file moe_v2_src_to_dst_op.h
 * \brief
 */
   
#ifndef MOE_V2_SRC_TO_DST_H
#define MOE_V2_SRC_TO_DST_H

#include "moe_v2_common.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

class MoeV2SrcToDstOp {
 public:
  __aicore__ inline MoeV2SrcToDstOp(){};
  template <typename TilingData>
  __aicore__ inline void Init(GM_ADDR expandSrcToDstRow, GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace, const TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t progress);
  __aicore__ inline void Compute(int64_t progress);
  __aicore__ inline void ComputeTotalRows();
  __aicore__ inline void CopyInTotalRows();
  __aicore__ inline void CopyOut();

  __aicore__ inline void SyncAll();
  __aicore__ inline void AssistInit();

 private:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> copyInQueue, inQueue_total_rows;
  TQue<QuePosition::VECOUT, 1> copyOutQueue;
  TBuf<TPosition::VECCALC> assistBuffer;

  GlobalTensor<int32_t> expandDstToSrcRowGm;
  GlobalTensor<int32_t> expandSrcToDstRowGm;
  GlobalTensor<int32_t> assistGm;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;

  const MoeV2GatherOutComputeTilingData* srcToDstTilingData;

  int64_t coreNum;
  int64_t blockIdx;
  int64_t totalLength;
  int64_t currentLoopRows;
  int64_t coreRows;
  int64_t perLoopRows;
  int64_t lastLoopRows;

  int64_t expertNum;
  int64_t start_expertId_;
  int32_t start_row_;
  int64_t device_id_;
  int32_t align_cnt_total_rows;
  int32_t per_dim_total_;
  int32_t BASE_SIZE_TOTAL;




};


__aicore__ inline void MoeV2SrcToDstOp::AssistInit() {
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
  OOMCheckAddrRange(assistGm.GetPhyAddr(), ASSIST_NUM * sizeof(int32_t));
#endif
  LocalTensor<int32_t> assistTensor = assistBuffer.Get<int32_t>(ASSIST_NUM);
  DataCopy(assistTensor, assistGm, ASSIST_NUM);
  SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
  Adds(assistTensor, assistTensor, (int32_t)(this->blockIdx * this->srcToDstTilingData->perCoreRows), ASSIST_NUM);
}


__aicore__ inline void MoeV2SrcToDstOp::CopyIn(int64_t progress) {
  LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
  DataCopy(inLocal, expandDstToSrcRowGm[progress * perLoopRows], Align(currentLoopRows, sizeof(int32_t)));
  copyInQueue.EnQue<int32_t>(inLocal);
}




__aicore__ inline void MoeV2SrcToDstOp::Compute(int64_t progress) {
  LocalTensor<int32_t> outLocal = copyOutQueue.AllocTensor<int32_t>();
  LocalTensor<int32_t> assistTensor = assistBuffer.Get<int32_t>(ASSIST_NUM);

  pipe_barrier(PIPE_V);
  int64_t loops = Ceil(currentLoopRows, ASSIST_INDEX_NUM);
  for (int64_t i = 0; i < loops; i++) {
    Adds(outLocal[i * ASSIST_NUM], assistTensor,
         static_cast<int32_t>(this->perLoopRows * progress + i * ASSIST_INDEX_NUM), ASSIST_NUM);
         
  }
  pipe_barrier(PIPE_V);
  copyOutQueue.EnQue<int32_t>(outLocal);
}


__aicore__ inline void MoeV2SrcToDstOp::CopyInTotalRows() {

      LocalTensor<int32_t> total_rowsLocal = inQueue_total_rows.AllocTensor<int32_t>();
      DataCopy(total_rowsLocal, expertTokensCountOrCumsumGm[0], align_cnt_total_rows);
      inQueue_total_rows.EnQue(total_rowsLocal);
}

__aicore__ inline void MoeV2SrcToDstOp::ComputeTotalRows() {

  LocalTensor<int32_t> total_rowsLocal = inQueue_total_rows.DeQue<int32_t>();
  this->start_row_ = 0;
  if(this->device_id_ != 0)
  {
    this->start_row_ = total_rowsLocal.GetValue(this->start_expertId_);
  }
  inQueue_total_rows.FreeTensor(total_rowsLocal);

}

__aicore__ inline void MoeV2SrcToDstOp::CopyOut() {
  LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> outLocal = copyOutQueue.DeQue<int32_t>();
  SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
  DataCopyParams intriParams;
  intriParams.blockCount = 1;
  intriParams.blockLen = sizeof(int32_t);
  uint32_t outOffset;
  for (int64_t idx = 0; idx < currentLoopRows; idx++) {
    outOffset = inLocal.GetValue(idx);
    Adds(outLocal[idx * INT32_ONE_BLOCK_NUM], outLocal[idx * INT32_ONE_BLOCK_NUM], -1 * this->start_row_, 1);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(expandSrcToDstRowGm[outOffset], outLocal[idx * INT32_ONE_BLOCK_NUM], intriParams);
  }

  copyInQueue.FreeTensor(inLocal);
  copyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeV2SrcToDstOp::SyncAll() {
  if (coreNum == 1) {
    return;
  }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif
}

template <typename TilingData>
__aicore__ inline void MoeV2SrcToDstOp::Init(GM_ADDR expandSrcToDstRow,  GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                             const TilingData* tilingData, TPipe* tPipe) {
  int64_t blockNum = GetBlockNum();
  this->pipe = tPipe;
  this->blockIdx = GetBlockIdx();

  this->coreNum = tilingData->coreNum;
  this->totalLength = tilingData->n * tilingData->k;
  this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);

  this->expertNum = tilingData->expertNum;
  this->start_expertId_ = tilingData->start_expertId;
  this->device_id_ = tilingData -> device_id;
  BASE_SIZE_TOTAL = 32/sizeof(int32_t);
  per_dim_total_ = this->expertNum;// 256
  align_cnt_total_rows = DivCeil(per_dim_total_, BASE_SIZE_TOTAL)* BASE_SIZE_TOTAL;

  if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
    this->coreRows = this->srcToDstTilingData->lastCoreRows;
    this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
  } else {
    this->coreRows = this->srcToDstTilingData->perCoreRows;
    this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
    this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
  }
  expertTokensCountOrCumsumGm.SetGlobalBuffer(( __gm__ int32_t*)expertTokensCountOrCumsum, this->expertNum);
  pipe->InitBuffer(inQueue_total_rows, 1, align_cnt_total_rows*sizeof(int32_t));


  expandSrcToDstRowGm.SetGlobalBuffer((__gm__ int32_t*)expandSrcToDstRow, Align(this->totalLength, sizeof(int32_t)));
  expandDstToSrcRowGm.SetGlobalBuffer((__gm__ int32_t*)workspace + Align(this->totalLength, sizeof(int32_t)) +
                                          this->blockIdx * this->srcToDstTilingData->perCoreRows,
                                      Align(this->coreRows, sizeof(int32_t)));
  assistGm.SetGlobalBuffer((__gm__ int32_t*)assist, ASSIST_NUM);

  pipe->InitBuffer(copyInQueue, 1, this->perLoopRows * BLOCK_BYTES);
  pipe->InitBuffer(copyOutQueue, 1, Ceil(this->perLoopRows, ASSIST_NUM) * ASSIST_NUM * BLOCK_BYTES);
  pipe->InitBuffer(assistBuffer, ASSIST_NUM * sizeof(int32_t));
}

__aicore__ inline void MoeV2SrcToDstOp::Process() {
  if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
    int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
    currentLoopRows = perLoopRows;
    AssistInit();
    CopyInTotalRows();
    ComputeTotalRows();
    for (int64_t loop = 0; loop < loops - 1; loop++) {
      CopyIn(loop);
      Compute(loop);
      CopyOut();
    }
    currentLoopRows = lastLoopRows;
    CopyIn(loops - 1);
    Compute(loops - 1);
    CopyOut();
  }
  this->SyncAll();
}
}  // namespace MoeInitRoutingV2
#endif  // MOE_V2_SRC_TO_DST_H