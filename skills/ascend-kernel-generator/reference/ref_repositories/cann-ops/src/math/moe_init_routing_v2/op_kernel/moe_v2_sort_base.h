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
 * \file moe_v2_sort_base.h
 * \brief
 */
   
#ifndef MOE_V2_SORT_BASE_H
#define MOE_V2_SORT_BASE_H

#include "kernel_operator.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

class MoeV2SortBase {
 public:
  __aicore__ inline MoeV2SortBase(){};

 protected:
  __aicore__ inline void SyncAll();

 protected:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
  TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
  TBuf<TPosition::VECCALC> tempBuffer;
  TBuf<TPosition::VECCALC> sortedBuffer;

  GlobalTensor<int32_t> expertIdxGm;
  GlobalTensor<int32_t> sortedexpertIdxGm;
  GlobalTensor<int32_t> expandDstToSrcRowGm;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

  int64_t tileLength;
  int64_t bufferNum = 1;
  int64_t totalLength;
  int64_t coreNum;
  int64_t n;
  int64_t k;
  int64_t existRowIdx;
  int64_t expertNum;
  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;

  static constexpr int64_t SYNC_GM_NUM = 2;
  static constexpr int64_t WORK_GM_NUM = 2;
  static constexpr int64_t DST_BLK_STRIDE = 1;
  static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeV2SortBase::SyncAll() {
  if (coreNum == 1) {
    return;
  }
#ifndef __CCE_KT_TEST__
  AscendC::SyncAll();
#endif
}

}  // namespace MoeInitRoutingV2
#endif  // MOE_V2_SORT_BASE_H