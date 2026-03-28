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
 * \file max_pool3d_with_argmax_v2_no_expand_indices.h
 * \brief
 */

#ifndef MAX_POOL3D_WITH_ARGMAX_V2_NO_EXPAND_INDICES_H_
#define MAX_POOL3D_WITH_ARGMAX_V2_NO_EXPAND_INDICES_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

__aicore__ inline uint64_t FloorValue(uint64_t x, uint64_t y) {
  return y == 0 ? x : x / y * y;
}
__aicore__ inline uint64_t CeilValue(uint64_t x, uint64_t y) {
  return y == 0 ? x : (x + y - 1) / y * y;
}
__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y) {
  return y == 0 ? x : (x + y - 1) / y;
}
template <typename T1, typename T2, const uint32_t IS_PAD = 0>
class MaxPool3DWithArgmaxV2NoExpandIndices {
 public:
  __aicore__ inline MaxPool3DWithArgmaxV2NoExpandIndices(void){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, TPipe* pipe_in,
                              const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict tiling);
  __aicore__ inline void Process(void);
  __aicore__ inline void CalNextIdxData(uint64_t idx);
  __aicore__ inline void CalCurIdxData(uint64_t idx);
  __aicore__ inline void BaseCompute(uint64_t idx, uint64_t endIdx);
#if ORIG_DTYPE_X == DT_BF16
  __aicore__ inline void CastBF16ToF32(LocalTensor<T2> xLocal, LocalTensor<T1> origXLocal,
                                       uint8_t castRepeateTimes, uint8_t castRepeateStride);
#endif
  __aicore__ inline void ExtPadInput(LocalTensor<T2> xLocal, uint8_t padRepeatTimes, uint8_t padRepeatStride);
  __aicore__ inline void CopyInputNcRepeat(LocalTensor<T2> xLocal, uint64_t xGmOffset,
                                           DataCopyExtParams& extParams, DataCopyPadExtParams<T1>& padExtParams);
  __aicore__ inline void CopyInputDxRepeat(LocalTensor<T2> xLocal, uint64_t xGmOffset,
                                           DataCopyExtParams& extParams, DataCopyPadExtParams<T1>& padExtParams);
  __aicore__ inline void CopyInputHxRepeat(LocalTensor<T2> xLocal, uint64_t xGmOffset,
                                           DataCopyExtParams& extParams, DataCopyPadExtParams<T1>& padExtParams);
  __aicore__ inline void CopyInput(LocalTensor<T2> xLocal);
  template<typename U>
  __aicore__ inline void Transpose(LocalTensor<U> xTLocal, LocalTensor<U> xLocal, uint64_t rowNum, uint64_t colNum);
  __aicore__ inline void PoolMaxAndIndices(LocalTensor<T2> maxLocal, LocalTensor<int32_t> indicesLocal,
                                           LocalTensor<T2> inputLocal);
#if ORIG_DTYPE_X == DT_BF16
  __aicore__ inline void CastF32ToBF16(LocalTensor<T1> origMaxOutLocal, LocalTensor<T2> maxOutLocal,
                                       uint8_t castRepeateTimes, uint8_t castRepeateStride);
#endif
  __aicore__ inline void CopyMaxOutNcRepeat(LocalTensor<T2> maxTransLocal, uint64_t maxGmOffset,
                                            DataCopyExtParams& extParams);
  __aicore__ inline void CopyMaxOutDyRepeat(LocalTensor<T2> maxTransLocal, uint64_t maxGmOffset,
                                            DataCopyExtParams& extParams);
  __aicore__ inline void CopyMaxOutHyRepeat(LocalTensor<T2> maxTransLocal, uint64_t maxGmOffset,
                                            DataCopyExtParams& extParams);
  __aicore__ inline void CopyMaxOut(LocalTensor<T2> maxTransLocal);
  __aicore__ inline void CopyIndicesOutNcRepeat(LocalTensor<int32_t> indicesTransLocal, uint64_t indicesGmOffset,
                                                DataCopyExtParams& extParams);
  __aicore__ inline void CopyIndicesOutDyRepeat(LocalTensor<int32_t> indicesTransLocal, uint64_t indicesGmOffset,
                                                DataCopyExtParams& extParams);
  __aicore__ inline void CopyIndicesOutHyRepeat(LocalTensor<int32_t> indicesTransLocal, uint64_t indicesGmOffset,
                                                DataCopyExtParams& extParams);
  __aicore__ inline void CopyIndicesOut(LocalTensor<int32_t> indicesTransLocal);

  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> inputQue;
  TQue<QuePosition::VECOUT, 1> maxQue;
  TQue<QuePosition::VECOUT, 1> indicesQue;
  TBuf<> inputBuffer;
  TBuf<> maxBuffer;
  TBuf<> indicesBuffer;
  TBuf<> indicesUpdateBuffer;
  TBuf<> indicesTemplateBuffer;
  TBuf<> gtBuffer;
  TBuf<> geBuffer;

  GlobalTensor<T1> xGm, maxGm;
  GlobalTensor<int32_t> indicesGm;
  LocalTensor<int32_t> indicesTemplateLocal;

  const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* tilingData;

  uint32_t cBlockIdx;
  // variable from tiling
  uint64_t nc = 1;
  uint64_t dx = 1;
  uint64_t hx = 1;
  uint64_t wx = 1;

  uint64_t kd = 1;
  uint64_t kh = 1;
  uint64_t kw = 1;

  uint64_t dy = 1;
  uint64_t hy = 1;
  uint64_t wy = 1;

  uint64_t sd = 1;
  uint64_t sh = 1;
  uint64_t sw = 1;

  uint64_t pf = 1;
  uint64_t pb = 1;
  uint64_t pt = 1;
  uint64_t pd = 1;
  uint64_t pl = 1;
  uint64_t pr = 1;

  uint64_t ncFactor = 1;
  uint64_t ncTail = 1;
  uint64_t ncOuter = 1;

  uint64_t dyFactor = 1;
  uint64_t dyTail = 1;
  uint64_t dyOuter = 1;

  uint64_t hyFactor = 1;
  uint64_t hyTail = 1;
  uint64_t hyOuter = 1;

  uint64_t wyFactor = 1;
  uint64_t wyTail = 1;
  uint64_t wyOuter = 1;

  uint64_t blockFactor = 1;
  uint64_t blockTail = 1;
  uint64_t totalIdx = 1;
  uint64_t coreNums = 1;

  uint64_t inputBufferSize = 1;
  uint64_t outputMaxBufferSize = 1;
  uint64_t outputIndiceBufferSize = 1;
  uint64_t indiceTempBufferSize = 1;
  uint64_t maskBufferSize = 1;

  // self-defined variable
  uint64_t dxFactor = 1;
  uint64_t hxFactor = 1;
  uint64_t wxFactor = 1;

  // cur
  uint64_t curNcFactor = 1;
  uint64_t curNcFactorAlign = 1;

  uint64_t curDyFactor = 1;
  uint64_t curHyFactor = 1;

  uint64_t curWyFactor = 1;
  uint64_t curWyFactorAlign = 1;

  uint64_t curDxFactor = 1;
  uint64_t curHxFactor = 1;

  uint64_t curWxFactor = 1;
  uint64_t curWxFactorAlign = 1;

  uint64_t curNcIdx = 0;
  uint64_t curDyIdx = 0;
  uint64_t curHyIdx = 0;
  uint64_t curWyIdx = 0;

  // next
  uint64_t nextNcFactor = 1;

  uint64_t nextDyFactor = 1;
  uint64_t nextHyFactor = 1;

  uint64_t nextWyFactor = 1;

  uint64_t nextDxFactor = 1;
  uint64_t nextHxFactor = 1;

  uint64_t nextWxFactor = 1;
  uint64_t nextWxFactorAlign = 1;

  uint64_t nextNcIdx = 0;
  uint64_t nextDyIdx = 0;
  uint64_t nextHyIdx = 0;
  uint64_t nextWyIdx = 0;

  constexpr static uint32_t ENABLE = 1;
  constexpr static uint32_t DISABLE = 0;
  constexpr static uint16_t ORMASK = 0xFFFF;
  constexpr static uint32_t ORMASK_SIZE = 128;
  constexpr static uint32_t ORMASK_BUFFER_SIZE = ORMASK_SIZE * sizeof(uint16_t);
  constexpr static uint64_t BLOCK_DATA = 32;
  constexpr static uint64_t BYTE_T1 = sizeof(T1);
  constexpr static uint64_t BYTE_T2 = sizeof(T2);
  constexpr static uint64_t BLOCK_ALIGN_T1 = BLOCK_DATA / BYTE_T1;
  constexpr static uint64_t BLOCK_ALIGN_T2 = BLOCK_DATA / BYTE_T2;
  constexpr static uint64_t BYTE_INDICES = sizeof(int32_t);
  constexpr static uint64_t BLOCK_ALIGN_INDICES = BLOCK_DATA / BYTE_INDICES;
  constexpr static uint64_t TRANS_ALIGN = 16;
  constexpr static uint64_t VEC_INST_CAL_DATA_NUM = 256 / BYTE_T2;
  constexpr static uint64_t INT32_DIV_T2_SIZE = sizeof(int32_t) / sizeof(T2);
  constexpr static uint64_t UINT16_BITS = 16;
  constexpr static uint64_t VEC_INST_CAL_MASK_NUM = VEC_INST_CAL_DATA_NUM / UINT16_BITS;
  constexpr static uint64_t VEC_MAX_REPEAT_TIMES = 255;
  constexpr static uint64_t VEC_MAX_REPEAT_STRIDE = 255;
  constexpr static uint64_t MTE_MAX_GM_STRIDE = 4294967295;

  TBuf<> orBuffer;
  LocalTensor<uint16_t> orMaskLocal;

  // cur
  int64_t curPfPadFactor = 1;
  int64_t curPbPadFactor = 1;
  int64_t curPtPadFactor = 1;
  int64_t curPdPadFactor = 1;
  int64_t curPlPadFactor = 1;
  int64_t curPrPadFactor = 1;

  // next
  uint64_t nextCopyInWxFactor = 1;

  int64_t nextPfPadFactor = 1;
  int64_t nextPbPadFactor = 1;
  int64_t nextPtPadFactor = 1;
  int64_t nextPdPadFactor = 1;
  int64_t nextPlPadFactor = 1;
  int64_t nextPrPadFactor = 1;

  int64_t nextExtPlPadFactor = 0;
  int64_t nextExtPrPadFactor = 0;

  constexpr static uint64_t MAX_PAD_NUM = BLOCK_DATA / BYTE_T1;
  constexpr static uint64_t BLOCK_ALIGN_MASK = BLOCK_DATA / sizeof(uint16_t);
#if ORIG_DTYPE_X == DT_FLOAT
  constexpr static int32_t NEG_INF_T1 = 0xFF800000;
  constexpr static int32_t NEG_INF_T2 = 0xFF800000;
#elif ORIG_DTYPE_X == DT_FLOAT16
  constexpr static int32_t NEG_INF_T1 = 0xFC00;
  constexpr static int32_t NEG_INF_T2 = 0xFC00;
#elif ORIG_DTYPE_X == DT_BF16
  constexpr static int32_t NEG_INF_T1 = 0xFF80;
  constexpr static int32_t NEG_INF_T2 = 0xFF800000;
#endif
};

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::Init(
                       GM_ADDR x, GM_ADDR y, GM_ADDR indices, TPipe* pipe_in,
                       const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict tiling) {
  // variable from tiling
  tilingData = tiling;
  nc = tilingData->nc;

  dx = tilingData->dx;
  hx = tilingData->hx;
  wx = tilingData->wx;

  kd = tilingData->kd;
  kh = tilingData->kh;
  kw = tilingData->kw;

  sd = tilingData->sd;
  sh = tilingData->sh;
  sw = tilingData->sw;

  pf = tilingData->pf;
  pb = tilingData->pb;
  pl = tilingData->pl;
  pr = tilingData->pr;
  pt = tilingData->pt;
  pd = tilingData->pd;

  dy = tilingData->dy;
  hy = tilingData->hy;
  wy = tilingData->wy;

  ncFactor = tilingData->ncFactor;
  ncTail = tilingData->ncTail;
  ncOuter = tilingData->ncOuter;

  dyFactor = tilingData->dyFactor;
  dyTail = tilingData->dyTail;
  dyOuter = tilingData->dyOuter;

  hyFactor = tilingData->hyFactor;
  hyTail = tilingData->hyTail;
  hyOuter = tilingData->hyOuter;

  wyFactor = tilingData->wyFactor;
  wyTail = tilingData->wyTail;
  wyOuter = tilingData->wyOuter;

  blockFactor = tilingData->blockFactor;
  blockTail = tilingData->blockTail;
  totalIdx = tilingData->totalIdx;
  coreNums = tilingData->coreNums;

  inputBufferSize = tilingData->inputBufferSize;
  outputMaxBufferSize = tilingData->outputMaxBufferSize;
  outputIndiceBufferSize = tilingData->outputIndiceBufferSize;
  indiceTempBufferSize = tilingData->indiceTempBufferSize;
  maskBufferSize = tilingData->maskBufferSize;

  // self-defined variable
  dxFactor = (dyFactor - 1) * sd + kd;
  hxFactor = (hyFactor - 1) * sh + kh;
  wxFactor = (wyFactor - 1) * sw + kw;

  // base info
  pipe = pipe_in;
  cBlockIdx = GetBlockIdx();
  // GM
  xGm.SetGlobalBuffer((__gm__ T1*)x);
  maxGm.SetGlobalBuffer((__gm__ T1*)y);
  indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);

  // buffer init
  // que init
  pipe->InitBuffer(inputQue, 1, inputBufferSize);
  pipe->InitBuffer(maxQue, 1, outputMaxBufferSize);
  pipe->InitBuffer(indicesQue, 1, outputIndiceBufferSize);
  // tbuffer init
  pipe->InitBuffer(inputBuffer, inputBufferSize);
  pipe->InitBuffer(maxBuffer, outputMaxBufferSize);
  pipe->InitBuffer(indicesBuffer, outputIndiceBufferSize);
  pipe->InitBuffer(indicesUpdateBuffer, indiceTempBufferSize);
  pipe->InitBuffer(indicesTemplateBuffer, indiceTempBufferSize);
  pipe->InitBuffer(gtBuffer, maskBufferSize);
  pipe->InitBuffer(geBuffer, maskBufferSize);
  // init indices
  indicesTemplateLocal = indicesTemplateBuffer.Get<int32_t>();
  int32_t indicesValue = 0;
  uint64_t indicesOffset = 0;
  for (int32_t idx = 0; idx < wyFactor; idx++) {
    Duplicate(indicesTemplateLocal[indicesOffset], indicesValue, VEC_INST_CAL_DATA_NUM);
    indicesValue += sw;
    indicesOffset += VEC_INST_CAL_DATA_NUM;
  }
  if constexpr (IS_PAD == ENABLE) {
    pipe->InitBuffer(orBuffer, ORMASK_BUFFER_SIZE);
    orMaskLocal = orBuffer.Get<uint16_t>();
    Duplicate(orMaskLocal, ORMASK, ORMASK_SIZE);
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CalNextIdxData(uint64_t idx) {
  nextNcIdx = idx / (dyOuter * hyOuter * wyOuter);
  nextNcFactor = nextNcIdx == (ncOuter - 1) ? ncTail : ncFactor;

  uint64_t loopIdx = idx % (dyOuter * hyOuter * wyOuter);

  nextDyIdx = loopIdx / (hyOuter * wyOuter);
  nextDyFactor = nextDyIdx == (dyOuter - 1) ? dyTail : dyFactor;

  nextHyIdx = loopIdx % (hyOuter * wyOuter) / wyOuter;
  nextHyFactor = nextHyIdx == (hyOuter - 1) ? hyTail : hyFactor;

  nextWyIdx = loopIdx % wyOuter;
  nextWyFactor = nextWyIdx == (wyOuter - 1) ? wyTail : wyFactor;

  nextDxFactor = (nextDyFactor - 1) * sd + kd;
  nextHxFactor = (nextHyFactor - 1) * sh + kh;
  nextWxFactor = (nextWyFactor - 1) * sw + kw;

  nextWxFactorAlign = CeilValue(nextWxFactor, BLOCK_ALIGN_T1);

  if constexpr (IS_PAD == ENABLE) {
    nextPfPadFactor = pf - nextDyIdx * dyFactor * sd;
    nextPfPadFactor = nextPfPadFactor > 0 ? nextPfPadFactor : 0;

    nextPbPadFactor = (nextDyIdx * dyFactor + nextDyFactor - 1) * sd + kd - dx - pf;
    nextPbPadFactor = nextPbPadFactor > 0 ? nextPbPadFactor : 0;

    nextPtPadFactor = pt - nextHyIdx * hyFactor * sh;
    nextPtPadFactor = nextPtPadFactor > 0 ? nextPtPadFactor : 0;

    nextPdPadFactor = (nextHyIdx * hyFactor + nextHyFactor - 1) * sh + kh - hx - pt;
    nextPdPadFactor = nextPdPadFactor > 0 ? nextPdPadFactor : 0;

    nextPlPadFactor = pl - nextWyIdx * wyFactor * sw;
    nextPlPadFactor = nextPlPadFactor > 0 ? nextPlPadFactor : 0;

    nextPrPadFactor = (nextWyIdx * wyFactor + nextWyFactor - 1) * sw + kw - wx - pl;
    nextPrPadFactor = nextPrPadFactor > 0 ? nextPrPadFactor : 0;

    nextDxFactor -= (nextPfPadFactor + nextPbPadFactor);
    nextHxFactor -= (nextPtPadFactor + nextPdPadFactor);
    nextCopyInWxFactor = nextWxFactor - (nextPlPadFactor + nextPrPadFactor);

    nextExtPlPadFactor = FloorValue(nextPlPadFactor, MAX_PAD_NUM);
    nextExtPrPadFactor = FloorValue(nextPrPadFactor, MAX_PAD_NUM);
    /*
      How to cal padding num：
      (1) When the padding num <= max_pad_num, the nextExtPlPadFactor = 0, nextExtPrPadFactor = 0
      (2) When the padding num >  max_pad_num, the nextExtPlPadFactor = nextPlPadFactor / max_pad_num * max_pad_num to align block.
          the nextExtPrPadFactor is set to nextWxFactor - CeilValue(nextPlPadFactor + nextCopyInWxFactor, BLOCK_ALIGN_T2). For ex:
          pading is 9, kernel is 20, nextCopyInWxFactor = 20;
          [p  p  p  p  p  p  p  p  p  n  n  n  n ... n  n  n  n  p  p  p  p  p  p  p  p  p  p]
          |---nextExtPlPadFactor--|-|---nextCopyInWxFactor-----|-------move pad---------|----|
                                   ↑ move pad                                             ↑ nextExtPrPadFactor
    */
    if (nextExtPrPadFactor != 0) {
      nextExtPrPadFactor = nextWxFactor - CeilValue(nextPlPadFactor + nextCopyInWxFactor, BLOCK_ALIGN_T2);
    }
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CalCurIdxData(uint64_t idx) {
  curNcIdx = idx / (dyOuter * hyOuter * wyOuter);
  curNcFactor = curNcIdx == (ncOuter - 1) ? ncTail : ncFactor;

  uint64_t loopIdx = idx % (dyOuter * hyOuter * wyOuter);

  curDyIdx = loopIdx / (hyOuter * wyOuter);
  curDyFactor = curDyIdx == (dyOuter - 1) ? dyTail : dyFactor;

  curHyIdx = loopIdx % (hyOuter * wyOuter) / wyOuter;
  curHyFactor = curHyIdx == (hyOuter - 1) ? hyTail : hyFactor;

  curWyIdx = loopIdx % wyOuter;
  curWyFactor = curWyIdx == (wyOuter - 1) ? wyTail : wyFactor;

  curDxFactor = (curDyFactor - 1) * sd + kd;
  curHxFactor = (curHyFactor - 1) * sh + kh;
  curWxFactor = (curWyFactor - 1) * sw + kw;

  curWxFactorAlign = CeilValue(curWxFactor, BLOCK_ALIGN_T1);

  curWyFactorAlign = CeilValue(curWyFactor, TRANS_ALIGN);

  curNcFactorAlign = CeilValue(curNcFactor, TRANS_ALIGN);

  if constexpr (IS_PAD == ENABLE) {
    curPfPadFactor = pf - curDyIdx * dyFactor * sd;
    curPfPadFactor = curPfPadFactor > 0 ? curPfPadFactor : 0;

    curPbPadFactor = (curDyIdx * dyFactor + curDyFactor - 1) * sd + kd - dx - pf;
    curPbPadFactor = curPbPadFactor > 0 ? curPbPadFactor : 0;

    curPtPadFactor = pt - curHyIdx * hyFactor * sh;
    curPtPadFactor = curPtPadFactor > 0 ? curPtPadFactor : 0;

    curPdPadFactor = (curHyIdx * hyFactor + curHyFactor - 1) * sh + kh - hx - pt;
    curPdPadFactor = curPdPadFactor > 0 ? curPdPadFactor : 0;

    curPlPadFactor = pl - curWyIdx * wyFactor * sw;
    curPlPadFactor = curPlPadFactor > 0 ? curPlPadFactor : 0;

    curPrPadFactor = (curWyIdx * wyFactor + curWyFactor - 1) * sw + kw - wx - pl;
    curPrPadFactor = curPrPadFactor > 0 ? curPrPadFactor : 0;

    curDxFactor -= (curPfPadFactor + curPbPadFactor);
    curHxFactor -= (curPtPadFactor + curPdPadFactor);
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::Process(void) {
  if(cBlockIdx >= coreNums) {
    return;
  }
  uint64_t beginIdx = cBlockIdx * blockFactor;
  uint64_t endIdx = beginIdx + blockFactor;
  endIdx = (endIdx > totalIdx) ? totalIdx : endIdx;
  uint64_t loopIdx = beginIdx;
  CalNextIdxData(beginIdx);
  LocalTensor<T2> xLocal = inputQue.AllocTensor<T2>();
  CopyInput(xLocal);
  inputQue.EnQue(xLocal);

  for (uint64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
    CalCurIdxData(curIdx);  
    BaseCompute(curIdx, endIdx);
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::BaseCompute(uint64_t idx, uint64_t endIdx) {
  // transpose input
  LocalTensor<T2> xLocal = inputQue.DeQue<T2>();
  LocalTensor<T2> transLocal = inputBuffer.Get<T2>();
  Transpose(transLocal, xLocal, curNcFactorAlign, curDxFactor * curHxFactor * curWxFactorAlign);
  inputQue.FreeTensor<T2>(xLocal);

  // copy in next
  if (idx < endIdx -1) {
    CalNextIdxData(idx + 1);
    xLocal = inputQue.AllocTensor<T2>();
    CopyInput(xLocal);
    inputQue.EnQue(xLocal);
  }
  // max
  LocalTensor<T2> maxLocal = maxBuffer.Get<T2>();
  LocalTensor<int32_t> indicesLocal = indicesBuffer.Get<int32_t>();
  PoolMaxAndIndices(maxLocal, indicesLocal, transLocal);

  // transpose max
  LocalTensor<T2> maxOutLocal = maxQue.AllocTensor<T2>();
  Transpose(maxOutLocal, maxLocal, curDyFactor * curHyFactor * curWyFactorAlign, curNcFactorAlign);

  // copy max out
  maxQue.EnQue(maxOutLocal);
  maxOutLocal = maxQue.DeQue<T2>();
  CopyMaxOut(maxOutLocal);
  maxQue.FreeTensor<T2>(maxOutLocal);

  // transpose indices
  LocalTensor<int32_t> indicesOutLocal = indicesQue.AllocTensor<int32_t>();
  Transpose(indicesOutLocal, indicesLocal,
            curDyFactor * curHyFactor * curWyFactorAlign, VEC_INST_CAL_DATA_NUM);

  // copy indices out
  indicesQue.EnQue(indicesOutLocal);
  indicesOutLocal = indicesQue.DeQue<int32_t>();
  CopyIndicesOut(indicesOutLocal);
  indicesQue.FreeTensor<int32_t>(indicesOutLocal);
}

#if ORIG_DTYPE_X == DT_BF16
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CastBF16ToF32(LocalTensor<T2> xLocal,
                                                                                           LocalTensor<T1> origXLocal,
                                                                                           uint8_t castRepeateTimes,
                                                                                           uint8_t castRepeateStride) {
  event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
  uint64_t castLoopNum = nextWxFactor / VEC_INST_CAL_DATA_NUM;
  uint64_t castTailNum = nextWxFactor % VEC_INST_CAL_DATA_NUM;
  uint64_t xLocalCastOffset = 0;
  UnaryRepeatParams castParams = {1, 1, castRepeateStride, castRepeateStride};
  for (uint64_t i = 0; i < castLoopNum; i++) {
    Cast(xLocal[xLocalCastOffset], origXLocal[nextWxFactorAlign + xLocalCastOffset],
          RoundMode::CAST_NONE, VEC_INST_CAL_DATA_NUM, castRepeateTimes, castParams);
    xLocalCastOffset += VEC_INST_CAL_DATA_NUM;
  }
  if (castTailNum != 0) {
    Cast(xLocal[xLocalCastOffset], origXLocal[nextWxFactorAlign + xLocalCastOffset],
          RoundMode::CAST_NONE, castTailNum, castRepeateTimes, castParams);
  }
}
#endif

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::ExtPadInput(LocalTensor<T2> xLocal,
                                                                                         uint8_t padRepeatTimes,
                                                                                         uint8_t padRepeatStride) {
  event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
  SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
  WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
#if ORIG_DTYPE_X == DT_BF16
  AscendC::PipeBarrier<PIPE_V>();
#endif
  if constexpr (IS_PAD == ENABLE) {
    uint64_t padLoopNum = nextExtPlPadFactor / VEC_INST_CAL_DATA_NUM;
    uint64_t padTailNum = nextExtPlPadFactor % VEC_INST_CAL_DATA_NUM;
    uint64_t xLocalPadOffset = 0;
    for (uint64_t i = 0; i < padLoopNum; i++) {
      Duplicate(xLocal[xLocalPadOffset], *((T2 *) &NEG_INF_T2),
                VEC_INST_CAL_DATA_NUM, padRepeatTimes, 1, padRepeatStride);
      xLocalPadOffset += VEC_INST_CAL_DATA_NUM;
    }
    if (padTailNum != 0) {
      Duplicate(xLocal[xLocalPadOffset], *((T2 *) &NEG_INF_T2), padTailNum, padRepeatTimes, 1, padRepeatStride);
    }
    padLoopNum = nextExtPrPadFactor / VEC_INST_CAL_DATA_NUM;
    padTailNum = nextExtPrPadFactor % VEC_INST_CAL_DATA_NUM;
    xLocalPadOffset = CeilValue(nextPlPadFactor + nextCopyInWxFactor, BLOCK_ALIGN_T2);
    for (uint64_t i = 0; i < padLoopNum; i++) {
      Duplicate(xLocal[xLocalPadOffset], *((T2 *) &NEG_INF_T2),
                VEC_INST_CAL_DATA_NUM, padRepeatTimes, 1, padRepeatStride);
      xLocalPadOffset += VEC_INST_CAL_DATA_NUM;
    }
    if (padTailNum != 0) {
      Duplicate(xLocal[xLocalPadOffset], *((T2 *) &NEG_INF_T2), padTailNum, padRepeatTimes, 1, padRepeatStride);
    }
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyInputNcRepeat(
                                                                             LocalTensor<T2> xLocal,
                                                                             uint64_t xGmOffset,
                                                                             DataCopyExtParams &extParams,
                                                                             DataCopyPadExtParams<T1> &padExtParams) {
  uint64_t xLocalOffset = 0;
  uint64_t ncDimStride = dx * hx * wx;
  uint64_t vecStride = nextDxFactor * nextHxFactor * nextWxFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = 0;
  uint64_t mteDstStride = 0;
  if constexpr (IS_PAD == ENABLE) {
    mteSrcStride = (ncDimStride - nextCopyInWxFactor) * BYTE_T1;
    mteDstStride = (vecStride / BLOCK_ALIGN_T2) -
                   CeilDiv(padExtParams.leftPadding + nextCopyInWxFactor + padExtParams.rightPadding, BLOCK_ALIGN_T1);
  } else {
    mteSrcStride = (ncDimStride - nextWxFactor) * BYTE_T1;
    mteDstStride = (vecStride / BLOCK_ALIGN_T2) -
                   CeilDiv(nextWxFactor, BLOCK_ALIGN_T1);
  }
  if (mteSrcStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = nextNcFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = nextNcFactor;
  }
  for (uint64_t i = 0; i < nextDxFactor; i++) {
    uint64_t xGmOffsetTmp = xGmOffset;
    uint64_t xLocalOffsetTmp = xLocalOffset;
    for (uint64_t j = 0; j < nextHxFactor; j++) {
      uint64_t xGmMteOffsetTmp = xGmOffsetTmp;
      uint64_t xLocalMteOffsetTmp = xLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
#if ORIG_DTYPE_X == DT_BF16
        LocalTensor<T1> origXLocal = xLocal[xLocalMteOffsetTmp].template ReinterpretCast<T1>();
        DataCopyPad(origXLocal[nextWxFactorAlign + nextExtPlPadFactor], xGm[xGmMteOffsetTmp], extParams, padExtParams);
        CastBF16ToF32(xLocal[xLocalMteOffsetTmp], origXLocal, extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
#else
        DataCopyPad(xLocal[xLocalMteOffsetTmp + nextExtPlPadFactor], xGm[xGmMteOffsetTmp], extParams, padExtParams);
#endif
        if constexpr (IS_PAD == ENABLE) {
          ExtPadInput(xLocal[xLocalMteOffsetTmp], extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
        }
        xGmMteOffsetTmp += ncDimStride;
        xLocalMteOffsetTmp += vecStride;
      }
      xGmOffsetTmp += wx;
      xLocalOffsetTmp += nextWxFactorAlign;
    }
    xGmOffset += hx * wx;
    xLocalOffset += nextHxFactor * nextWxFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyInputDxRepeat(
                                                                             LocalTensor<T2> xLocal,
                                                                             uint64_t xGmOffset,
                                                                             DataCopyExtParams &extParams,
                                                                             DataCopyPadExtParams<T1> &padExtParams) {
  uint64_t xLocalOffset = 0;
  uint64_t ncDimStride = dx * hx * wx;
  uint64_t vecStride = nextHxFactor * nextWxFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = 0;
  uint64_t mteDstStride = 0;
  if constexpr (IS_PAD == ENABLE) {
    mteSrcStride = (hx * wx - nextCopyInWxFactor) * BYTE_T1;
    mteDstStride = (vecStride / BLOCK_ALIGN_T2) -
                   CeilDiv(padExtParams.leftPadding + nextCopyInWxFactor + padExtParams.rightPadding, BLOCK_ALIGN_T1);
  } else {
    mteSrcStride = (hx * wx - nextWxFactor) * BYTE_T1;
    mteDstStride = (vecStride / BLOCK_ALIGN_T2) -
                   CeilDiv(nextWxFactor, BLOCK_ALIGN_T1);
  }
  if (mteSrcStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = nextDxFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = nextDxFactor;
  }
  for (uint64_t i = 0; i < nextNcFactor; i++) {
    uint64_t xGmOffsetTmp = xGmOffset;
    uint64_t xLocalOffsetTmp = xLocalOffset;
    for (uint64_t j = 0; j < nextHxFactor; j++) {
      uint64_t xGmMteOffsetTmp = xGmOffsetTmp;
      uint64_t xLocalMteOffsetTmp = xLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
#if ORIG_DTYPE_X == DT_BF16
        LocalTensor<T1> origXLocal = xLocal[xLocalMteOffsetTmp].template ReinterpretCast<T1>();
        DataCopyPad(origXLocal[nextWxFactorAlign + nextExtPlPadFactor], xGm[xGmMteOffsetTmp], extParams, padExtParams);
        CastBF16ToF32(xLocal[xLocalMteOffsetTmp], origXLocal, extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
#else
        DataCopyPad(xLocal[xLocalMteOffsetTmp + nextExtPlPadFactor], xGm[xGmMteOffsetTmp], extParams, padExtParams);
#endif
        if constexpr (IS_PAD == ENABLE) {
          ExtPadInput(xLocal[xLocalMteOffsetTmp], extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
        }
        xGmMteOffsetTmp += hx * wx;
        xLocalMteOffsetTmp += vecStride;
      }
      xGmOffsetTmp += wx;
      xLocalOffsetTmp += nextWxFactorAlign;
    }
    xGmOffset += ncDimStride;
    xLocalOffset += nextDxFactor * nextHxFactor * nextWxFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyInputHxRepeat(
                                                                             LocalTensor<T2> xLocal,
                                                                             uint64_t xGmOffset,
                                                                             DataCopyExtParams &extParams,
                                                                             DataCopyPadExtParams<T1> &padExtParams) {
  uint64_t xLocalOffset = 0;
  uint64_t ncDimStride = dx * hx * wx;
  uint64_t vecStride = nextWxFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = 0;
  uint64_t mteDstStride = 0;
  if constexpr (IS_PAD == ENABLE) {
    mteSrcStride = (wx - nextCopyInWxFactor) * BYTE_T1;
    mteDstStride = (vecStride / BLOCK_ALIGN_T2) -
                   CeilDiv(padExtParams.leftPadding + nextCopyInWxFactor + padExtParams.rightPadding, BLOCK_ALIGN_T1);
  } else {
    mteSrcStride = ( wx - nextWxFactor) * BYTE_T1;
    mteDstStride = (vecStride / BLOCK_ALIGN_T2) -
                   CeilDiv(nextWxFactor, BLOCK_ALIGN_T1);
  }
  if (mteSrcStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = nextHxFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = nextHxFactor;
  }
  for (uint64_t i = 0; i < nextNcFactor; i++) {
    uint64_t xGmOffsetTmp = xGmOffset;
    uint64_t xLocalOffsetTmp = xLocalOffset;
    for (uint64_t j = 0; j < nextDxFactor; j++) {
      uint64_t xGmMteOffsetTmp = xGmOffsetTmp;
      uint64_t xLocalMteOffsetTmp = xLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
#if ORIG_DTYPE_X == DT_BF16
        LocalTensor<T1> origXLocal = xLocal[xLocalMteOffsetTmp].template ReinterpretCast<T1>();
        DataCopyPad(origXLocal[nextWxFactorAlign + nextExtPlPadFactor], xGm[xGmMteOffsetTmp], extParams, padExtParams);
        CastBF16ToF32(xLocal[xLocalMteOffsetTmp], origXLocal, extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
#else
        DataCopyPad(xLocal[xLocalMteOffsetTmp + nextExtPlPadFactor], xGm[xGmMteOffsetTmp], extParams, padExtParams);
#endif
        if constexpr (IS_PAD == ENABLE) {
          ExtPadInput(xLocal[xLocalMteOffsetTmp], extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
        }
        xGmMteOffsetTmp += wx;
        xLocalMteOffsetTmp += vecStride;
      }
      xGmOffsetTmp += hx * wx;
      xLocalOffsetTmp += nextHxFactor * nextWxFactorAlign;
    }
    xGmOffset += ncDimStride;
    xLocalOffset += nextDxFactor * nextHxFactor * nextWxFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyInput(
                                                                             LocalTensor<T2> xLocal) {
  uint64_t ncDimStride = dx * hx * wx;
  uint64_t ncOffset = nextNcIdx * ncFactor * ncDimStride;
  uint64_t dxOffset = nextDyIdx * sd * dyFactor * hx * wx;
  uint64_t hxOffset = nextHyIdx * sh * hyFactor * wx;
  uint64_t wxOffset = nextWyIdx * sw * wyFactor;

  if constexpr (IS_PAD == ENABLE)
  {
    if (nextPfPadFactor == 0) {
      dxOffset -= pf * hx * wx;
    } else {
      dxOffset = 0;
    }
    if (nextPtPadFactor == 0) {
      hxOffset -= pt * wx;
    } else {
      hxOffset = 0;
    }
    if (nextPlPadFactor == 0) {
      wxOffset -= pl;
    } else {
      wxOffset = 0;
    }
  }

  uint64_t xGmOffset = ncOffset + dxOffset + hxOffset + wxOffset;

  DataCopyExtParams extParams;
  if constexpr (IS_PAD == ENABLE) {
    extParams.blockLen = nextCopyInWxFactor * BYTE_T1;
  } else {
    extParams.blockLen = nextWxFactor * BYTE_T1;
  }

  DataCopyPadExtParams<T1> padExtParams;
  if constexpr (IS_PAD == ENABLE) {
    padExtParams.isPad = true;
    padExtParams.leftPadding = nextPlPadFactor % MAX_PAD_NUM;
    padExtParams.rightPadding = nextPrPadFactor - nextExtPrPadFactor;
    padExtParams.paddingValue = *((T1 *)& NEG_INF_T1);
  } else {
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
  }

  if (nextDxFactor <= nextNcFactor && nextHxFactor <= nextNcFactor) {
    CopyInputNcRepeat(xLocal, xGmOffset, extParams, padExtParams);
  } else if (nextNcFactor <= nextDxFactor && nextHxFactor <= nextDxFactor) {
    CopyInputDxRepeat(xLocal, xGmOffset, extParams, padExtParams);
  } else if (nextNcFactor <= nextHxFactor && nextDxFactor <= nextHxFactor) {
    CopyInputHxRepeat(xLocal, xGmOffset, extParams, padExtParams);
  }
#if ORIG_DTYPE_X == DT_BF16
  AscendC::PipeBarrier<PIPE_V>();
#endif
}

template <typename T1, typename T2, const uint32_t IS_PAD>
template <typename U>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::Transpose(
                                                                             LocalTensor<U> xTLocal,
                                                                             LocalTensor<U> xLocal,
                                                                             uint64_t rowNum,
                                                                             uint64_t colNum) {
  uint64_t blockAlign = BLOCK_DATA / sizeof(U);
  uint64_t int32DivUSize = sizeof(int32_t) / sizeof(U);
  uint64_t srcAddrList[TRANS_ALIGN];
  uint64_t dstAddrList[TRANS_ALIGN];
  struct TransDataTo5HDParams transDataParams;
  uint64_t colRepeateNum = colNum / blockAlign;
  uint64_t rowRepeateNum = rowNum / TRANS_ALIGN;
  if (colRepeateNum >= rowRepeateNum) {
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    transDataParams.repeatTimes = colRepeateNum;
    if (transDataParams.repeatTimes != 1) {
      transDataParams.srcRepStride = 1;
      transDataParams.dstRepStride = rowNum;
    } else {
      transDataParams.srcRepStride = 0;
      transDataParams.dstRepStride = 0;
    }
    for (uint64_t loop = 0; loop < rowRepeateNum; loop++) {
      uint64_t inputOffsetTmp = inputOffset;
      uint64_t outputOffsetTmp = outputOffset;
      for (uint64_t i = 0; i < TRANS_ALIGN; i+=2) {
        srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
        dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
        inputOffsetTmp += 2 * colNum;
        outputOffsetTmp += int32DivUSize * rowNum;
      }
      inputOffsetTmp = inputOffset + colNum;
      if (int32DivUSize == 2) {
        outputOffsetTmp = outputOffset + rowNum;
      } else {
        outputOffsetTmp = outputOffset + blockAlign;
      }
      for (uint64_t i = 1; i < TRANS_ALIGN; i+=2) {
        srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
        dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
        inputOffsetTmp += 2 * colNum;
        outputOffsetTmp += int32DivUSize * rowNum;
      }
      TransDataTo5HD<U>(dstAddrList, srcAddrList, transDataParams);
      inputOffset += TRANS_ALIGN * colNum;
      outputOffset += TRANS_ALIGN;
    }
  } else {
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    transDataParams.repeatTimes = rowRepeateNum;
    if (transDataParams.repeatTimes != 1) {
      transDataParams.srcRepStride = TRANS_ALIGN * colRepeateNum;
      transDataParams.dstRepStride = TRANS_ALIGN / blockAlign;
    } else {
      transDataParams.srcRepStride = 0;
      transDataParams.dstRepStride = 0;
    }
    for (uint64_t loop = 0; loop < colRepeateNum; loop++) {
      uint64_t inputOffsetTmp = inputOffset;
      uint64_t outputOffsetTmp = outputOffset;
      for (uint64_t i = 0; i < TRANS_ALIGN; i+=2) {
        srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
        dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
        inputOffsetTmp += 2 * colNum;
        outputOffsetTmp += int32DivUSize * rowNum;
      }
      inputOffsetTmp = inputOffset + colNum;
      if (int32DivUSize == 2) {
        outputOffsetTmp = outputOffset + rowNum;
      } else {
        outputOffsetTmp = outputOffset + blockAlign;
      }
      for (uint64_t i = 1; i < TRANS_ALIGN; i+=2) {
        srcAddrList[i] = (uint64_t)(xLocal[inputOffsetTmp].GetPhyAddr());
        dstAddrList[i] = (uint64_t)(xTLocal[outputOffsetTmp].GetPhyAddr());
        inputOffsetTmp += 2 * colNum;
        outputOffsetTmp += int32DivUSize * rowNum;
      }
      TransDataTo5HD<U>(dstAddrList, srcAddrList, transDataParams);
      inputOffset += blockAlign;
      outputOffset += blockAlign * rowNum;
    }
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::PoolMaxAndIndices(
                                                                             LocalTensor<T2> maxLocal,
                                                                             LocalTensor<int32_t> indicesLocal,
                                                                             LocalTensor<T2> inputLocal) {
  int32_t curDValue = 0;
  if constexpr (IS_PAD == ENABLE) {
    if (!curPfPadFactor) {
      curDValue += ((curDyIdx * sd * dyFactor - pf) * hx * wx);
    }
    if (!curPtPadFactor) {
      curDValue += ((curHyIdx * sh * hyFactor - pt) * wx);
    }
    curDValue += ((curWyIdx * sw * wyFactor) - pl);
  } else {
    curDValue = (curDyIdx * sd * dyFactor * hx * wx) +
                (curHyIdx * sh * hyFactor * wx) +
                (curWyIdx * sw * wyFactor);
  }
  uint64_t indicesDOffset = 0;
  uint64_t inputDOffset = 0;
  uint64_t maxDOffset = 0;

  LocalTensor<int32_t> indicesUpdateLocal = indicesUpdateBuffer.Get<int32_t>();

  int32_t kdOffset;
  int32_t kwOffset;
  int32_t khOffset;

  uint64_t mask = curNcFactor;
  uint8_t vecRepeatTimes = curWyFactor;
  uint8_t kernelRepeatStride = 8;
  uint64_t vecLoopNum = 1;
  if (sw * curNcFactorAlign / BLOCK_ALIGN_T2 <= VEC_MAX_REPEAT_STRIDE) {
    kernelRepeatStride =  sw * curNcFactorAlign / BLOCK_ALIGN_T2;
  } else {
    vecRepeatTimes = 1;
    vecLoopNum = curWyFactor;
  }
  uint8_t maxRepeatStride = curNcFactorAlign / BLOCK_ALIGN_T2;
  uint8_t indicesRepeatStride = VEC_INST_CAL_DATA_NUM / BLOCK_ALIGN_INDICES / INT32_DIV_T2_SIZE;

  LocalTensor<uint16_t> gtMaskUb = gtBuffer.Get<uint16_t>();
  LocalTensor<uint16_t> geMaskUb = geBuffer.Get<uint16_t>();
  for (int i = 0; i < curDyFactor; i++) {
    int32_t curDhValue = curDValue;
    int32_t indicesDhOffset = indicesDOffset;
    uint64_t inputDhOffset = inputDOffset;
    uint64_t maxDhOffset = maxDOffset;
    int64_t kd_pf;
    int64_t kd_pb;
    uint64_t kd_ = kd;
    if constexpr (IS_PAD == ENABLE) {
      kd_pf = pf - (curDyIdx * dyFactor + i) * sd;
      kd_pf = kd_pf > 0 ? kd_pf : 0;
      kd_pb = (curDyIdx * dyFactor + i) * sd + kd - dx - pf;
      kd_pb = kd_pb > 0 ? kd_pb : 0;
      kd_ -= (kd_pf + kd_pb);
    }
    for (uint64_t j = 0; j < curHyFactor; j++) {
      int32_t curDhVecValue = curDhValue;
      int32_t indicesDhVecOffset = indicesDhOffset;
      uint64_t inputDhVecOffset = inputDhOffset;
      uint64_t maxDhVecOffset = maxDhOffset;
      int64_t kh_pt;
      int64_t kh_pd;
      uint64_t kh_ = kh;
      uint64_t setValueNum;
      if constexpr (IS_PAD == ENABLE) {
        kh_pt = pt - (curHyIdx * hyFactor + j) * sh;
        kh_pt = kh_pt > 0 ? kh_pt : 0;
        kh_pd = (curHyIdx * hyFactor + j) * sh + kh - hx - pt;
        kh_pd = kh_pd > 0 ? kh_pd : 0;
        kh_ -= (kh_pt + kh_pd);
        setValueNum = CeilDiv(curPlPadFactor, sw);
      }
      for (uint64_t v = 0; v < vecLoopNum; v++) {
        // init indicesLocal
        Duplicate(indicesLocal[indicesDhVecOffset], curDhVecValue, mask, vecRepeatTimes * INT32_DIV_T2_SIZE,
                  1, indicesRepeatStride);
        AscendC::PipeBarrier<PIPE_V>();
        Add(indicesLocal[indicesDhVecOffset], indicesLocal[indicesDhVecOffset], indicesTemplateLocal, mask,
            vecRepeatTimes * INT32_DIV_T2_SIZE,
            {1, 1, 1, indicesRepeatStride, indicesRepeatStride, indicesRepeatStride});
        // init maxLocal indicesLocal
        DataCopyParams InitMaxCopyParams;
        InitMaxCopyParams.blockCount = vecRepeatTimes;
        InitMaxCopyParams.blockLen = curNcFactorAlign / BLOCK_ALIGN_T2;
        InitMaxCopyParams.srcStride = curNcFactorAlign * (sw - 1) / BLOCK_ALIGN_T2;
        DataCopy(maxLocal[maxDhVecOffset], inputLocal[inputDhVecOffset], InitMaxCopyParams);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint64_t k = 1; k < kd_ * kh_ * kw; k++) {
          kdOffset = k / (kh_ * kw);
          khOffset = k % (kh_ * kw) / kw;
          kwOffset = k % kw;
          uint64_t inputKenelOffset = (inputDhVecOffset) +
                                      (kdOffset * curHxFactor * curWxFactorAlign * curNcFactorAlign) +
                                      (khOffset * curWxFactorAlign * curNcFactorAlign) +
                                      (kwOffset * curNcFactorAlign);

          int32_t indicesKernelOffset = (curDhVecValue) +
                                        (kdOffset * hx * wx) +
                                        (khOffset * wx) +
                                        (kwOffset);
          Duplicate(indicesUpdateLocal, indicesKernelOffset, mask, vecRepeatTimes * INT32_DIV_T2_SIZE,
                    1, indicesRepeatStride);
          AscendC::PipeBarrier<PIPE_V>();
          Add(indicesUpdateLocal, indicesUpdateLocal, indicesTemplateLocal,
              mask, vecRepeatTimes * INT32_DIV_T2_SIZE,
              {1, 1, 1, indicesRepeatStride, indicesRepeatStride, indicesRepeatStride});
          // cmp_gt
          Compare(gtMaskUb.template ReinterpretCast<uint8_t>(), inputLocal[inputKenelOffset], maxLocal[maxDhVecOffset],
                  CMPMODE::GT, mask, vecRepeatTimes, {1, 1, 1, 1, kernelRepeatStride, maxRepeatStride});
          if constexpr (IS_PAD == ENABLE) {
            /*
              For avoiding to select -inf, we must set gtmask when for loop from the padding -inf to normal -inf. For example:
              the kernel is 4 and stride is 1, the padding is 2, when the input data is all -inf.
              The input data is:
              [-inf -inf -inf -inf -inf -inf -inf -inf]
              After padding:
              |-padding-|---------------normal------------------]
              [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]
               ------------------- ← first kernel              | gtmask set secondly
                    ------------------- ← second kernel        | gtmask set firstly
                         ------------------- ← third kernel    | gtmask don't need to be set
              Fristly, we traversal the second point in kernel(to compare with first point). And the gtmask for second kernel should
              be set. Then we traversal the third point in kernel(to compare with second point). And the gtmask for first kernel should
              be set.
              setValueNum is the kernel num need to set gt mask.
            */
            if (curPlPadFactor && setValueNum && (kwOffset + (setValueNum - 1) * sw - curPlPadFactor == 0)) {
              AscendC::PipeBarrier<PIPE_V>();
              uint64_t setValueNumLoop = (setValueNum - 1) / VEC_INST_CAL_MASK_NUM;
              uint64_t setValueNumTail = (setValueNum - 1) % VEC_INST_CAL_MASK_NUM;
              uint64_t orMask[2];
  #if ORIG_DTYPE_X == DT_FLOAT16
              orMask[0] = (uint64_t)0xFF << (setValueNumTail * VEC_INST_CAL_MASK_NUM);
  #else
              orMask[0] = (uint64_t)0xF << (setValueNumTail * VEC_INST_CAL_MASK_NUM);
  #endif
              orMask[1] = 0;
              Or(gtMaskUb[setValueNumLoop * BLOCK_ALIGN_MASK], gtMaskUb[setValueNumLoop * BLOCK_ALIGN_MASK], orMaskLocal,
                orMask, 1, {1, 1, 1, 8, 8, 8});
                setValueNum--;
            }
          }
          // cmp_ge
          Compare(geMaskUb.template ReinterpretCast<uint8_t>(), inputLocal[inputKenelOffset], inputLocal[inputKenelOffset],
                  CMPMODE::EQ, mask, vecRepeatTimes, {1, 1, 1, 1, kernelRepeatStride, kernelRepeatStride});
          AscendC::PipeBarrier<PIPE_V>();
          // not
          Not(geMaskUb, geMaskUb, curWyFactorAlign * VEC_INST_CAL_MASK_NUM);
          AscendC::PipeBarrier<PIPE_V>();
          // or
          Or(gtMaskUb, gtMaskUb, geMaskUb, curWyFactorAlign * VEC_INST_CAL_MASK_NUM);
          AscendC::PipeBarrier<PIPE_V>();
          // update indices
          Select(indicesLocal[indicesDhVecOffset].ReinterpretCast<float>(), gtMaskUb,
                indicesUpdateLocal.ReinterpretCast<float>(),
                indicesLocal[indicesDhVecOffset].ReinterpretCast<float>(),
                SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, vecRepeatTimes * INT32_DIV_T2_SIZE,
                {1, 1, 1, indicesRepeatStride, indicesRepeatStride, indicesRepeatStride});
          Max(maxLocal[maxDhVecOffset], maxLocal[maxDhVecOffset], inputLocal[inputKenelOffset], mask, vecRepeatTimes,
              {1, 1, 1, maxRepeatStride, maxRepeatStride, kernelRepeatStride});
        }
        curDhVecValue += sw;
        inputDhVecOffset += sw * curNcFactorAlign;
        indicesDhVecOffset += VEC_INST_CAL_DATA_NUM;
        maxDhVecOffset += curNcFactorAlign;
      }
      if constexpr (IS_PAD == ENABLE) {
        if ((int64_t)sh - kh_pt > 0) {
          curDhValue += ((int64_t)sh - kh_pt) * wx;
          inputDhOffset += ((int64_t)sh - kh_pt) * curWxFactorAlign * curNcFactorAlign;
        }
      } else {
        curDhValue += sh * wx;
        inputDhOffset += sh * curWxFactorAlign * curNcFactorAlign;
      }
      indicesDhOffset += curWyFactorAlign * VEC_INST_CAL_DATA_NUM;
      maxDhOffset += curWyFactorAlign * curNcFactorAlign;
    }
    if constexpr (IS_PAD == ENABLE) {
      if ((int64_t)sd - kd_pf > 0) {
        curDValue += ((int64_t)sd - kd_pf) * hx * wx;
        inputDOffset += ((int64_t)sd - kd_pf) * curHxFactor * curWxFactorAlign * curNcFactorAlign;
      }
    } else {
      curDValue += sd * hx * wx;
      inputDOffset += sd * curHxFactor * curWxFactorAlign * curNcFactorAlign;
    }
    indicesDOffset += curHyFactor * curWyFactorAlign * VEC_INST_CAL_DATA_NUM;
    maxDOffset += curHyFactor * curWyFactorAlign * curNcFactorAlign;
  }
}
#if ORIG_DTYPE_X == DT_BF16
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CastF32ToBF16(
                                                                             LocalTensor<T1> origMaxOutLocal,
                                                                             LocalTensor<T2> maxOutLocal,
                                                                             uint8_t castRepeateTimes,
                                                                             uint8_t castRepeateStride) {
  uint64_t castLoopNum = curWyFactor / VEC_INST_CAL_DATA_NUM;
  uint64_t castTailNum = curWyFactor % VEC_INST_CAL_DATA_NUM;
  uint64_t maxOutLocalCastOffset = 0;
  UnaryRepeatParams castParams = {1, 1, castRepeateStride, castRepeateStride};
  for (uint64_t i = 0; i < castLoopNum; i++) {
    Cast(origMaxOutLocal[maxOutLocalCastOffset], maxOutLocal[maxOutLocalCastOffset],
          RoundMode::CAST_ROUND, VEC_INST_CAL_DATA_NUM, castRepeateTimes, castParams);
    maxOutLocalCastOffset += VEC_INST_CAL_DATA_NUM;
  }
  if (castTailNum != 0) {
    Cast(origMaxOutLocal[maxOutLocalCastOffset], maxOutLocal[maxOutLocalCastOffset],
          RoundMode::CAST_ROUND, castTailNum, castRepeateTimes, castParams);
  }
  event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
  WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}
#endif

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyMaxOutNcRepeat(
                                                                             LocalTensor<T2> maxOutLocal,
                                                                             uint64_t maxGmOffset,
                                                                             DataCopyExtParams& extParams) {
  uint64_t maxOutLocalOffset = 0;
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t vecStride = curDyFactor * curHyFactor * curWyFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = (vecStride / BLOCK_ALIGN_T2) -
                          CeilDiv(curWyFactor, BLOCK_ALIGN_T1);
  uint64_t mteDstStride = (ncDimStride - curWyFactor) * BYTE_T1;
  if (mteDstStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = curNcFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = curNcFactor;
  }
  for (uint64_t i = 0; i < curDyFactor; i++) {
    uint64_t maxGmOffsetTmp = maxGmOffset;
    uint64_t maxOutLocalOffsetTmp = maxOutLocalOffset;
    for (uint64_t j = 0; j < curHyFactor; j++) {
      uint64_t maxGmMteOffsetTmp = maxGmOffsetTmp;
      uint64_t maxOutLocalMteOffsetTmp = maxOutLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
#if ORIG_DTYPE_X == DT_BF16
        LocalTensor<T1> origMaxOutLocal = maxOutLocal[maxOutLocalMteOffsetTmp].template ReinterpretCast<T1>();
        CastF32ToBF16(origMaxOutLocal, maxOutLocal[maxOutLocalMteOffsetTmp],
                      extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
        DataCopyPad(maxGm[maxGmMteOffsetTmp], origMaxOutLocal, extParams);
#else
        DataCopyPad(maxGm[maxGmMteOffsetTmp], maxOutLocal[maxOutLocalMteOffsetTmp], extParams);
#endif
        maxGmMteOffsetTmp += ncDimStride;
        maxOutLocalMteOffsetTmp += vecStride;
      }
      maxGmOffsetTmp += wy;
      maxOutLocalOffsetTmp += curWyFactorAlign;
    }
    maxGmOffset += hy * wy;
    maxOutLocalOffset += curHyFactor * curWyFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyMaxOutDyRepeat(
                                                                             LocalTensor<T2> maxOutLocal,
                                                                             uint64_t maxGmOffset,
                                                                             DataCopyExtParams& extParams) {
  uint64_t maxOutLocalOffset = 0;
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t vecStride = curHyFactor * curWyFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = (vecStride / BLOCK_ALIGN_T2) -
                          CeilDiv(curWyFactor, BLOCK_ALIGN_T1);
  uint64_t mteDstStride = (hy * wy - curWyFactor) * BYTE_T1;
  if (mteDstStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = curDyFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = curDyFactor;
  }
  for (uint64_t i = 0; i < curNcFactor; i++) {
    uint64_t maxGmOffsetTmp = maxGmOffset;
    uint64_t maxOutLocalOffsetTmp = maxOutLocalOffset;
    for (uint64_t j = 0; j < curHyFactor; j++) {
      uint64_t maxGmMteOffsetTmp = maxGmOffsetTmp;
      uint64_t maxOutLocalMteOffsetTmp = maxOutLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
#if ORIG_DTYPE_X == DT_BF16
        LocalTensor<T1> origMaxOutLocal = maxOutLocal[maxOutLocalMteOffsetTmp].template ReinterpretCast<T1>();
        CastF32ToBF16(origMaxOutLocal, maxOutLocal[maxOutLocalMteOffsetTmp],
                      extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
        DataCopyPad(maxGm[maxGmMteOffsetTmp], origMaxOutLocal, extParams);
#else
        DataCopyPad(maxGm[maxGmMteOffsetTmp], maxOutLocal[maxOutLocalMteOffsetTmp], extParams);
#endif
        maxGmMteOffsetTmp += hy * wy;
        maxOutLocalMteOffsetTmp += vecStride;
      }
      maxGmOffsetTmp += wy;
      maxOutLocalOffsetTmp += curWyFactorAlign;
    }
    maxGmOffset += ncDimStride;
    maxOutLocalOffset += curDyFactor * curHyFactor * curWyFactorAlign;
  }
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyMaxOutHyRepeat(
                                                                             LocalTensor<T2> maxOutLocal,
                                                                             uint64_t maxGmOffset,
                                                                             DataCopyExtParams& extParams) {
  uint64_t maxOutLocalOffset = 0;
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t vecStride = curWyFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = (vecStride / BLOCK_ALIGN_T2) -
                          CeilDiv(curWyFactor, BLOCK_ALIGN_T1);
  uint64_t mteDstStride = (wy - curWyFactor) * BYTE_T1;
  if (mteDstStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = curHyFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = curHyFactor;
  }
  for (uint64_t i = 0; i < curNcFactor; i++) {
    uint64_t maxGmOffsetTmp = maxGmOffset;
    uint64_t maxOutLocalOffsetTmp = maxOutLocalOffset;
    for (uint64_t j = 0; j < curDyFactor; j++) {
      uint64_t maxGmMteOffsetTmp = maxGmOffsetTmp;
      uint64_t maxOutLocalMteOffsetTmp = maxOutLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
#if ORIG_DTYPE_X == DT_BF16
        LocalTensor<T1> origMaxOutLocal = maxOutLocal[maxOutLocalMteOffsetTmp].template ReinterpretCast<T1>();
        CastF32ToBF16(origMaxOutLocal, maxOutLocal[maxOutLocalMteOffsetTmp],
                      extParams.blockCount, vecStride / BLOCK_ALIGN_T2);
        DataCopyPad(maxGm[maxGmMteOffsetTmp], origMaxOutLocal, extParams);
#else
        DataCopyPad(maxGm[maxGmMteOffsetTmp], maxOutLocal[maxOutLocalMteOffsetTmp], extParams);
#endif
        maxGmMteOffsetTmp += wy;
        maxOutLocalMteOffsetTmp += vecStride;
      }
      maxGmOffsetTmp += hy * wy;
      maxOutLocalOffsetTmp += curHyFactor * curWyFactorAlign;
    }
    maxGmOffset += ncDimStride;
    maxOutLocalOffset += curDyFactor * curHyFactor * curWyFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyMaxOut(LocalTensor<T2> maxOutLocal) {
#if ORIG_DTYPE_X == DT_BF16
  AscendC::PipeBarrier<PIPE_V>();
#endif
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t ncOffset = curNcIdx * ncFactor * ncDimStride;
  uint64_t dyOffset = curDyIdx * dyFactor * hy * wy;
  uint64_t hyOffset = curHyIdx * hyFactor * wy;
  uint64_t wyOffset = curWyIdx * wyFactor;

  uint64_t maxGmOffset = ncOffset + dyOffset + hyOffset + wyOffset;

  DataCopyExtParams extParams;
  extParams.blockLen = curWyFactor * BYTE_T1;

  if (curDyFactor <= curNcFactor && curHyFactor <= curNcFactor) {
    CopyMaxOutNcRepeat(maxOutLocal, maxGmOffset, extParams);
  } else if (curNcFactor <= curDyFactor && curHyFactor <= curDyFactor) {
    CopyMaxOutDyRepeat(maxOutLocal, maxGmOffset, extParams);
  } else if (curNcFactor <= curHyFactor && curDyFactor <= curHyFactor) {
    CopyMaxOutHyRepeat(maxOutLocal, maxGmOffset, extParams);
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyIndicesOutNcRepeat(
                                                                             LocalTensor<int32_t> indicesTransLocal,
                                                                             uint64_t indicesGmOffset,
                                                                             DataCopyExtParams& extParams) {
  uint64_t indicesTransLocalOffset = 0;
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t vecStride = curDyFactor * curHyFactor * curWyFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = (vecStride - curWyFactor) / BLOCK_ALIGN_INDICES;
  uint64_t mteDstStride = (ncDimStride - curWyFactor) * BYTE_INDICES;
  if (mteDstStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = curNcFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = curNcFactor;
  }
  for (uint64_t i = 0; i < curDyFactor; i++) {
    uint64_t indicesGmOffsetTmp = indicesGmOffset;
    uint64_t indicesTransLocalOffsetTmp = indicesTransLocalOffset;
    for (uint64_t j = 0; j < curHyFactor; j++) {
      uint64_t indicesGmMteOffsetTmp = indicesGmOffsetTmp;
      uint64_t indicesTransLocalMteOffsetTmp = indicesTransLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
        DataCopyPad(indicesGm[indicesGmMteOffsetTmp], indicesTransLocal[indicesTransLocalMteOffsetTmp], extParams);
        indicesGmMteOffsetTmp += ncDimStride;
        indicesTransLocalMteOffsetTmp += vecStride;
      }
      indicesGmOffsetTmp += wy;
      indicesTransLocalOffsetTmp += curWyFactorAlign;
    }
    indicesGmOffset += hy * wy;
    indicesTransLocalOffset += curHyFactor * curWyFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyIndicesOutDyRepeat(
                                                                             LocalTensor<int32_t> indicesTransLocal,
                                                                             uint64_t indicesGmOffset,
                                                                             DataCopyExtParams& extParams) {
  uint64_t indicesTransLocalOffset = 0;
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t vecStride = curHyFactor * curWyFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = (vecStride - curWyFactor) / BLOCK_ALIGN_INDICES;
  uint64_t mteDstStride = (hy * wy - curWyFactor) * BYTE_INDICES;
  if (mteDstStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = curDyFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = curDyFactor;
  }
  for (uint64_t i = 0; i < curNcFactor; i++) {
    uint64_t indicesGmOffsetTmp = indicesGmOffset;
    uint64_t indicesTransLocalOffsetTmp = indicesTransLocalOffset;
    for (uint64_t j = 0; j < curHyFactor; j++) {
      uint64_t indicesGmMteOffsetTmp = indicesGmOffsetTmp;
      uint64_t indicesTransLocalMteOffsetTmp = indicesTransLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
        DataCopyPad(indicesGm[indicesGmMteOffsetTmp], indicesTransLocal[indicesTransLocalMteOffsetTmp], extParams);
        indicesGmMteOffsetTmp += hy * wy;
        indicesTransLocalMteOffsetTmp += vecStride;
      }
      indicesGmOffsetTmp += wy;
      indicesTransLocalOffsetTmp += curWyFactorAlign;
    }
    indicesGmOffset += ncDimStride;
    indicesTransLocalOffset += curDyFactor * curHyFactor * curWyFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyIndicesOutHyRepeat(
                                                                             LocalTensor<int32_t> indicesTransLocal,
                                                                             uint64_t indicesGmOffset,
                                                                             DataCopyExtParams& extParams) {
  uint64_t indicesTransLocalOffset = 0;
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t vecStride = curWyFactorAlign;
  uint64_t mteLoopNum = 1;
  uint64_t mteSrcStride = (vecStride - curWyFactor) / BLOCK_ALIGN_INDICES;
  uint64_t mteDstStride = (wy - curWyFactor) * BYTE_INDICES;
  if (mteDstStride <= MTE_MAX_GM_STRIDE) {
      extParams.srcStride = mteSrcStride;
      extParams.dstStride = mteDstStride;
      extParams.blockCount = curHyFactor;
  } else {
    extParams.blockCount = 1;
    mteLoopNum = curHyFactor;
  }
  for (uint64_t i = 0; i < curNcFactor; i++) {
    uint64_t indicesGmOffsetTmp = indicesGmOffset;
    uint64_t indicesTransLocalOffsetTmp = indicesTransLocalOffset;
    for (uint64_t j = 0; j < curDyFactor; j++) {
      uint64_t indicesGmMteOffsetTmp = indicesGmOffsetTmp;
      uint64_t indicesTransLocalMteOffsetTmp = indicesTransLocalOffsetTmp;
      for (uint64_t m = 0; m < mteLoopNum; m++) {
        DataCopyPad(indicesGm[indicesGmMteOffsetTmp], indicesTransLocal[indicesTransLocalMteOffsetTmp], extParams);
        indicesGmMteOffsetTmp += hy * wy;
        indicesTransLocalMteOffsetTmp += vecStride;
      }
      indicesGmOffsetTmp += hy * wy;
        indicesTransLocalOffsetTmp += curHyFactor * curWyFactorAlign;
    }
    indicesGmOffset += ncDimStride;
    indicesTransLocalOffset += curDyFactor * curHyFactor * curWyFactorAlign;
  }
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2NoExpandIndices<T1, T2, IS_PAD>::CopyIndicesOut(
                                                                             LocalTensor<int32_t> indicesTransLocal) {
  uint64_t ncDimStride = dy * hy * wy;
  uint64_t ncOffset = curNcIdx * VEC_INST_CAL_DATA_NUM * ncDimStride;
  uint64_t dyOffset = curDyIdx * dyFactor * hy * wy;
  uint64_t hyOffset = curHyIdx * hyFactor * wy;
  uint64_t wyOffset = curWyIdx * wyFactor;

  uint64_t indicesGmOffset = ncOffset + dyOffset + hyOffset + wyOffset;

  DataCopyExtParams extParams;
  extParams.blockLen = curWyFactor * BYTE_INDICES;

  if (curDyFactor <= curNcFactor && curHyFactor <= curNcFactor) {
    CopyIndicesOutNcRepeat(indicesTransLocal, indicesGmOffset, extParams);
  } else if (curNcFactor <= curDyFactor && curHyFactor <= curDyFactor) {
    CopyIndicesOutDyRepeat(indicesTransLocal, indicesGmOffset, extParams);
  } else if (curNcFactor <= curHyFactor && curDyFactor <= curHyFactor) {
    CopyIndicesOutHyRepeat(indicesTransLocal, indicesGmOffset, extParams);
  }
}

#endif  // MAX_POOL3D_WITH_ARGMAX_V2_NO_EXPAND_INDICES_H_
