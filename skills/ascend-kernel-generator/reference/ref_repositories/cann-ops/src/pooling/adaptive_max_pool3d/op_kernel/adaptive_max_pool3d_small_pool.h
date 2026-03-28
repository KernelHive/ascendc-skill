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
 * \file adaptive_max_pool3d_small_pool.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_SAMLL_POOL_H_
#define ADAPTIVE_MAX_POOL3D_SAMLL_POOL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

template <typename T>
class AdaptiveMaxPool3dSmallPool {
 public:
  __aicore__ inline AdaptiveMaxPool3dSmallPool(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe,
                              const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling);
  __aicore__ inline void InitTiling(const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void InitReset();
  __aicore__ inline void CalReset(const uint8_t diFactor, const uint8_t hiFactor);
  __aicore__ inline void CopyIn(int64_t curIdx);
  __aicore__ inline void CopyInput(int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor,
                                   const uint8_t wiFactor, int64_t xGmOffset);
  __aicore__ inline void TransInput(int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor,
                                    const uint8_t wiFactor);
  __aicore__ inline void MaxPoolW(const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor,
                                  int64_t curWoIdx, int64_t curWoFactor);
  __aicore__ inline void MaxPoolH(const uint8_t diFactor, const uint8_t hiFactor, const uint8_t curWoFactor,
                                  int64_t curHoIdx, int64_t curHoFactor);
  __aicore__ inline void MaxPoolD(const uint8_t diFactor, const uint8_t hiFactor, const uint8_t curWoFactor,
                                  int64_t curDoIdx, int64_t curDoFactor);
  __aicore__ inline void TransOutAndIdx(int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor,
                                        int64_t UbIdxOffset);
  __aicore__ inline void CopyOutAndIdx(int64_t curNcFactor, int64_t curDoFactor, int64_t curHoFactor,
                                       int64_t curWoFactor, int64_t yGmOffset);
  __aicore__ inline void OutTranspose(LocalTensor<float> xLocalTrans, LocalTensor<float> xLocal, int32_t rowNum,
                                      int32_t colNum);

  TQue<QuePosition::VECIN, 1> inputQue;
  TQue<QuePosition::VECOUT, 1> maxQue;
  TQue<QuePosition::VECOUT, 1> indexQue;
  TBuf<> inputTransBuffer;
  TBuf<> cmpMaskBuffer;
  TBuf<> cmpNanMaskBuffer;
  TBuf<> resetIndexBuf;
  TBuf<> nextCmpBuffer;
  TBuf<> mulWBuffer;
  TBuf<> mulWIdxBuffer;

  GlobalTensor<T> xGm, maxGm;
  GlobalTensor<int32_t> indicesGm;

  uint32_t cBlockIdx = 0;

  int64_t N = 1;
  int64_t C = 1;
  int64_t Di = 1;
  int64_t Hi = 1;
  int64_t Wi = 1;
  int64_t Do = 1;
  int64_t Ho = 1;
  int64_t Wo = 1;
  int64_t DiHiWi = 1;
  int64_t HiWi = 1;
  const int32_t VL_NUM = 64;  // Vector calculate length / float size

  // 多核切分的整尾块
  int64_t ncFactor = 0;
  int64_t doFactor = 0;
  int64_t hoFactor = 0;
  int64_t woFactor = 0;
  int64_t ncTail = 0;
  int64_t doTail = 0;
  int64_t hoTail = 0;
  int64_t woTail = 0;

  // 多核切分的数量
  int64_t ncOuter = 0;
  int64_t doOuter = 0;
  int64_t hoOuter = 0;
  int64_t woOuter = 0;

  int64_t totalIdx = 0;     // 总UB计算块
  int64_t blockFactor = 0;  // 每个核最多计算的UB块
  int64_t useCoreNum = 0;   // 使用核数
  int64_t blockTail = 0;    // 多核尾块

  int64_t beginIdx = 0;  // 当前核计算块起始id
  int64_t endIdx = 0;    // 当前核计算块终止id

  SELMODE selMode = SELMODE::VSEL_TENSOR_TENSOR_MODE;
};

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::InitTiling(
    const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling) {
  useCoreNum = tiling->useCoreNum;
  N = tiling->N;
  C = tiling->C;
  Di = tiling->Di;
  Hi = tiling->Hi;
  Wi = tiling->Wi;
  Do = tiling->Do;
  Ho = tiling->Ho;
  Wo = tiling->Wo;
  totalIdx = tiling->totalIdx;
  blockFactor = tiling->blockFactor;
  blockTail = tiling->blockTail;
  ncFactor = tiling->ncFactor;
  woFactor = tiling->woFactor;
  hoFactor = tiling->hoFactor;
  doFactor = tiling->doFactor;
  doOuter = tiling->doOuter;
  doTail = tiling->doTail;
  hoOuter = tiling->hoOuter;
  hoTail = tiling->hoTail;
  woOuter = tiling->woOuter;
  woTail = tiling->woTail;
  ncOuter = tiling->ncOuter;
  ncTail = tiling->ncTail;
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe,
    const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling) {
  InitTiling(tiling);

  cBlockIdx = GetBlockIdx();
  if (cBlockIdx >= useCoreNum) {
    return;
  }

  DiHiWi = Di * Hi * Wi;
  HiWi = Hi * Wi;
  int64_t calBlockNum = blockFactor;
  if (cBlockIdx == useCoreNum - 1) {
    calBlockNum = blockTail;
  }
  beginIdx = cBlockIdx * blockFactor;
  endIdx = cBlockIdx * blockFactor + calBlockNum;

  xGm.SetGlobalBuffer((__gm__ T*)x);
  maxGm.SetGlobalBuffer((__gm__ T*)y);
  indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);

  // 初始化que
  pipe->InitBuffer(inputQue, 1, 32 * 1024);  // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(T) 有问题？
  pipe->InitBuffer(maxQue, 1, 8 * 1024);     // VL_NUM*doFactor*hoFactor*woFactorAlign*sizeof(T)
  pipe->InitBuffer(indexQue, 1, 8 * 1024);   // VL_NUM*doFactor*hoFactor*woFactorAlign*sizeof(int32)

  // 初始化Tbuf
  // VL_NUM/8 * ((kW>1)diFactor*hiFactor | (kH>1)woFactorAlign*diFactor | (kD>1)hoFactor * woFactorAlign)
  pipe->InitBuffer(cmpMaskBuffer, 512);
  pipe->InitBuffer(cmpNanMaskBuffer, 512);
  pipe->InitBuffer(inputTransBuffer, 32 * 1024);  // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(float)
  pipe->InitBuffer(resetIndexBuf, 4 * 1024);      // VL_NUM*diFactor*hiFactor*sizeof(int32)
  pipe->InitBuffer(nextCmpBuffer, 4 * 1024);      // VL_NUM*diFactor*hiFactor*sizeof(int32)
  pipe->InitBuffer(mulWBuffer, 64 * 1024);        // VL_NUM*diFactor*hiFactor*wiFactor16Align*sizeof(float)
  pipe->InitBuffer(mulWIdxBuffer, 32 * 1024);     // VL_NUM*diFactor*hiFactor*woFactorAlign*sizeof(int32)
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::OutTranspose(LocalTensor<float> xLocalTrans,
                                                                   LocalTensor<float> xLocal, int32_t rowNum,
                                                                   int32_t colNum) {
  LocalTensor<float> dstList[16];
  LocalTensor<float> srcList[16];

  event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

  TransDataTo5HDParams transDataParams;
  transDataParams.dstHighHalf = false;
  transDataParams.srcHighHalf = false;
  if (colNum == 8) {
    transDataParams.repeatTimes = rowNum / 16;
    transDataParams.dstRepStride = 2;
    transDataParams.srcRepStride = 16;

    for (int32_t i = 0; i < 16; i++) {
      srcList[i] = xLocal[i * 8];
    }

    for (int32_t i = 0; i < 8; i++) {
      dstList[i * 2] = xLocalTrans[i * rowNum];
      dstList[i * 2 + 1] = xLocalTrans[i * rowNum + 8];
    }

    SetFlag<HardEvent::S_V>(eventSV);
    WaitFlag<HardEvent::S_V>(eventSV);
    TransDataTo5HD<float>(dstList, srcList, transDataParams);
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
  } else {
    transDataParams.repeatTimes = colNum / 8;
    transDataParams.dstRepStride = rowNum;
    transDataParams.srcRepStride = 1;
    for (int32_t j = 0; j < rowNum / 16; j++) {
      for (int32_t i = 0; i < 16; i++) {
        srcList[i] = xLocal[i * colNum + j * 16 * colNum];
      }

      for (int32_t i = 0; i < 8; i++) {
        dstList[i * 2] = xLocalTrans[i * rowNum + j * 16];
        dstList[i * 2 + 1] = xLocalTrans[i * rowNum + 8 + j * 16];
      }

      SetFlag<HardEvent::S_V>(eventSV);
      WaitFlag<HardEvent::S_V>(eventSV);
      TransDataTo5HD<float>(dstList, srcList, transDataParams);
      SetFlag<HardEvent::V_S>(eventVS);
      WaitFlag<HardEvent::V_S>(eventVS);
    }
  }
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::InitReset() {
  int32_t inputVal(0);
  LocalTensor<int32_t> resetIdx = resetIndexBuf.Get<int32_t>();
  Duplicate<int32_t>(resetIdx, inputVal, VL_NUM);
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CalReset(const uint8_t diFactor, const uint8_t hiFactor) {
  LocalTensor<int32_t> resetIdx = resetIndexBuf.Get<int32_t>();

  for (int i = 1; i < hiFactor; i++) {
    Adds(resetIdx[VL_NUM * i], resetIdx, (int32_t)(Wi * i), VL_NUM);
  }
  AscendC::PipeBarrier<PIPE_V>();
  for (int i = 1; i < diFactor; i++) {
    Adds(resetIdx[VL_NUM * hiFactor * i], resetIdx, (int32_t)(Wi * Hi * i), VL_NUM * hiFactor);
  }
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CopyIn(int64_t curIdx) {
  auto curNcIdx = curIdx / (doOuter * hoOuter * woOuter);
  auto curNcFactor = curNcIdx == (ncOuter - 1) ? ncTail : ncFactor;
  auto tmpIdx = curIdx % (doOuter * hoOuter * woOuter);
  auto curDoIdx = tmpIdx / (hoOuter * woOuter);
  auto curDoFactor = curDoIdx == (doOuter - 1) ? doTail : doFactor;
  tmpIdx = tmpIdx % (hoOuter * woOuter);
  auto curHoIdx = tmpIdx / woOuter;
  auto curHoFactor = curHoIdx == (hoOuter - 1) ? hoTail : hoFactor;
  auto curWoIdx = tmpIdx % woOuter;
  auto curWoFactor = curWoIdx == (woOuter - 1) ? woTail : woFactor;

  int32_t kerDStartIdxTotal = ((curDoIdx * doFactor) * Di) / Do;
  int32_t kerHStartIdxTotal = ((curHoIdx * hoFactor) * Hi) / Ho;
  int32_t kerWStartIdxTotal = ((curWoIdx * woFactor) * Wi) / Wo;
  int32_t kerDEndIdxTotal = Ceil((curDoFactor + curDoIdx * doFactor) * Di, Do);
  int32_t kerHEndIdxTotal = Ceil((curHoFactor + curHoIdx * hoFactor) * Hi, Ho);
  int32_t kerWEndIdxTotal = Ceil((curWoFactor + curWoIdx * woFactor) * Wi, Wo);

  const uint8_t diFactor = kerDEndIdxTotal - kerDStartIdxTotal;
  const uint8_t hiFactor = kerHEndIdxTotal - kerHStartIdxTotal;
  const uint8_t wiFactor = kerWEndIdxTotal - kerWStartIdxTotal;

  auto xGmOffset = curNcIdx * ncFactor * DiHiWi + kerDStartIdxTotal * HiWi + kerHStartIdxTotal * Wi + kerWStartIdxTotal;

  CopyInput(curNcFactor, diFactor, hiFactor, wiFactor, xGmOffset);
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CopyInput(int64_t curNcFactor, const uint8_t diFactor,
                                                                const uint8_t hiFactor, const uint8_t wiFactor,
                                                                int64_t xGmOffset) {
  LocalTensor<T> xLocal = inputQue.AllocTensor<T>();

  const uint8_t wiFactor16Align = Ceil(wiFactor, 32 / sizeof(T)) * 32 / sizeof(T);

  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  DataCopyExtParams paramsIn;
  paramsIn.blockCount = hiFactor;
  paramsIn.blockLen = wiFactor * sizeof(T);
  paramsIn.srcStride = (Wi - wiFactor) * sizeof(T);
  paramsIn.dstStride = 0;
  for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
    for (int64_t dCopyi = 0; dCopyi < diFactor; dCopyi++) {
      auto srcAddr = xGmOffset + ncCopyi * DiHiWi + dCopyi * HiWi;
      auto dstAddr = (ncCopyi * diFactor + dCopyi) * hiFactor * wiFactor16Align;
      DataCopyPad(xLocal[dstAddr], xGm[srcAddr], paramsIn, padParams);
    }
  }
  inputQue.EnQue(xLocal);
}

/*
 * 功能：input类型转换 <T> -> <fp32>类型, 并转置，把[VL, D, H, W] 转为[D, H, W, VL]
 */
template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::TransInput(int64_t curNcFactor, const uint8_t diFactor,
                                                                 const uint8_t hiFactor, const uint8_t wiFactor) {
  const uint8_t wiFactor16Align = Ceil(wiFactor, 32 / sizeof(T)) * 32 / sizeof(T);
  const uint8_t wiFactorAlign = Ceil(wiFactor, 8) * 8;
  LocalTensor<T> xLocal = inputQue.DeQue<T>();
  LocalTensor<float> xLocalTransVL = inputTransBuffer.Get<float>();
  if constexpr (IsSameType<T, float>::value) {
    OutTranspose(xLocalTransVL, xLocal, VL_NUM, diFactor * hiFactor * wiFactorAlign);
  } else {
    LocalTensor<float> xLocalCast = mulWBuffer.Get<float>();
    UnaryRepeatParams repeatCastParams{(uint16_t)(wiFactorAlign / 8), (uint16_t)(wiFactor16Align / 8),
                                       (uint8_t)(wiFactorAlign / 8 * Ceil(diFactor * hiFactor, 2)),
                                       (uint8_t)(wiFactor16Align / 8 * Ceil(diFactor * hiFactor, 2))};

    Cast(xLocalCast, xLocal, RoundMode::CAST_NONE, wiFactor16Align * curNcFactor * diFactor * hiFactor);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(xLocalCast, xLocalCast, float(0.0), uint8_t(wiFactorAlign * Ceil(diFactor * hiFactor, 2)), curNcFactor * 2,
         repeatCastParams);

    AscendC::PipeBarrier<PIPE_V>();
    OutTranspose(xLocalTransVL, xLocalCast, VL_NUM, diFactor * hiFactor * wiFactorAlign);
  }
  AscendC::PipeBarrier<PIPE_V>();
  inputQue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::MaxPoolW(const uint8_t diFactor, const uint8_t hiFactor,
                                                               const uint8_t wiFactor, int64_t curWoIdx,
                                                               int64_t curWoFactor) {
  LocalTensor<int32_t> resetIdx = resetIndexBuf.Get<int32_t>();

  LocalTensor<float> xLocalTransVL = inputTransBuffer.Get<float>();

  const uint8_t wiFactorAlign = Ceil(wiFactor, 8) * 8;

  int32_t kerWStartIdxTotal = ((curWoIdx * woFactor) * Wi) / Wo;

  LocalTensor<int32_t> cmpIdx = nextCmpBuffer.Get<int32_t>();
  auto cmpIdxTmp = cmpIdx.ReinterpretCast<float>();

  LocalTensor<uint16_t> cmpMask = cmpMaskBuffer.Get<uint16_t>();
  LocalTensor<uint16_t> cmpMask2 = cmpNanMaskBuffer.Get<uint16_t>();
  uint64_t mask = 256 / sizeof(float);
  auto repeat = hiFactor * diFactor;
  UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(VL_NUM / 8 * wiFactorAlign)};
  BinaryRepeatParams repeatParams{1, 1, 1, 8, (uint8_t)(8 * wiFactorAlign), 8};
  BinaryRepeatParams repeatParams2{1, 1, 1, 8, (uint8_t)(8 * wiFactorAlign), (uint8_t)(8 * wiFactorAlign)};

  LocalTensor<float> mulWUb = mulWBuffer.Get<float>();
  LocalTensor<int32_t> mulWIdxUb = mulWIdxBuffer.Get<int32_t>();
  auto mulWIdxCastUb = mulWIdxUb.ReinterpretCast<float>();

  for (int kernelIdx = 0; kernelIdx < curWoFactor; kernelIdx++) {
    int32_t kerWStartIdx = ((kernelIdx + curWoIdx * woFactor) * Wi) / Wo;
    int32_t kerWEndIdx = Ceil((kernelIdx + curWoIdx * woFactor + 1) * Wi, Wo);
    auto mulWOffset = kernelIdx * diFactor * hiFactor * VL_NUM;
    auto inputOffset = VL_NUM * (kerWStartIdx - kerWStartIdxTotal);

    Adds(mulWUb[mulWOffset], xLocalTransVL[inputOffset], (float)0.0, VL_NUM, repeat, repeatCopyParams);
    Adds(mulWIdxUb[mulWOffset], resetIdx, (kerWStartIdx - kerWStartIdxTotal), VL_NUM * repeat);
    AscendC::PipeBarrier<PIPE_V>();
    for (int i = kerWStartIdx + 1; i < kerWEndIdx; i++) {
      Adds(cmpIdx, resetIdx, (i - kerWStartIdxTotal), VL_NUM * repeat);
      auto nexCmpOffset = VL_NUM * (i - kerWStartIdxTotal);
      Compare(cmpMask, xLocalTransVL[nexCmpOffset], mulWUb[mulWOffset], CMPMODE::GT, mask, repeat, repeatParams);
      Compare(cmpMask2, xLocalTransVL[nexCmpOffset], xLocalTransVL[nexCmpOffset], CMPMODE::EQ, mask, repeat,
              repeatParams2);
      AscendC::PipeBarrier<PIPE_V>();
      Not(cmpMask2, cmpMask2, 128);
      AscendC::PipeBarrier<PIPE_V>();
      Or(cmpMask, cmpMask, cmpMask2, 128);
      AscendC::PipeBarrier<PIPE_V>();
      Select(mulWUb[mulWOffset], cmpMask, xLocalTransVL[nexCmpOffset], mulWUb[mulWOffset], selMode, mask, repeat,
             repeatParams);
      Select(mulWIdxCastUb[mulWOffset], cmpMask, cmpIdxTmp, mulWIdxCastUb[mulWOffset], selMode, VL_NUM * repeat);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::MaxPoolH(const uint8_t diFactor, const uint8_t hiFactor,
                                                               const uint8_t curWoFactor, int64_t curHoIdx,
                                                               int64_t curHoFactor) {
  auto woFactorAlign = Ceil(curWoFactor, 8) * 8;

  int32_t kerHStartIdxTotal = ((curHoIdx * hoFactor) * Hi) / Ho;
  LocalTensor<int32_t> cmpIdx = nextCmpBuffer.Get<int32_t>();
  auto cmpIdxTmp = cmpIdx.ReinterpretCast<float>();
  LocalTensor<uint16_t> cmpMask = cmpMaskBuffer.Get<uint16_t>();
  LocalTensor<uint16_t> cmpMask2 = cmpNanMaskBuffer.Get<uint16_t>();
  uint64_t mask = 256 / sizeof(float);
  auto repeat = woFactorAlign * diFactor;
  UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(VL_NUM / 8 * hiFactor)};
  BinaryRepeatParams repeatParams{1, 1, 1, 8, (uint8_t)(8 * hiFactor), 8};
  BinaryRepeatParams repeatParams2{1, 1, 1, 8, (uint8_t)(8 * hiFactor), (uint8_t)(8 * hiFactor)};
  LocalTensor<float> mulWUb = mulWBuffer.Get<float>();
  LocalTensor<int32_t> mulWIdxUb = mulWIdxBuffer.Get<int32_t>();
  auto mulWIdxCastUb = mulWIdxUb.ReinterpretCast<float>();
  LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
  LocalTensor<int32_t> mulWBufferInt = mulWBuffer.Get<int32_t>();
  LocalTensor<int32_t> mulHIdxUb = mulWBufferInt[8 * 1024];  // use mulWBuffer last 32K as index buff
  auto mulHIdxCastUb = mulHIdxUb.ReinterpretCast<float>();

  for (int kernelIdx = 0; kernelIdx < curHoFactor; kernelIdx++) {
    int32_t kerHStartIdx = ((kernelIdx + curHoIdx * hoFactor) * Hi) / Ho;
    int32_t kerHEndIdx = Ceil((kernelIdx + curHoIdx * hoFactor + 1) * Hi, Ho);
    auto mulHOffset = kernelIdx * repeat * VL_NUM;
    auto mulWOffset = VL_NUM * (kerHStartIdx - kerHStartIdxTotal);

    Adds(mulHUb[mulHOffset], mulWUb[mulWOffset], (float)0.0, VL_NUM, repeat, repeatCopyParams);
    Adds(mulHIdxUb[mulHOffset], mulWIdxUb[mulWOffset], (int32_t)0, VL_NUM, repeat, repeatCopyParams);
    AscendC::PipeBarrier<PIPE_V>();
    for (int i = kerHStartIdx + 1; i < kerHEndIdx; i++) {
      auto nexCmpOffset = VL_NUM * (i - kerHStartIdxTotal);
      Compare(cmpMask, mulWUb[nexCmpOffset], mulHUb[mulHOffset], CMPMODE::GT, mask, repeat, repeatParams);
      Compare(cmpMask2, mulWUb[nexCmpOffset], mulWUb[nexCmpOffset], CMPMODE::EQ, mask, repeat, repeatParams2);
      AscendC::PipeBarrier<PIPE_V>();
      Not(cmpMask2, cmpMask2, 128);
      AscendC::PipeBarrier<PIPE_V>();
      Or(cmpMask, cmpMask, cmpMask2, 128);
      AscendC::PipeBarrier<PIPE_V>();
      Select(mulHUb[mulHOffset], cmpMask, mulWUb[nexCmpOffset], mulHUb[mulHOffset], selMode, mask, repeat,
             repeatParams);
      Select(mulHIdxCastUb[mulHOffset], cmpMask, mulWIdxCastUb[nexCmpOffset], mulHIdxCastUb[mulHOffset], selMode, mask,
             repeat, repeatParams);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::MaxPoolD(const uint8_t diFactor, const uint8_t hiFactor,
                                                               const uint8_t curWoFactor, int64_t curDoIdx,
                                                               int64_t curDoFactor) {
  auto woFactorAlign = Ceil(curWoFactor, 8) * 8;

  int32_t kerDStartIdxTotal = ((curDoIdx * doFactor) * Di) / Do;
  LocalTensor<int32_t> cmpIdx = nextCmpBuffer.Get<int32_t>();
  auto cmpIdxTmp = cmpIdx.ReinterpretCast<float>();
  LocalTensor<uint16_t> cmpMask = cmpMaskBuffer.Get<uint16_t>();
  LocalTensor<uint16_t> cmpMask2 = cmpNanMaskBuffer.Get<uint16_t>();
  uint64_t mask = 256 / sizeof(float);
  auto repeat = hoFactor * woFactorAlign;
  UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(VL_NUM / 8 * diFactor)};
  BinaryRepeatParams repeatParams{1, 1, 1, 8, (uint8_t)(8 * diFactor), 8};
  BinaryRepeatParams repeatParams2{1, 1, 1, 8, (uint8_t)(8 * diFactor), (uint8_t)(8 * diFactor)};
  LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
  LocalTensor<int32_t> mulWBufferInt = mulWBuffer.Get<int32_t>();
  LocalTensor<int32_t> mulHIdxUb = mulWBufferInt[8 * 1024];  // use mulWBuffer last 32K as index buff
  auto mulHIdxCastUb = mulHIdxUb.ReinterpretCast<float>();

  LocalTensor<float> mulDUb = mulWBuffer.Get<float>();
  LocalTensor<int32_t> mulDIdxUb = mulWIdxBuffer.Get<int32_t>();
  auto mulDIdxCastUb = mulDIdxUb.ReinterpretCast<float>();

  for (int kernelIdx = 0; kernelIdx < curDoFactor; kernelIdx++) {
    int32_t kerDStartIdx = ((kernelIdx + curDoIdx * doFactor) * Di) / Do;
    int32_t kerDEndIdx = Ceil((kernelIdx + curDoIdx * doFactor + 1) * Di, Do);
    auto mulDOffset = kernelIdx * repeat * VL_NUM;
    auto mulHOffset = VL_NUM * (kerDStartIdx - kerDStartIdxTotal);

    Adds(mulDUb[mulDOffset], mulHUb[mulHOffset], (float)0.0, VL_NUM, repeat, repeatCopyParams);
    Adds(mulDIdxUb[mulDOffset], mulHIdxUb[mulHOffset], (int32_t)0.0, VL_NUM, repeat, repeatCopyParams);
    AscendC::PipeBarrier<PIPE_V>();
    for (int i = kerDStartIdx + 1; i < kerDEndIdx; i++) {
      auto nexCmpOffset = VL_NUM * (i - kerDStartIdxTotal);
      Compare(cmpMask, mulHUb[nexCmpOffset], mulDUb[mulDOffset], CMPMODE::GT, mask, repeat, repeatParams);
      Compare(cmpMask2, mulHUb[nexCmpOffset], mulHUb[nexCmpOffset], CMPMODE::EQ, mask, repeat, repeatParams2);
      AscendC::PipeBarrier<PIPE_V>();
      Not(cmpMask2, cmpMask2, 128);
      AscendC::PipeBarrier<PIPE_V>();
      Or(cmpMask, cmpMask, cmpMask2, 128);
      AscendC::PipeBarrier<PIPE_V>();
      Select(mulDUb[mulDOffset], cmpMask, mulHUb[nexCmpOffset], mulDUb[mulDOffset], selMode, mask, repeat,
             repeatParams);
      Select(mulDIdxCastUb[mulDOffset], cmpMask, mulHIdxCastUb[nexCmpOffset], mulDIdxCastUb[mulDOffset], selMode, mask,
             repeat, repeatParams);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }
}

/*
 * 功能：<fp32>类型out和<int32>类型index的转置，把[D, H, W, VL]转为[VL, D, H, W] 
 * 同时会cast out为<T>类型  
 */
template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::TransOutAndIdx(int64_t curDoFactor, int64_t curHoFactor,
                                                                     int64_t curWoFactor, int64_t UbIdxOffset) {
  auto curWoFactorAlign = Ceil(curWoFactor, 8) * 8;
  auto curWoFactorAlign16 = Ceil(curWoFactor, 32 / sizeof(T)) * 32 / sizeof(T);
  LocalTensor<int32_t> mulDIdxUb = mulWIdxBuffer.Get<int32_t>();
  auto mulDIdxCastUb = mulDIdxUb.ReinterpretCast<float>();
  Adds(mulDIdxUb, mulDIdxUb, (int32_t)UbIdxOffset, hoFactor * curWoFactorAlign * doFactor * VL_NUM);
  LocalTensor<int32_t> indexLocal = indexQue.AllocTensor<int32_t>();
  LocalTensor<float> indexLocalTmp = indexLocal.ReinterpretCast<float>();

  AscendC::PipeBarrier<PIPE_V>();
  LocalTensor<T> yLocal = maxQue.AllocTensor<T>();
  LocalTensor<float> mulDUb = mulWBuffer.Get<float>();
  LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
  if constexpr (IsSameType<T, float>::value) {
    OutTranspose(yLocal, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, VL_NUM);

  } else {
    if (curWoFactorAlign == curWoFactorAlign16) {
      OutTranspose(mulHUb, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, VL_NUM);
    } else {
      UnaryRepeatParams repeatCastParams2{(uint16_t)(curWoFactorAlign16 / 8), (uint16_t)(curWoFactorAlign / 8),
                                          (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign16, 16) * 2),
                                          (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 2)};
      OutTranspose(mulHUb[4096], mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, VL_NUM);
      AscendC::PipeBarrier<PIPE_V>();

      Adds(mulHUb, mulHUb[4096], (float)0.0, (uint8_t)(curWoFactorAlign * curDoFactor * curHoFactor), VL_NUM,
           repeatCastParams2);
    }
    AscendC::PipeBarrier<PIPE_V>();

    Cast(yLocal, mulHUb, RoundMode::CAST_ROUND, VL_NUM * curWoFactorAlign16 * curDoFactor * curHoFactor);
  }
  maxQue.EnQue(yLocal);

  OutTranspose(indexLocalTmp, mulDIdxCastUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, VL_NUM);
  indexQue.EnQue(indexLocal);
}

/*
 * 功能：搬出<T>类型out和<int32>类型index    
 */
template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CopyOutAndIdx(int64_t curNcFactor, int64_t curDoFactor,
                                                                    int64_t curHoFactor, int64_t curWoFactor,
                                                                    int64_t yGmOffset) {
  auto curWoFactorAlign = Ceil(curWoFactor, 8) * 8;
  auto curWoFactorAlign16 = Ceil(curWoFactor, 32 / sizeof(T)) * 32 / sizeof(T);

  LocalTensor<T> yLocal = maxQue.DeQue<T>();

  DataCopyExtParams paramsOut2;
  paramsOut2.blockCount = curHoFactor;
  paramsOut2.blockLen = curWoFactor * sizeof(T);
  paramsOut2.srcStride = 0;
  paramsOut2.dstStride = (Wo - curWoFactor) * sizeof(T);
  for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
    for (int64_t dCopyi = 0; dCopyi < curDoFactor; dCopyi++) {
      auto dstAddr = yGmOffset + ncCopyi * Do * Ho * Wo + dCopyi * Ho * Wo;
      auto srcAddr = ncCopyi * Ceil(curDoFactor * curHoFactor * curWoFactorAlign16, 16) * 16 +
                     dCopyi * curHoFactor * curWoFactorAlign16;
      DataCopyPad(maxGm[dstAddr], yLocal[srcAddr], paramsOut2);
    }
  }
  maxQue.FreeTensor(yLocal);

  paramsOut2.blockLen = curWoFactor * sizeof(int32_t);
  paramsOut2.dstStride = (Wo - curWoFactor) * sizeof(int32_t);
  LocalTensor<int32_t> indexLocal = indexQue.DeQue<int32_t>();
  for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
    for (int64_t dCopyi = 0; dCopyi < curDoFactor; dCopyi++) {
      auto dstAddr = yGmOffset + ncCopyi * Do * Ho * Wo + dCopyi * Ho * Wo;
      auto srcAddr = ncCopyi * Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16 +
                     dCopyi * curHoFactor * curWoFactorAlign;
      DataCopyPad(indicesGm[dstAddr], indexLocal[srcAddr], paramsOut2);
    }
  }
  indexQue.FreeTensor(indexLocal);
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::Process() {
  if (cBlockIdx >= useCoreNum) {
    return;
  }

  InitReset();
  for (auto curIdx = beginIdx; curIdx < endIdx; curIdx++) {
    // 按照outer切分，当前在[NC, Doo, Hoo, Woo]上的第几个UB块和当前计算多少块kernel
    auto curNcIdx = curIdx / (doOuter * hoOuter * woOuter);
    auto curNcFactor = curNcIdx == (ncOuter - 1) ? ncTail : ncFactor;
    auto tmpIdx = curIdx % (doOuter * hoOuter * woOuter);
    auto curDoIdx = tmpIdx / (hoOuter * woOuter);
    auto curDoFactor = curDoIdx == (doOuter - 1) ? doTail : doFactor;
    tmpIdx = tmpIdx % (hoOuter * woOuter);
    auto curHoIdx = tmpIdx / woOuter;
    auto curHoFactor = curHoIdx == (hoOuter - 1) ? hoTail : hoFactor;
    auto curWoIdx = tmpIdx % woOuter;
    auto curWoFactor = curWoIdx == (woOuter - 1) ? woTail : woFactor;
    // 按照inner切分，计算当前起始和终止位置
    int32_t kerDStartIdxTotal = ((curDoIdx * doFactor) * Di) / Do;
    int32_t kerHStartIdxTotal = ((curHoIdx * hoFactor) * Hi) / Ho;
    int32_t kerWStartIdxTotal = ((curWoIdx * woFactor) * Wi) / Wo;
    int32_t kerDEndIdxTotal = Ceil((curDoFactor + curDoIdx * doFactor) * Di, Do);
    int32_t kerHEndIdxTotal = Ceil((curHoFactor + curHoIdx * hoFactor) * Hi, Ho);
    int32_t kerWEndIdxTotal = Ceil((curWoFactor + curWoIdx * woFactor) * Wi, Wo);

    const uint8_t diFactor = kerDEndIdxTotal - kerDStartIdxTotal;
    const uint8_t hiFactor = kerHEndIdxTotal - kerHStartIdxTotal;
    const uint8_t wiFactor = kerWEndIdxTotal - kerWStartIdxTotal;

    // 搬入搬出和ub内index相对gm的偏移
    auto xGmOffset =
        curNcIdx * ncFactor * DiHiWi + kerDStartIdxTotal * HiWi + kerHStartIdxTotal * Wi + kerWStartIdxTotal;
    auto yGmOffset = curNcIdx * ncFactor * Do * Ho * Wo + curDoIdx * doFactor * Ho * Wo + curHoIdx * hoFactor * Wo +
                     curWoIdx * woFactor;
    int64_t UbIdxOffset = kerDStartIdxTotal * HiWi + kerHStartIdxTotal * Wi + kerWStartIdxTotal;
    if (curIdx == beginIdx) {
      CopyInput(curNcFactor, diFactor, hiFactor, wiFactor, xGmOffset);
    }
    // [VL_NUM, diFactor, hiFactor, wiFactorAlign] => [diFactor, hiFactor, wiFactorAlign, VL_NUM]
    TransInput(curNcFactor, diFactor, hiFactor, wiFactor);

    if (curIdx != endIdx - 1) {
      CopyIn(curIdx + 1);
    }

    CalReset(diFactor, hiFactor);
    // [diFactor, hiFactor, wiFactorAlign, VL_NUM] => [woFactorAlign, diFactor, hiFactor, VL_NUM]
    MaxPoolW(diFactor, hiFactor, wiFactor, curWoIdx, curWoFactor);
    // [woFactorAlign, diFactor, hiFactor, VL_NUM] => [hoFactor, woFactorAlign, diFactor, VL_NUM]
    MaxPoolH(diFactor, hiFactor, curWoFactor, curHoIdx, curHoFactor);
    // [hoFactor, woFactorAlign, diFactor, VL_NUM] => [doFactor, hoFactor, woFactorAlign, VL_NUM]
    MaxPoolD(diFactor, hiFactor, curWoFactor, curDoIdx, curDoFactor);
    // [doFactor, hoFactor, woFactorAlign, VL_NUM] => [VL_NUM, doFactor, hoFactor, woFactorAlign]
    TransOutAndIdx(curDoFactor, curHoFactor, curWoFactor, UbIdxOffset);
    CopyOutAndIdx(curNcFactor, curDoFactor, curHoFactor, curWoFactor, yGmOffset);
  }
}
#endif  // ADAPTIVE_MAX_POOL3D_SAMLL_POOL_H_