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
 * \file dynamic_quant_large_shape_opt.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_LARGE_SHAPE_OPT_H
#define DYNAMIC_QUANT_LARGE_SHAPE_OPT_H

#include "dynamic_quant_base.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

template <typename T1, typename yDtype>
class DynamicQuantLargeShapeOpt : public DynamicQuantBase {
 public:
  __aicore__ inline DynamicQuantLargeShapeOpt(TPipe *pipe) {
    pPipe = pipe;
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                              GM_ADDR workSpace, const DynamicQuantTilingData* __restrict tilingData) {
    ParseTilingData(tilingData);
    InitLargeShapeParams(offset);
    InitLargeShapeBaseBuffer();
    InitAndSetBuffer(x, smooth_scales, y, scale, offset);
  }

  __aicore__ inline void Process() {
    if (isAsymmetrical) {
      ProcessAsymmetrical();
    } else {
      ProcessSymmetrical();
    }
  }

  __aicore__ inline void ProcessSymmetrical() {
    uint32_t offsetBase = 0;
    uint32_t srcOffset = 0;
    float maxUpdateValue = 0.0;
    float scale;

    LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
    for (uint32_t i = 0; i < loopCnt; i++) {
      maxUpdateValue = MIN_FLOAT_VALUE;
      offsetBase = i * tilingData_.rowLen;
      // 计算每一行的最大值
      for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
        ComputeMaxValue(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0, maxUpdateValue);
      }
      // 如果核内切的有尾块
      if (tilingData_.innerLoopTail > 0) {
        srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
        // 直接进行计算，可以最后一次尾块的搬运
        CopyInByEle(srcOffset, tilingData_.innerLoopTimes, tilingData_.innerLoopTail, rightPadding);
        ComputeTail(tilingData_.innerLoopTail, maxUpdateValue);
        CopyOut(srcOffset, tilingData_.innerLoopTail);
      }
      GetScale(maxUpdateValue, scale);
      scaleLocal.SetValue(i, 1 / scale);

      // 量化计算
      for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
        QuantCompute(srcOffset, innerLoopIndex, scale, 0.0, tilingData_.innerLoopEle, 0);
      }
    }
    scaleQueue.EnQue<float>(scaleLocal);
    CopyScalesOut(loopCnt);
  }

  __aicore__ inline void ProcessAsymmetrical() {
    uint32_t offsetBase = 0;
    uint32_t srcOffset = 0;
    float maxUpdateValue = 0.0;
    float minUpdateValue = 0.0;
    float scale;
    float offset;

    LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
    LocalTensor<float> offsetLocal = offsetQueue.AllocTensor<float>();
    for (uint32_t i = 0; i < loopCnt; i++) {
      maxUpdateValue = MIN_FLOAT_VALUE;
      minUpdateValue = MAX_FLOAT_VALUE;
      offsetBase = i * tilingData_.rowLen;
      // 计算每一行的最大值以及最小值
      for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
        ComputeMaxAndMinValue(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0, maxUpdateValue, minUpdateValue);
      }
      // 如果核内切的有尾块
      if (tilingData_.innerLoopTail > 0) {
        srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
        ComputeMaxAndMinValue(srcOffset, tilingData_.innerLoopTimes, tilingData_.innerLoopTail, rightPadding,
                              maxUpdateValue, minUpdateValue);
      }

      GetScaleAndOffset(maxUpdateValue, minUpdateValue, scale, offset);
      scaleLocal.SetValue(i, scale);
      offsetLocal.SetValue(i, offset);

      // 量化计算
      for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
        QuantCompute(srcOffset, innerLoopIndex, scale, offset, tilingData_.innerLoopEle, 0);
      }
      // 如果核内切的有尾块
      if (tilingData_.innerLoopTail > 0) {
        srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
        QuantCompute(srcOffset, tilingData_.innerLoopTimes, scale, offset, tilingData_.innerLoopTail, rightPadding);
      }
    }
    scaleQueue.EnQue<float>(scaleLocal);
    CopyScalesOut(loopCnt);
    offsetQueue.EnQue<float>(offsetLocal);
    CopyOffsetOut(loopCnt);
  }

 private:
  __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset) {
    if (tilingData_.hasSmooth) {
      smoothGm.SetGlobalBuffer((__gm__ T1*)smooth_scales);
      pPipe->InitBuffer(smoothQueue, BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T1));
    }
    if (blockIdx < tilingData_.headCoreNum) {
      inGm.SetGlobalBuffer((__gm__ T1*)x + blockIdx * lenHead, lenHead);
      outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * outLenHead, outLenHead);
      scaleGm.SetGlobalBuffer((__gm__ float*)scale + blockIdx * rowPerHeadCore, rowPerHeadCore);
      if (isAsymmetrical) {
        offsetGm.SetGlobalBuffer((__gm__ float*)offset + blockIdx * rowPerHeadCore, rowPerHeadCore);
      }
    } else {
      inGm.SetGlobalBuffer(
          (__gm__ T1*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail, lenTail);
      outGm.SetGlobalBuffer(
          (__gm__ yDtype*)y + tilingData_.headCoreNum * outLenHead + (blockIdx - tilingData_.headCoreNum) * outLenTail,
          outLenTail);
      scaleGm.SetGlobalBuffer((__gm__ float*)scale + tilingData_.headCoreNum * rowPerHeadCore +
                                  (blockIdx - tilingData_.headCoreNum) * rowPerTailCore,
                              rowPerTailCore);
      if (isAsymmetrical) {
        offsetGm.SetGlobalBuffer((__gm__ float*)offset + tilingData_.headCoreNum * rowPerHeadCore +
                                     (blockIdx - tilingData_.headCoreNum) * rowPerTailCore,
                                 rowPerTailCore);
      }
    }
    if (isAsymmetrical) {
      pPipe->InitBuffer(offsetQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
    }
    // innerLoopEle已经保证了32Bytes对齐
    pPipe->InitBuffer(inQueue, BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T1));
    pPipe->InitBuffer(outQueue, BUFFER_NUM, tilingData_.innerLoopEle * sizeof(yDtype));
    pPipe->InitBuffer(scaleQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
  }

  __aicore__ inline void CopyInByEle(uint32_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding) {
    LocalTensor<T1> inLocal = inQueue.AllocTensor<T1>();
    DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(T1)), 0, 0};
    DataCopyPadParams padParams{true, 0, rightPadding, 0};
    DataCopyPad(inLocal, inGm[offset], copyParams, padParams);
    inQueue.EnQue(inLocal);

    if (tilingData_.hasSmooth) {
      LocalTensor<T1> smoothLocal = smoothQueue.AllocTensor<T1>();
      // 一次拷贝不完
      DataCopyParams copyParams1{1, (uint16_t)(elementNum * sizeof(T1)), 0, 0};
      DataCopyPadParams padParams1{true, 0, rightPadding, 0};
      DataCopyPad(smoothLocal, smoothGm[loopIndex * tilingData_.innerLoopEle], copyParams1, padParams1);
      smoothQueue.EnQue(smoothLocal);
    }
  }

  __aicore__ inline void CopyScalesOut(uint32_t element) {
    LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
    DataCopyParams copyParams{1, (uint16_t)(element * sizeof(float)), 0, 0};
    DataCopyPad(scaleGm, scaleLocal, copyParams);
    scaleQueue.FreeTensor(scaleLocal);
  }

  __aicore__ inline void CopyOffsetOut(uint32_t element) {
    LocalTensor<float> offsetLocal = offsetQueue.DeQue<float>();
    DataCopyParams copyParams{1, (uint16_t)(element * sizeof(float)), 0, 0};
    DataCopyPad(offsetGm, offsetLocal, copyParams);
    offsetQueue.FreeTensor(offsetLocal);
  }

  __aicore__ inline void QuantCompute(uint32_t offset, uint32_t loopIndex, float scale, float offsetOut,
                                      uint32_t elementNum, uint8_t rightPadding) {
    CopyInByEle(offset, loopIndex, elementNum, rightPadding);
    Compute(scale, offsetOut, elementNum);
    CopyOut(offset, elementNum);
  }

  __aicore__ inline void ComputeMaxValue(uint32_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding,
                                         float& maxUpdateValue) {
    CopyInByEle(offset, loopIndex, elementNum, rightPadding);
    ComputeGetMaxValue(elementNum, maxUpdateValue);
  }

  __aicore__ inline void ComputeTail(uint32_t elementNum, float& maxUpdateValue) {
    float scale = 0.0f;
    LocalTensor<T1> smoothScalsLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
    LocalTensor<int32_t> tempCastInt32 = fp32_buf_.Get<int32_t>();
    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();
    if (tilingData_.hasSmooth) {
      smoothScalsLocal = smoothQueue.DeQue<T1>();
      Cast(temp, smoothScalsLocal, RoundMode::CAST_NONE, elementNum);
      PipeBarrier<PIPE_V>();
      Mul(tempFp32, tempFp32, temp, elementNum);
      PipeBarrier<PIPE_V>();
      smoothQueue.FreeTensor(smoothScalsLocal);
    }
    Abs(temp, tempFp32, elementNum);
    PipeBarrier<PIPE_V>();
    ReduceMaxInplace(temp, elementNum);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float maxValue = temp.GetValue(0);
    maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
    GetScale(maxUpdateValue, scale);
    Muls(tempFp32, tempFp32, scale, elementNum);
    PipeBarrier<PIPE_V>();
    Cast(tempCastInt32, tempFp32, RoundMode::CAST_RINT, elementNum);
    PipeBarrier<PIPE_V>();
    AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
    SetDeqScale(static_cast<half>(1.0));
    PipeBarrier<PIPE_V>();
    Cast(tempHalfCast, tempCastInt32, RoundMode::CAST_ROUND, elementNum);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, tempHalfCast, RoundMode::CAST_TRUNC, elementNum);
    outQueue.EnQue<yDtype>(outLocal);
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void ComputeGetMaxValue(uint32_t elementNum, float& maxUpdateValue) {
    LocalTensor<T1> smoothLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();

    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();
    if (tilingData_.hasSmooth) {
      smoothLocal = smoothQueue.DeQue<T1>();
      Cast(temp, smoothLocal, RoundMode::CAST_NONE, elementNum);
      PipeBarrier<PIPE_V>();
      Mul(tempFp32, tempFp32, temp, elementNum);
      PipeBarrier<PIPE_V>();
      smoothQueue.FreeTensor(smoothLocal);
    }
    Abs(temp, tempFp32, elementNum);
    PipeBarrier<PIPE_V>();
    ReduceMaxInplace(temp, elementNum);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float maxValue = temp.GetValue(0);
    maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void ComputeMaxAndMinValue(uint32_t offset, uint32_t loopIndex, uint32_t elementNum,
                                               uint8_t rightPadding, float& maxUpdateValue, float& minUpdataValue) {
    CopyInByEle(offset, loopIndex, elementNum, rightPadding);
    ComputeGetMaxAndMinValue(elementNum, maxUpdateValue, minUpdataValue);
  }

  __aicore__ inline void ComputeGetMaxAndMinValue(uint32_t elementNum, float& maxUpdateValue, float& minUpdateValue) {
    LocalTensor<T1> smoothLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();

    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();
    if (tilingData_.hasSmooth) {
      smoothLocal = smoothQueue.DeQue<T1>();
      Cast(temp, smoothLocal, RoundMode::CAST_NONE, elementNum);
      PipeBarrier<PIPE_V>();
      Mul(tempFp32, tempFp32, temp, elementNum);
      PipeBarrier<PIPE_V>();
      smoothQueue.FreeTensor(smoothLocal);
    }
    ReduceMax(temp, tempFp32, temp, elementNum, false);
    PipeBarrier<PIPE_V>();
    float maxValue = temp.GetValue(0);
    maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
    ReduceMin(temp, tempFp32, temp, elementNum, false);
    PipeBarrier<PIPE_V>();
    float minValue = temp.GetValue(0);
    minUpdateValue = (minUpdateValue > minValue) ? minValue : minUpdateValue;
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void Compute(float scale, float offset, uint32_t elementNum) {
    LocalTensor<T1> smoothScalsLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
    LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();

    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();
    if (tilingData_.hasSmooth) {
      smoothScalsLocal = smoothQueue.DeQue<T1>();
      Cast(temp, smoothScalsLocal, RoundMode::CAST_NONE, elementNum);
      PipeBarrier<PIPE_V>();
      Mul(tempFp32, tempFp32, temp, elementNum);
      PipeBarrier<PIPE_V>();
      smoothQueue.FreeTensor(smoothScalsLocal);
    }
    if (isAsymmetrical) {
      Muls(tempFp32, tempFp32, 1 / scale, elementNum);
      PipeBarrier<PIPE_V>();
      Adds(tempFp32, tempFp32, offset, elementNum);
    } else {
      Muls(tempFp32, tempFp32, scale, elementNum);
    }
    PipeBarrier<PIPE_V>();
    Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, elementNum);
    PipeBarrier<PIPE_V>();
    AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
    SetDeqScale(static_cast<half>(1.0));
    PipeBarrier<PIPE_V>();
    Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, elementNum);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, tempHalfCast, RoundMode::CAST_TRUNC, elementNum);
    outQueue.EnQue<yDtype>(outLocal);
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t element) {
    LocalTensor<yDtype> outLocal = outQueue.DeQue<yDtype>();
    DataCopyExtParams copyExtParams{(uint16_t)1, (uint16_t)(element * sizeof(yDtype)), 0, 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
      copyExtParams.blockLen = copyExtParams.blockLen >> 1;
      uint32_t index = offset;
      DataCopyPad(outGm[index], outLocal, copyExtParams);
    } else {
      DataCopyPad(outGm[offset], outLocal, copyExtParams);
    }
    outQueue.FreeTensor(outLocal);
  }

 private:
  /* ascendc variable */
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> smoothQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> scaleQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> offsetQueue;

  /* global memory address */
  GlobalTensor<T1> inGm;
  GlobalTensor<T1> smoothGm;
  GlobalTensor<yDtype> outGm;
  GlobalTensor<float> scaleGm;
  GlobalTensor<float> offsetGm;
};
}  // namespace DynamicQuantNDOpt
#endif  // DYNAMIC_QUANT_LARGE_SHAPE_OPT_H