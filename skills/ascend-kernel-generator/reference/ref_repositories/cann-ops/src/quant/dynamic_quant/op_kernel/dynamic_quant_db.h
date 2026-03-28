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
 * \file dynamic_quant_db.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_DB_H
#define DYNAMIC_QUANT_DB_H

#include "dynamic_quant_base.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class DynamicQuantDbOpt : public DynamicQuantBase {
 public:
  __aicore__ inline DynamicQuantDbOpt(TPipe* pipe) {
    pPipe = pipe;
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                              GM_ADDR workSpace, const DynamicQuantTilingData* __restrict tilingData) {
    ParseTilingData(tilingData);
    InitParams(offset);
    InitBaseBuffer();
    InitAndSetBuffer(x, smooth_scales, y, scale, offset);
  }

  __aicore__ inline void Process() {
    LocalTensor<xDtype> smoothHalfLocal;
    LocalTensor<float> smoothLocal;
    // copy smooth from GM to UB
    if (tilingData_.hasSmooth) {
      smoothLocal = smooth_buf_.Get<float>();
      SmoothCopyIn();
      smoothHalfLocal = smoothQueue.DeQue<xDtype>();
      PipeBarrier<PIPE_V>();
      Cast(smoothLocal, smoothHalfLocal, RoundMode::CAST_NONE, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
    }
    DuplicateConst();
    CopyIn(multiRowNum, 0);
    for (uint32_t i = 1; i < loopCnt; i++) {
      CopyIn(multiRowNum, i);
      ProcessMultiRow(smoothLocal, multiRowNum, i - 1);
    }

    if (remainRow > 0) {
      CopyIn(remainRow, loopCnt);
    }

    ProcessMultiRow(smoothLocal, multiRowNum, loopCnt - 1);

    if (remainRow > 0) {
      ProcessMultiRow(smoothLocal, remainRow, loopCnt);
    }

    if (tilingData_.hasSmooth) {
      smoothQueue.FreeTensor(smoothHalfLocal);
    }
  }

 private:
  __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset) {
    if (tilingData_.hasSmooth) {
      smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
      pPipe->InitBuffer(smooth_buf_, sizeHalfLen * sizeof(float));
      pPipe->InitBuffer(smoothQueue, BUFFER_NUM, sizeHalfLen * sizeof(xDtype));
    }
    if (blockIdx < tilingData_.headCoreNum) {
      inGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
      outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * outLenHead, outLenHead);
      scaleGm.SetGlobalBuffer((__gm__ float*)scale + blockIdx * rowPerHeadCore, rowPerHeadCore);
      if (isAsymmetrical) {
        offsetGm.SetGlobalBuffer((__gm__ float*)offset + blockIdx * rowPerHeadCore, rowPerHeadCore);
      }
    } else {
      inGm.SetGlobalBuffer(
          (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
          lenTail);
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
      pPipe->InitBuffer(offsetBuf, sizeFloatLen * sizeof(float));
    }
    pPipe->InitBuffer(inQueue, DOUBLE_BUFFER_NUM, lenMultiRow * sizeof(xDtype));
    pPipe->InitBuffer(outBuf, outLen * sizeof(yDtype));
    pPipe->InitBuffer(scaleBuf, sizeFloatLen * sizeof(float));
  }

  __aicore__ inline void SmoothCopyIn() {
    LocalTensor<xDtype> smoothLocal = smoothQueue.AllocTensor<xDtype>();
    if (isPad) {
      DataCopyParams copyParams{1, (uint16_t)(tilingData_.rowLen * sizeof(xDtype)), 0, 0};
      DataCopyPadParams padParams{true, 0, rightPadding, 0};
      DataCopyPad(smoothLocal, smoothGm, copyParams, padParams);
    } else {
      DataCopy(smoothLocal, smoothGm, tilingData_.rowLen);
    }
    smoothQueue.EnQue(smoothLocal);
  }

  __aicore__ inline void CopyIn(uint32_t multiRow, uint32_t loopNum) {
    LocalTensor<xDtype> inLocal = inQueue.AllocTensor<xDtype>();
    if (isPad) {
      DataCopyParams copyParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(xDtype)), 0, 0};
      DataCopyPadParams padParams{true, 0, rightPadding, 0};
      DataCopyPad(inLocal, inGm[loopNum * lenGMMultiRow], copyParams, padParams);
    } else {
      DataCopy(inLocal, inGm[loopNum * lenGMMultiRow], multiRow * tilingData_.rowLen);
    }
    inQueue.EnQue(inLocal);
  }

  __aicore__ inline void ProcessMultiRow(const LocalTensor<float>& smoothLocal, uint32_t multiRow, uint32_t loopNum) {
    if (isAsymmetrical) {
      ProcessAsymmetric(smoothLocal, multiRow, loopNum);
    } else {
      if constexpr (IsSameType<xDtype, bfloat16_t>::value) {
        ProcessBf16(smoothLocal, multiRow, loopNum);
      } else {
        ProcessFp16(smoothLocal, multiRow, loopNum);
      }
    }
  }

  __aicore__ inline void ProcessAsymmetric(const LocalTensor<float>& smoothLocal, uint32_t multiRow, uint32_t loopNum) {
    uint32_t index = 0;
    float max_value;
    float min_value;
    float back_scale;
    float scale;
    float offset;

    LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> offsetLocal = offsetBuf.Get<float>();
    LocalTensor<float> tempCast = tempCastUb.Get<float>();
    LocalTensor<yDtype> outLocal = outBuf.AllocTensor<yDtype>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
    AscendC::LocalTensor<float> tempInt32 = fp32_buf_.Get<float>();
    auto tempHalfCast = temp.ReinterpretCast<half>();

    event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
    event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    set_flag(PIPE_MTE3, PIPE_V, event_mte3_v);
    for (uint32_t i = 0; i < multiRow; i++) {
      // cast to fp32
      Cast(tempCast, inLocal[i * sizeHalfLen], RoundMode::CAST_NONE, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      if (tilingData_.hasSmooth) {
        Mul(tempCast, tempCast, smoothLocal, sizeHalfLen);
        PipeBarrier<PIPE_V>();
      }
      ReduceMax(temp, tempCast, temp, tilingData_.rowLen, false);
      event_t event_v_s_max = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, event_v_s_max);
      wait_flag(PIPE_V, PIPE_S, event_v_s_max);
      max_value = temp.GetValue(0);
      ReduceMin(temp, tempCast, temp, tilingData_.rowLen, false);
      event_t event_v_s_min = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, event_v_s_min);
      wait_flag(PIPE_V, PIPE_S, event_v_s_min);
      min_value = temp.GetValue(0);
      if constexpr (IsSameType<yDtype, int4b_t>::value) {
        scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT4_SCALE, DYNAMIC_QUANT_EPSINON);
        offset = DYNAMIC_QUANT_INT4_OFFSET - max_value / scale;
      } else {
        scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT8_SCALE, DYNAMIC_QUANT_EPSINON);
        offset = DYNAMIC_QUANT_INT8_OFFSET - max_value / scale;
      }
      back_scale = 1 / scale;
      if (unlikely(i == 0)) {
        wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
      }
      scaleLocal.SetValue(i, scale);
      offsetLocal.SetValue(i, offset);
      Muls(tempCast, tempCast, back_scale, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      Adds(tempCast, tempCast, offset, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      Cast(tempInt32, tempCast, RoundMode::CAST_RINT, tilingData_.rowLen);
      SetDeqScale(static_cast<half>(1.0));
      PipeBarrier<PIPE_V>();
      Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      if (unlikely(i == 0)) {
        wait_flag(PIPE_MTE3, PIPE_V, event_mte3_v);
      }
      Cast(outLocal[i * outAlignLen], tempHalfCast, RoundMode::CAST_TRUNC, tilingData_.rowLen);
    }
    inQueue.FreeTensor(inLocal);

    event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, event_v_mte3);
    wait_flag(PIPE_V, PIPE_MTE3, event_v_mte3);
    DataCopyExtParams copyOutParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
      copyOutParams.blockLen = copyOutParams.blockLen >> 1;
      uint32_t index = loopNum * lenGMMultiRow;
      DataCopyPad(outGm[index], outLocal, copyOutParams);
    } else {
      DataCopyPad(outGm[loopNum * lenGMMultiRow], outLocal, copyOutParams);
    }

    DataCopyParams copyParams{1, (uint16_t)(multiRow * sizeof(float)), 0, 0};
    DataCopyPad(scaleGm[loopNum * multiRowNum], scaleLocal, copyParams);
    DataCopyPad(offsetGm[loopNum * multiRowNum], offsetLocal, copyParams);
  }

  __aicore__ inline void ProcessBf16(const LocalTensor<float>& smoothLocal, uint32_t multiRow, uint32_t loopNum) {
    uint32_t index = 0;
    float scale;
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
    LocalTensor<float> tempCast = tempCastUb.Get<float>();
    LocalTensor<yDtype> outLocal = outBuf.template Get<yDtype>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
    AscendC::LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
    auto tempHalf = temp.ReinterpretCast<half>();

    event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
    event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    set_flag(PIPE_MTE3, PIPE_V, event_mte3_v);
    for (int32_t i = 0; i < multiRow; i++) {
      // cast to fp32
      index = i * sizeHalfLen;
      Cast(tempCast, inLocal[index], RoundMode::CAST_NONE, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      if (tilingData_.hasSmooth) {
        Mul(tempCast, tempCast, smoothLocal, sizeHalfLen);
        PipeBarrier<PIPE_V>();
      }
      Abs(temp, tempCast, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      ReduceMaxInplace(temp, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      Div(temp, constScale, temp, MAX_VALUE_NUM);
      event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, event_v_s);
      wait_flag(PIPE_V, PIPE_S, event_v_s);
      scale = temp.GetValue(0);
      if (unlikely(i == 0)) {
        wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
      }
      scaleLocal.SetValue(i, 1 / scale);
      Muls(tempCast, tempCast, scale, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      Cast(tempInt32, tempCast, RoundMode::CAST_RINT, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      SetDeqScale(static_cast<half>(1.0));
      PipeBarrier<PIPE_V>();
      Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      if (unlikely(i == 0)) {
        wait_flag(PIPE_MTE3, PIPE_V, event_mte3_v);
      }
      Cast(outLocal[i * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, tilingData_.rowLen);
    }
    inQueue.FreeTensor(inLocal);

    event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, event_v_mte3);
    wait_flag(PIPE_V, PIPE_MTE3, event_v_mte3);
    DataCopyExtParams copyOutParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
      copyOutParams.blockLen = copyOutParams.blockLen >> 1;
      uint32_t index = loopNum * lenGMMultiRow;
      DataCopyPad(outGm[index], outLocal, copyOutParams);
    } else {
      DataCopyPad(outGm[loopNum * lenGMMultiRow], outLocal, copyOutParams);
    }
    DataCopyParams copyParams{1, (uint16_t)(multiRow * sizeof(float)), 0, 0};
    DataCopyPad(scaleGm[loopNum * multiRowNum], scaleLocal, copyParams);
  }

  __aicore__ inline void ProcessFp16(const LocalTensor<float>& smoothLocal, uint32_t multiRow, uint32_t loopNum) {
    uint32_t index = 0;
    float scale;
    LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> tempCast = tempCastUb.Get<float>();
    LocalTensor<yDtype> outLocal = outBuf.template Get<yDtype>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
    AscendC::LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
    auto tempHalfCast = temp.ReinterpretCast<half>();

    event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
    event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    set_flag(PIPE_MTE3, PIPE_V, event_mte3_v);
    for (int32_t i = 0; i < multiRow; i++) {
      // cast to fp32
      index = i * sizeHalfLen;
      Cast(tempCast, inLocal[index], RoundMode::CAST_NONE, tilingData_.rowLen);
      if (tilingData_.hasSmooth) {
        PipeBarrier<PIPE_V>();
        Mul(tempCast, tempCast, smoothLocal, sizeHalfLen);
        PipeBarrier<PIPE_V>();
        Abs(temp, tempCast, tilingData_.rowLen);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(temp, tilingData_.rowLen);
      } else {
        auto x = inLocal[index];
        Abs(x, x, tilingData_.rowLen);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(x, tilingData_.rowLen);
        PipeBarrier<PIPE_V>();
        Cast(temp, x, RoundMode::CAST_NONE, tilingData_.rowLen);
      }
      PipeBarrier<PIPE_V>();
      Div(temp, constScale, temp, MAX_VALUE_NUM);
      event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
      set_flag(PIPE_V, PIPE_S, eventVS);
      wait_flag(PIPE_V, PIPE_S, eventVS);
      scale = temp.GetValue(0);
      if (unlikely(i == 0)) {
        wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
      }
      scaleLocal.SetValue(i, 1 / scale);
      Muls(tempCast, tempCast, scale, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      Cast(tempInt32, tempCast, RoundMode::CAST_RINT, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      SetDeqScale(static_cast<half>(1.0));
      PipeBarrier<PIPE_V>();
      Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
      PipeBarrier<PIPE_V>();
      if (unlikely(i == 0)) {
        wait_flag(PIPE_MTE3, PIPE_V, event_mte3_v);
      }
      Cast(outLocal[i * outAlignLen], tempHalfCast, RoundMode::CAST_TRUNC, tilingData_.rowLen);
    }
    inQueue.FreeTensor(inLocal);

    event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, event_v_mte3);
    wait_flag(PIPE_V, PIPE_MTE3, event_v_mte3);
    DataCopyExtParams copyOutParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
      copyOutParams.blockLen = copyOutParams.blockLen >> 1;
      uint32_t index = loopNum * lenGMMultiRow;
      DataCopyPad(outGm[index], outLocal, copyOutParams);
    } else {
      DataCopyPad(outGm[loopNum * lenGMMultiRow], outLocal, copyOutParams);
    }
    DataCopyParams copyParams{1, (uint16_t)(multiRow * sizeof(float)), 0, 0};
    DataCopyPad(scaleGm[loopNum * multiRowNum], scaleLocal, copyParams);
  }

 private:
  /* ascendc variable */
  TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> smoothQueue;
  TBuf<> outBuf;
  TBuf<> scaleBuf;
  TBuf<> offsetBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> smooth_buf_;

  /* global memory address */
  GlobalTensor<xDtype> inGm;
  GlobalTensor<xDtype> smoothGm;
  GlobalTensor<yDtype> outGm;
  GlobalTensor<float> scaleGm;
  GlobalTensor<float> offsetGm;
};
}  // namespace DynamicQuantNDOpt
#endif  // DYNAMIC_QUANT_DB_H