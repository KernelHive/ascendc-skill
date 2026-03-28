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
 * \file upsample_bicubic2d_grad_base.cpp
 * \brief
 */

#include "upsample_bicubic2d_grad_base.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::Init(GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR usrWorkspace, const UpsampleBicubic2dGradTilingData* __restrict tiling_data) {
  // block_id                                              
  block_id = GetBlockIdx();

  tilingData = tiling_data;
  block_h = block_id / tiling_data->innerCoreNumH;
  block_inner_h = block_id % tiling_data->innerCoreNumH;
  block_w = block_id / tiling_data->innerCoreNumW;
  block_inner_w = block_id % tiling_data->innerCoreNumW;

  InitEventId();

  InitGlobalTensors(grad_output, grad_input);

  InitWorkspaceTensors(usrWorkspace);

  InitLocalTensors();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::InitEventId(){
  eventIdMTE3ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>());
  eventIdVToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_S>());
  eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
  eventIdSToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_V>());
  eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>());
}

template <typename T>
__aicore__ inline uint32_t UpsampleBicubic2dGradBase<T>::GetNumPerBlock(){
  if(std::is_same<T, float>::value) {
    return NUM_PER_BLOCK_FLOAT32;
  }
  return NUM_PER_BLOCK_FLOAT16;
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::InitWorkspaceTensors(GM_ADDR usrWorkspace) {
  // workspace
  GM_ADDR usrWorkspace_ = usrWorkspace;

  interGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(usrWorkspace_), tilingData->batch * tilingData->inputH * tilingData->outputW);
  usrWorkspace_ += (tilingData->batch * tilingData->inputH * tilingData->outputW + GetNumPerBlock() - 1) / GetNumPerBlock() * BLOCK_SIZE;

  coeffW.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(usrWorkspace_ + block_id * tilingData->baseNW * NUM_FRACTAL * sizeof(T)), tilingData->baseNW * NUM_FRACTAL);
  coeffWFloat.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_ + block_id * tilingData->baseNW * NUM_FRACTAL * sizeof(T)), tilingData->baseNW * NUM_FRACTAL);
  coeffH.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(usrWorkspace_ + block_id * tilingData->baseNH * NUM_FRACTAL * sizeof(T)), tilingData->baseNH * NUM_FRACTAL);
  coeffHFloat.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace_ + block_id * tilingData->baseNH * NUM_FRACTAL * sizeof(T)), tilingData->baseNH * NUM_FRACTAL);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::InitGlobalTensors(GM_ADDR grad_output, GM_ADDR grad_input) {
  inGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_output), tilingData->batch * tilingData->inputH * tilingData->inputW);
  outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_input), tilingData->batch * tilingData->outputH * tilingData->outputW);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::InitLocalTensors() {
  pipe.InitBuffer(UbBuf, UB_SIZE);
  LocalTensor<uint8_t> tmp = UbBuf.Get<uint8_t>();

  clearUb = tmp.ReinterpretCast<T>();

  coeffUbBuff = tmp.ReinterpretCast<float>();
  coeffUbRes = tmp[32 * 1024].ReinterpretCast<float>();
  coeffUbRes_ = tmp[96 * 1024].ReinterpretCast<T>();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::computeCoeff(int32_t offset, float scale, uint32_t scaleN, int32_t idx[16]) {
  LocalTensor<float> coeffUbBuff0 = coeffUbBuff;
  LocalTensor<float> coeffUbBuff1 = coeffUbBuff0[NUM_FRACTAL];
  LocalTensor<float> coeffUbBuff2 = coeffUbBuff1[NUM_FRACTAL];
  LocalTensor<float> coeffUbBuff3 = coeffUbBuff2[NUM_FRACTAL];
  LocalTensor<float> coeffUbBuff0_ = coeffUbBuff3[NUM_FRACTAL];
  LocalTensor<float> coeffUbBuff2_ = coeffUbBuff0_[NUM_TWO * NUM_FRACTAL];

  for(int i=0;i<16;i++){
    float tmp = static_cast<float>(offset + i);
    if(tilingData->alignCorners){
      tmp *= scale;
    } else {
      tmp = (tmp + static_cast<float>(0.5)) * scale - static_cast<float>(0.5);
    }
    idx[i] = static_cast<int32_t>(tmp);
    if(tmp < 0) idx[i]--;
    tmp -= idx[i];
    coeffUbBuff0.SetValue(i, tmp);
  }
  SetFlag<HardEvent::S_V>(eventIdSToV);
  WaitFlag<HardEvent::S_V>(eventIdSToV);
  Adds<float>(coeffUbBuff3, coeffUbBuff0, static_cast<float>(1), NUM_FRACTAL);
  Duplicate<float>(coeffUbBuff1, static_cast<float>(1), NUM_FRACTAL);
  Duplicate<float>(coeffUbBuff2, static_cast<float>(2), NUM_FRACTAL);
  PipeBarrier<PIPE_V>();
  Sub<float>(coeffUbBuff1, coeffUbBuff1, coeffUbBuff0, NUM_FRACTAL);
  Sub<float>(coeffUbBuff2, coeffUbBuff2, coeffUbBuff0, NUM_FRACTAL);
  PipeBarrier<PIPE_V>();
  Muls<float>(coeffUbBuff0_, coeffUbBuff0, static_cast<float>(1.25), NUM_TWO * NUM_FRACTAL);
  Muls<float>(coeffUbBuff2_, coeffUbBuff2, static_cast<float>(-0.75), NUM_TWO * NUM_FRACTAL);
  PipeBarrier<PIPE_V>();
  Adds<float>(coeffUbBuff0_, coeffUbBuff0_, static_cast<float>(-2.25), NUM_TWO * NUM_FRACTAL);
  Adds<float>(coeffUbBuff2_, coeffUbBuff2_, static_cast<float>(3.75), NUM_TWO * NUM_FRACTAL);
  PipeBarrier<PIPE_V>();
  Mul<float>(coeffUbBuff0_, coeffUbBuff0_, coeffUbBuff0, NUM_TWO * NUM_TWO * NUM_FRACTAL);
  PipeBarrier<PIPE_V>();
  Adds<float>(coeffUbBuff2_, coeffUbBuff2_, static_cast<float>(-6), NUM_TWO * NUM_FRACTAL);
  PipeBarrier<PIPE_V>();
  Mul<float>(coeffUbBuff0_, coeffUbBuff0_, coeffUbBuff0, NUM_TWO * NUM_TWO * NUM_FRACTAL);
  PipeBarrier<PIPE_V>();
  Adds<float>(coeffUbBuff0, coeffUbBuff0_, static_cast<float>(1), NUM_TWO * NUM_FRACTAL);
  Adds<float>(coeffUbBuff2, coeffUbBuff2_, static_cast<float>(3), NUM_TWO * NUM_FRACTAL);
  Duplicate<float>(coeffUbRes, static_cast<float>(0), NUM_FRACTAL * scaleN);
  SetFlag<HardEvent::V_S>(eventIdVToS);
  WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::fillAndCastCoeffW(int32_t offset, int32_t base[2], int32_t idx[16]) {
  for(int i=0;i<16;i++){
    if(offset + i >= tilingData->inputW) {
      break;
    }
    if(i==0) {
      base[0] = idx[i]-1;
      base[0] = base[0] > 0 ? base[0] : 0;
      base[0] = base[0] < tilingData->outputW ? base[0] : (tilingData->outputW - 1);
    }
    base[1] = idx[i]+2;

    float sum_head = 0;
    float sum_end = 0;
    bool need = false;
    for(int j=-1;j<3;j++){
      int32_t idx_ = idx[i] + j;
      float coeff_value = coeffUbBuff.GetValue(((j + 4) % 4) * NUM_FRACTAL + i);
      if(idx_ <= 0) {
        sum_head += coeff_value;
        need = true;
        if(j==2) coeffUbRes.SetValue(i * tilingData->baseNW, sum_head);
      } else if(idx_ >= (tilingData->outputW - 1)) {
        if(need) {
          if(tilingData->outputW == 1){
            sum_end += sum_head;
          } else {
            coeffUbRes.SetValue(i * tilingData->baseNW, sum_head);
          }
          need = false;
        }
        sum_end += coeff_value;
        if(j==2) coeffUbRes.SetValue(tilingData->outputW - 1 - base[0] + i * tilingData->baseNW, sum_end);
      } else{
        if(need) {
          coeffUbRes.SetValue(i * tilingData->baseNW, sum_head);
          need = false;
        }
        coeffUbRes.SetValue(idx_ - base[0] + i * tilingData->baseNW, coeff_value);
      }
    }
  }
  base[1] = base[1] > 0 ? base[1] : 0;
  base[1] = base[1] < tilingData->outputW ? base[1] : (tilingData->outputW - 1);

  if(std::is_same<T, float>::value) {
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    DataCopy<float>(coeffWFloat, coeffUbRes, NUM_FRACTAL * tilingData->baseNW);
    return;
  }

  SetFlag<HardEvent::S_V>(eventIdSToV);
  WaitFlag<HardEvent::S_V>(eventIdSToV);
  Cast<T, float>(coeffUbRes_, coeffUbRes, RoundMode::CAST_RINT, NUM_FRACTAL * tilingData->baseNW);
  SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  DataCopy<T>(coeffW, coeffUbRes_, NUM_FRACTAL * tilingData->baseNW);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::fillAndCastCoeffH(int32_t offset, int32_t base[2], int32_t idx[16]) {
  for(int i=0;i<16;i++){
    if(offset + i >= tilingData->inputH) break;
    if(i==0) {
      base[0] = idx[i]-1;
      base[0] = base[0] > 0 ? base[0] : 0;
      base[0] = base[0] < tilingData->outputH ? base[0] : (tilingData->outputH - 1);
    }
    base[1] = idx[i]+2;
    float sum_head = 0.0;
    float sum_end = 0.0;
    bool need = false;
    for(int j=-1;j<3;j++){
      int32_t idx_ = idx[i] + j;
      float coeff_value = coeffUbBuff.GetValue(((j + 4) % 4) * NUM_FRACTAL + i);
      if(idx_ <= 0) {
        sum_head += coeff_value;
        need = true;
        if(j==2) coeffUbRes.SetValue(i, sum_head);
      } else if(idx_ >= (tilingData->outputH - 1)) {
        if(need) {
          if(tilingData->outputH == 1){
            sum_end += sum_head;
          } else {
            coeffUbRes.SetValue(i, sum_head);
          }
          need = false;
        }
        sum_end += coeff_value;
        if(j==2) coeffUbRes.SetValue((tilingData->outputH - 1 - base[0]) * NUM_FRACTAL + i, sum_end);
      } else{
        if(need) {
          coeffUbRes.SetValue(i, sum_head);
          need = false;
        }
        coeffUbRes.SetValue((idx_ - base[0]) * NUM_FRACTAL + i, coeff_value);
      }
    }
  }
  base[1] = base[1] > 0 ? base[1] : 0;
  base[1] = base[1] < tilingData->outputH ? base[1] : (tilingData->outputH - 1);

  if(std::is_same<T, float>::value) {
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    DataCopy<float>(coeffHFloat, coeffUbRes, NUM_FRACTAL * tilingData->baseNH);
    return;
  }

  SetFlag<HardEvent::S_V>(eventIdSToV);
  WaitFlag<HardEvent::S_V>(eventIdSToV);
  Cast<T, float>(coeffUbRes_, coeffUbRes, RoundMode::CAST_RINT, NUM_FRACTAL * tilingData->baseNH);
  SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  DataCopy<T>(coeffH, coeffUbRes_, NUM_FRACTAL * tilingData->baseNH);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::ProcessW() {
  if(block_w >= tilingData->CoreNumW) {
    return;
  }
  uint32_t offsetW = block_w * tilingData->loopW + (block_w < tilingData->loopTailCoreW ? block_w : tilingData->loopTailCoreW);
  uint32_t realLoopW = tilingData->loopW + ((block_w < tilingData->loopTailCoreW) ? 1 : 0);
  uint32_t offsetInnerBatchW = block_inner_w * tilingData->innerBatchW + (block_inner_w < tilingData->innerBatchTailCoreW ? block_inner_w : tilingData->innerBatchTailCoreW);
  uint32_t realInnerBatchW = tilingData->innerBatchW + (block_inner_w < tilingData->innerBatchTailCoreW ? 1 : 0);
  if(realInnerBatchW != 0){
    MMW.SetOrgShape(realInnerBatchW * tilingData->inputH, tilingData->baseNW, tilingData->inputW, NUM_FRACTAL, tilingData->outputW);
    for(int i = offsetW;i<(offsetW + realLoopW);i++){
      int32_t base[2], idx[16];
      computeCoeff(i * NUM_FRACTAL, tilingData->scalesW, tilingData->baseNW, idx);
      fillAndCastCoeffW(i * NUM_FRACTAL, base, idx);
      MMW.SetTail(realInnerBatchW * tilingData->inputH, base[1]-base[0]+1, ((tilingData->inputW + NUM_FRACTAL - 1) / NUM_FRACTAL - 1) == i ? tilingData->tailW : NUM_FRACTAL);
      MMW.SetTensorA(inGm[offsetInnerBatchW * tilingData->inputH * tilingData->inputW + i * NUM_FRACTAL]);
      MMW.SetTensorB(coeffW);
      SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
      WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
      MMW.IterateAll(interGm[offsetInnerBatchW * tilingData->inputH * tilingData->outputW + base[0]], true);
    }
  }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::ProcessH() {
  if(block_h >= tilingData->CoreNumH) {
    return;
  }
  uint32_t offsetH = block_h * tilingData->loopH + (block_h < tilingData->loopTailCoreH ? block_h : tilingData->loopTailCoreH);
  uint32_t realLoopH = tilingData->loopH + ((block_h < tilingData->loopTailCoreH) ? 1 : 0);
  uint32_t offsetInnerBatchH = block_inner_h * tilingData->innerBatchH + (block_inner_h < tilingData->innerBatchTailCoreH ? block_inner_h : tilingData->innerBatchTailCoreH);
  uint32_t realInnerBatchH = tilingData->innerBatchH + (block_inner_h < tilingData->innerBatchTailCoreH ? 1 : 0);
  if(realInnerBatchH>0){
    for(int i = offsetH;i<(offsetH + realLoopH);i++){
      int32_t base[2], idx[16];
      computeCoeff(i * NUM_FRACTAL, tilingData->scalesH, tilingData->baseNH, idx);
      fillAndCastCoeffH(i * NUM_FRACTAL, base, idx);
      SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
      WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
      for(int j=0;j<realInnerBatchH;j++) {
        MMH.SetTail(base[1]-base[0]+1, tilingData->outputW, ((tilingData->inputH + NUM_FRACTAL - 1) / NUM_FRACTAL - 1) == i ? tilingData->tailH : NUM_FRACTAL);
        MMH.SetTensorA(coeffH);
        MMH.SetTensorB(interGm[i * NUM_FRACTAL * tilingData->outputW + (offsetInnerBatchH + j) * tilingData->inputH * tilingData->outputW]);
        MMH.IterateAll(outGm[(offsetInnerBatchH + j) * tilingData->outputH * tilingData->outputW + base[0] * tilingData->outputW], true);
      }
    }
  }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::ClearGM(const GlobalTensor<T> &dstGlobal, uint32_t loop,
                                                        uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum) {
  uint32_t offset = (loop * baseN + tailN) * block_id 
                  + (block_id < tailCoreNum ? block_id : tailCoreNum) * GetNumPerBlock();
  uint32_t tail = tailN + (block_id < tailCoreNum ? GetNumPerBlock() : 0);

  SetMaskCount();
  if(loop > 0) {
    SetVectorMask<T, MaskMode::COUNTER>(0, baseN);
  } else if(tail > 0) {
    SetVectorMask<T, MaskMode::COUNTER>(0, tail);
  } else {
    SetMaskNorm();
    ResetMask();
    return;
  }
  Duplicate<T, false>(clearUb, static_cast<T>(0), MASK_PLACEHOLDER, 1,
                  DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
  SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  SetMaskNorm();
  ResetMask();

  for(int i = 0; i < loop; i++) {
    DataCopy<T>(dstGlobal[offset], clearUb, baseN);
    offset += baseN;
  }
  if(tail > 0){
    DataCopy<T>(dstGlobal[offset], clearUb, tail);
  }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradBase<T>::Process() {
  ClearGM(interGm, tilingData->clearInterLoop, tilingData->clearBaseN, tilingData->clearInterTailN, tilingData->clearInterTailCoreNum);
  SyncAll();
  ProcessW();
  SyncAll();
  ClearGM(outGm, tilingData->clearOutLoop, tilingData->clearBaseN, tilingData->clearOutTailN, tilingData->clearOutTailCoreNum);
  SyncAll();
  ProcessH();
  PipeBarrier<PIPE_ALL>();
}