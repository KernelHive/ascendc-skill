/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file angle_v2_complex.h
 */
#ifndef _ANGLE_V2_COMPLEX_H_
#define _ANGLE_V2_COMPLEX_H_

#include "angle_v2_base.h"

namespace AngleV2N {
constexpr int32_t APPROXIMATE_VALUE = 10000;
constexpr float ATAN_FP32_MAX = 4611686018427387904.0;   // 2^62
constexpr float ATAN_FP32_MIN = 2.168404344971009e-19; // 2^-62
constexpr uint8_t TAYLOR_COUNT_FOUR = 4; // x > 0 and x < tan(pi/8)
constexpr uint8_t TAYLOR_COUNT_SIX = 6; // x > tan(pi/8) and x < tan(pi/4)
constexpr uint8_t GATHER_MASK_MODE_ONE = 1;
constexpr uint8_t GATHER_MASK_MODE_TWO = 2;
constexpr uint32_t ALIGN_NUM_CP64 = 4;
constexpr float NEG_ONE_BY_THREE = -0.3333333333333333;
constexpr float ONE_BY_FIVE = 0.2;
constexpr float NEG_ONE_BY_SEVEN = -0.14285714285714285;
constexpr float ONE_BY_NINE = 0.1111111111111111;
constexpr float NEG_ONE_BY_ELEVEN = -0.09090909090909091;
constexpr float ONE_BY_THIRTEEN =  0.07692307692307693;

template <typename yType>
class AngleV2Complex : public AngleV2Base<yType> {
 public:
  __aicore__ inline AngleV2Complex() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData) {
    this->BaseMemberDataInit(tilingData);
    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;

    uint32_t totalLengthAlignedCp64 = (this->totalLength + ALIGN_NUM_CP64 - 1) / ALIGN_NUM_CP64 * ALIGN_NUM_CP64;
    uint32_t diffLength = this->totalLengthAligned - totalLengthAlignedCp64;

    int64_t current_offset = this->offset;
    complexTailLength = this->lastTileLength * COEFFICENT;
    if (GetBlockIdx() == this->formerNum + this->tailNum - 1) {
      if (this->offset < diffLength) {
        // back the compute elements when the address can not back
        complexTailLength =  this->lastTileLength * COEFFICENT - diffLength * COEFFICENT;
      } else {
        // address back to avoid data stampede
        current_offset = this->offset - diffLength;
      }
    }

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(x) + current_offset * COEFFICENT,
                        this->blockLength * COEFFICENT);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + current_offset, this->blockLength);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(yType) * COEFFICENT);
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));

    pipe.InitBuffer(maskBuf1, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(maskBuf2, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(realBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(imagBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(oneBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf1, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf2, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf3, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf4, this->tileLength * sizeof(yType));
    pipe.InitBuffer(tempBuf5, this->tileLength * sizeof(yType));
  }

  __aicore__ inline void Process() {
    BufferGet();
    // loop count need to be doubled, due to double buffer
    uint32_t complexLength = this->tileLength * COEFFICENT;
    for (int32_t i = 0; i < this->tileNum; i++) {
      int64_t coreOffset = i * static_cast<int64_t>(complexLength);
      CopyInComplex(coreOffset, complexLength);
      event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
      WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
      Compute(this->mask, repeatTimes, this->tileLength);
      event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
      SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
      WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
      CopyOut(i * this->tileLength, this->tileLength);
    }

    if (this->lastTileLength > 0) {
      int64_t coreOffset = static_cast<int64_t>(this->blockLength) * COEFFICENT - complexLength;
      complexLength = this->lastTileLength * COEFFICENT;
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      CopyInComplex(coreOffset, complexTailLength);
      event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
      SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
      WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
      Compute(this->mask, repeatTimes, this->lastTileLength);
      event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
      SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
      WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
      CopyOut(this->blockLength - this->lastTileLength, this->lastTileLength);
    }
  }

 private:
  __aicore__ inline void BufferGet() {
    realLocal = realBuf.Get<yType>();
    imagLocal = imagBuf.Get<yType>();
    zeroTensor = zeroBuf.Get<yType>();
    oneTensor = oneBuf.Get<yType>();
    tempTensor1 = tempBuf1.Get<yType>();
    tempTensor2 = tempBuf2.Get<yType>();
    tempTensor3 = tempBuf3.Get<yType>();
    tempTensor4 = tempBuf4.Get<yType>();
    tempTensor5 = tempBuf5.Get<yType>();
    mask1 = maskBuf1.Get<uint8_t>();
    mask2 = maskBuf2.Get<uint8_t>();

    Duplicate(zeroTensor, static_cast<yType>(0.0), this->tileLength);
    Duplicate(oneTensor, static_cast<yType>(1.0), this->tileLength);
  }

  __aicore__ inline void SplitComplex(LocalTensor<yType> &realLocal, LocalTensor<yType> &imagLocal,
                                      LocalTensor<yType> &input, uint32_t calCount) {
#if (__CCE_AICORE__ >= 200)
    // SplitComplex with GatherMask
    GatherMaskParams params;
    uint64_t rsvdCnt = 0;
    uint32_t complexMask = calCount * COEFFICENT;
    params.repeatTimes = 1;
    GatherMask(realLocal, input, GATHER_MASK_MODE_ONE, true, complexMask, params, rsvdCnt);
    GatherMask(imagLocal, input, GATHER_MASK_MODE_TWO, true, complexMask, params, rsvdCnt);
    PipeBarrier<PIPE_V>();
#else
    // SplitComplex with SetValue
    for (int32_t i = 0; i < calCount; i++) {
      realLocal.SetValue(i, input.GetValue(COEFFICENT * i));
      imagLocal.SetValue(i, input.GetValue(COEFFICENT * i + 1));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
#endif
  }

  __aicore__ inline void CopyInComplex(int64_t coreOffset, uint32_t coreLength) {
    // alloc tensor from queue memory
    LocalTensor<yType> xLocal = inQueue.AllocTensor<yType>();
    // copy progress_th tile from global tensor to local tensor
    DataCopy(xLocal, xGm[coreOffset], coreLength);
    // enque input tensors to VECIN queue
    inQueue.EnQue(xLocal);
  }

  __aicore__ inline void Compute(uint64_t mask, uint8_t repeatTimes, uint32_t calCount) {
    // deque input tensors from VECIN queue
    LocalTensor<yType> input = inQueue.DeQue<yType>();
    LocalTensor<yType> result = outQueue.AllocTensor<yType>();

    // get the real and imag for input(complex)
    SplitComplex(realLocal, imagLocal, input, calCount);

    // calculate atan(imag / real)
    Div(tempTensor5, imagLocal, realLocal, calCount);
    PipeBarrier<PIPE_V>();
#if (__CCE_AICORE__ >= 200)
    // implementation of high level api Atan
    Atan<yType, false>(result, tempTensor5, calCount);
#else
    // implementation of AtanCompute
    AtanCompute(result, tempTensor5, tempTensor1, calCount);
#endif
    PipeBarrier<PIPE_V>();

    // fix result by imag
    mask1 = imagLocal >= zeroTensor;
    Duplicate(tempTensor1, static_cast<yType>(1.0), calCount);
    Duplicate(tempTensor2, static_cast<yType>(-1.0), calCount);
    PipeBarrier<PIPE_V>();
    // tempTensor1 = when imag >= 0 then 1 else -1
    this->DoSelect(tempTensor1, mask1, tempTensor1, tempTensor2, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = realLocal < zeroTensor;
    // tempTensor2 = when (real < 0 and imag >=0) then 1; when (real < 0 and imag < 0) then -1; else 0
    this->DoSelect(tempTensor2, mask1, tempTensor1, zeroTensor, mask, repeatTimes);
    Duplicate(tempTensor3, static_cast<yType>(constData.CONST_PI_BY_TWO), calCount);
    PipeBarrier<PIPE_V>();
    // tempTensor1 = when imag >= 0 then pi / 2 else -pi / 2
    Mul(tempTensor1, tempTensor3, tempTensor1, calCount);
    PipeBarrier<PIPE_V>();

    // fix result by real
    Duplicate(tempTensor3, static_cast<yType>(constData.CONST_PI), calCount);
    PipeBarrier<PIPE_V>();
    // tempTensor2 = when (real < 0 and imag >=0) then pi; when (real < 0 and imag < 0) then -pi; else 0 
    Mul(tempTensor2, tempTensor3, tempTensor2, calCount);
    mask1 = realLocal == zeroTensor;
    PipeBarrier<PIPE_V>();
    // result = when (x == 0 and imag >= 0) then pi / 2; when (x == 0 and imag < 0) then -pi / 2; else result
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);
    PipeBarrier<PIPE_V>();

    // result = when (real < 0 and imag >= 0) then result + pi; when (real < 0 and imag < 0) then result - pi; else result
    Add(result, tempTensor2, result, calCount);
    PipeBarrier<PIPE_V>();
    outQueue.EnQue<yType>(result);
    CornerProcess(mask, repeatTimes, calCount);
    // free input tensors for reuse
    inQueue.FreeTensor(input);
  }

  __aicore__ inline void CornerProcess(uint64_t mask, uint8_t repeatTimes, uint32_t calCount) {
    LocalTensor<yType> result = outQueue.DeQue<yType>();
  
    mask1 = imagLocal == zeroTensor;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor1, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = imagLocal < zeroTensor;
    PipeBarrier<PIPE_V>();
    // tempTensor2(signbit_imag) = when imag < 0 then 1 else 0
    this->DoSelect(tempTensor2, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    Mul(tempTensor2, tempTensor1, tempTensor2, calCount);
    PipeBarrier<PIPE_V>();
    // tempTensor3 = when (imag = 0 and imag >= 0) then 1; when (imag = 0 and imag < 0) then -1; else 0
    Sub(tempTensor3, tempTensor1, tempTensor2, calCount);

    // real < 0 and imag = +0 --> pi
    mask1 = realLocal < zeroTensor;
    PipeBarrier<PIPE_V>();
    // tempTensor1(signbit_real) = when real < 0 then 1 else 0
    this->DoSelect(tempTensor1, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    Sub(tempTensor4, oneTensor, tempTensor1, calCount);
    Mul(tempTensor5, tempTensor3, tempTensor1, calCount);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor5 == oneTensor;
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensor5, static_cast<yType>(constData.CONST_PI), calCount);
    PipeBarrier<PIPE_V>();
    // result = when (imag == +0 and real < 0) then pi else result
    this->DoSelect(result, mask1, tempTensor5, result, mask, repeatTimes);

    // real >= 0 and imag = +0 -->0
    Mul(tempTensor3, tempTensor4, tempTensor3, calCount);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor3 == oneTensor;
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, zeroTensor, result, mask, repeatTimes);

    // real < 0 and imag = -0 --> -pi
    Mul(tempTensor1, tempTensor2, tempTensor1, calCount);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor1 == oneTensor;
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensor1, static_cast<yType>(constData.CONST_NEG_PI), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);

    // real >= 0 and imag =-0 --> -0
    Mul(tempTensor4, tempTensor2, tempTensor4, calCount);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor4 == oneTensor;
    Duplicate(tempTensor1, static_cast<yType>(float(-0.0)), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);

    CornerProcessINFNAN(result, oneTensor, mask, repeatTimes, calCount);
    // enque the output tensor to VECOUT queue
    outQueue.EnQue<yType>(result);
  }

  __aicore__ inline void CornerProcessINFNAN(LocalTensor<yType> &result, LocalTensor<yType> &oneTensor, uint64_t mask,
                                             uint8_t repeatTimes, uint32_t calCount) {
    CornerProcessRealINF(result, oneTensor, mask, repeatTimes, calCount);
    CornerProcessRealNINF(result, oneTensor, mask, repeatTimes, calCount);
    CornerProcessNAN(result, mask, repeatTimes, calCount);
  }

  __aicore__ inline void CornerProcessRealINF(LocalTensor<yType> &result, LocalTensor<yType> &oneTensor, uint64_t mask,
                                              uint8_t repeatTimes, uint32_t calCount) {
    // real = inf and -inf < imag < 0 --> -0
    mask1 = imagLocal < zeroTensor;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor2, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    Duplicate(tempTensor3, static_cast<yType>(-INFINITY), calCount);
    PipeBarrier<PIPE_V>();
    mask1 = imagLocal > tempTensor3;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor2, mask1, tempTensor2, zeroTensor, mask, repeatTimes);
    Duplicate(tempTensor4, static_cast<yType>(INFINITY), calCount);
    PipeBarrier<PIPE_V>();
    mask2 = realLocal == tempTensor4;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor2, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor2 == oneTensor;
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);
    PipeBarrier<PIPE_V>();

    // real = inf and imag = inf --> pi / 4
    mask1 = imagLocal == tempTensor4;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor1, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = imagLocal == tempTensor3;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor2, mask1, oneTensor, zeroTensor, mask, repeatTimes);
    this->DoSelect(tempTensor5, mask2, tempTensor1, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor5 == oneTensor;
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensor5, static_cast<yType>(constData.CONST_PI_BY_FOUR), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor5, result, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
  
    // real = inf and imag = -inf --> -pi / 4
    this->DoSelect(tempTensor5, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor5 == oneTensor;
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensor5, static_cast<yType>(constData.CONST_NEG_PI_BY_FOUR), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor5, result, mask, repeatTimes);
  }

  __aicore__ inline void CornerProcessRealNINF(LocalTensor<yType> &result, LocalTensor<yType> &oneTensor, uint64_t mask,
                                               uint8_t repeatTimes, uint32_t calCount) {
    // real = -inf and imag = inf --> 3pi / 4
    mask2 = realLocal == tempTensor3;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor3, mask2, tempTensor1, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor3 == oneTensor;
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensor3, static_cast<yType>(constData.CONST_PI_BY_THREE_QUARTERS), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor3, result, mask, repeatTimes);
    PipeBarrier<PIPE_V>();

    // real = -inf and imag = -inf --> -3pi / 4
    this->DoSelect(tempTensor3, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor3 == oneTensor;
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensor3, static_cast<yType>(constData.CONST_NEG_PI_BY_THREE_QUARTERS), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor3, result, mask, repeatTimes);
    PipeBarrier<PIPE_V>();

    // abs_real < inf and imag = inf --> pi / 2
    Abs(tempTensor3, realLocal, calCount);
    PipeBarrier<PIPE_V>();
    mask2 = tempTensor3 < tempTensor4;
    PipeBarrier<PIPE_V>();
    this->DoSelect(tempTensor1, mask2, tempTensor1, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor1 == oneTensor;
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensor1, static_cast<yType>(constData.CONST_PI_BY_TWO), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);

    // abs_real < inf and imag = -inf --> -pi / 2
    this->DoSelect(tempTensor2, mask2, tempTensor2, zeroTensor, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = tempTensor2 == oneTensor;
    Duplicate(tempTensor1, static_cast<yType>(constData.CONST_NEG_PI_BY_TWO), calCount);
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, tempTensor1, result, mask, repeatTimes);
  }

  __aicore__ inline void CornerProcessNAN(LocalTensor<yType> &result, uint64_t mask, uint8_t repeatTimes,
                                          uint32_t calCount) {
    Duplicate(tempTensor1, static_cast<yType>(NAN), calCount);
    mask1 = realLocal == realLocal;
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, result, tempTensor1, mask, repeatTimes);
    PipeBarrier<PIPE_V>();
    mask1 = imagLocal == imagLocal;
    PipeBarrier<PIPE_V>();
    this->DoSelect(result, mask1, result, tempTensor1, mask, repeatTimes);
  }

  __aicore__ inline void CopyOut(uint32_t coreOffset, uint32_t coreLength) {
    // deque output tensor from VECOUT queue
    LocalTensor<yType> result = outQueue.DeQue<yType>();
    // copy progress_th tile from local tensor to global tensor
    DataCopy(yGm[coreOffset], result, coreLength);
    // free output tensor for reuse
    outQueue.FreeTensor(result);
  }

  __aicore__ inline void Sign(LocalTensor<yType> &dst, LocalTensor<yType> &src, LocalTensor<yType> &denominator,
                              uint32_t calCount) {
    Muls(dst, src, static_cast<yType>(ATAN_FP32_MAX), calCount);
    PipeBarrier<PIPE_V>();
    Abs(denominator, dst, calCount);
    PipeBarrier<PIPE_V>();
    Adds(denominator, denominator, static_cast<yType>(ATAN_FP32_MIN), calCount);
    PipeBarrier<PIPE_V>();
    Div(dst, dst, denominator, calCount);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void TaylorExpand(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                      LocalTensor<yType> &squareTensor, int32_t expandLevel, uint32_t calCount) {
    // arctan(x) = x - x^3/3 + x^5/5 + ... + (-1)^k*x^(k*2+1)/( k*2+1)
    // The initial value of dstTensor is assigned as the coefficient of the last item of expansion.
    Mul(squareTensor, srcTensor, srcTensor, calCount);
    Mul(dstTensor, srcTensor, srcTensor, calCount);
    PipeBarrier<PIPE_V>();
    Muls(dstTensor, dstTensor, factorList[expandLevel], calCount);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = expandLevel - 1; i > 0; --i) {
        // dst*x^2+ the previois expand factor
        Adds(dstTensor, dstTensor, factorList[i], calCount);
        PipeBarrier<PIPE_V>();
        Mul(dstTensor, dstTensor, squareTensor, calCount);
        PipeBarrier<PIPE_V>();
    }
    Adds(dstTensor, dstTensor, factorList[0], calCount);
    PipeBarrier<PIPE_V>();
    Mul(dstTensor, dstTensor, srcTensor, calCount);
  }

  __aicore__ inline void AtanTransform(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                       LocalTensor<yType> &tmpTensor, const float transFactor, uint32_t calCount) {
    // (x-y)/(1+xy)
    const float transFactorNeg = 0 - transFactor;

    // x*y
    Muls(dstTensor, srcTensor, transFactor, calCount);
    PipeBarrier<PIPE_V>();
    // x*y + 1
    Adds(dstTensor, dstTensor, static_cast<yType>(1.0), calCount);
    // x=x-y
    Adds(tmpTensor, srcTensor, transFactorNeg, calCount);
    PipeBarrier<PIPE_V>();
    // (x-y)/(1+xy)
    Div(dstTensor, tmpTensor, dstTensor, calCount);
    PipeBarrier<PIPE_V>();
    // abs(x)
    Abs(dstTensor, dstTensor, calCount);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void AtanImpl(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                  LocalTensor<yType> &tmpTensor, uint32_t calCount) {
    LocalTensor<yType> absTensor = tempTensor3;
    LocalTensor<yType> tmpTensor2 = tempTensor2;
    LocalTensor<yType> squareTensor = tempTensor4;

    Abs(absTensor, srcTensor, calCount);
    PipeBarrier<PIPE_V>();

    // when x's value is too large the first caculator of TaylorExpand will be overflow. when epsilon is 0.0001,
    // the approximate value of `tan(pi/2 - 0.0001)` is 10000
    Mins(absTensor, absTensor, static_cast<yType>(APPROXIMATE_VALUE), calCount);
    PipeBarrier<PIPE_V>();

    // calucate data less than one
    // 1.x
    TaylorExpand(dstTensor, absTensor, squareTensor, TAYLOR_COUNT_FOUR, calCount);

    // 2.(x-tan(pi/8)) / (1 + x*tan(pi/8))
    AtanTransform(tmpTensor, absTensor, tmpTensor2, constData.CONST_TAN_PI_BY_EIGHT, calCount); // tan(pi/8)
    TaylorExpand(tmpTensor2, tmpTensor, squareTensor, TAYLOR_COUNT_FOUR, calCount);
    PipeBarrier<PIPE_V>();
    // pi/8 + atan((x-tan(pi/8)) / (1 + x*tan(pi/8)))
    Adds(tmpTensor2, tmpTensor2, constData.CONST_PI_BY_EIGHT, calCount);
    PipeBarrier<PIPE_V>();
    Min(dstTensor, dstTensor, tmpTensor2, calCount);

    // calucate data more than one
    // (x-1)/(x+1)
    Adds(tmpTensor2, absTensor, static_cast<yType>(1.0), calCount);
    // x=x-y
    Adds(tmpTensor, absTensor, -static_cast<yType>(1.0), calCount);
    PipeBarrier<PIPE_V>();
    // (x-y)/(1+xy)
    Div(tmpTensor, tmpTensor, tmpTensor2, calCount);
    PipeBarrier<PIPE_V>();
    Abs(tmpTensor, tmpTensor, calCount); // take the absolute value
    PipeBarrier<PIPE_V>();

    // 3. atan((x-1)/(x+1))
    TaylorExpand(tmpTensor2, tmpTensor, squareTensor, TAYLOR_COUNT_FOUR, calCount);
    PipeBarrier<PIPE_V>();
    // pi/4 + atan((x-1)/(x+1))
    Adds(tmpTensor2, tmpTensor2, constData.CONST_PI_BY_FOUR, calCount);
    PipeBarrier<PIPE_V>();
    Min(dstTensor, dstTensor, tmpTensor2, calCount);

    // 4.reuse the Transform result in step 3, and calculate (x-tan(pi/8)) / (1 + x*tan(pi/8))
    AtanTransform(tmpTensor2, tmpTensor, squareTensor, constData.CONST_TAN_PI_BY_EIGHT, calCount); // tan(pi/8)
    TaylorExpand(tmpTensor, tmpTensor2, squareTensor, TAYLOR_COUNT_SIX, calCount);
    PipeBarrier<PIPE_V>();
    // pi/8 + atan((x-tan(pi/8)) / (1 + x*tan(pi/8)))
    Adds(tmpTensor, tmpTensor, constData.CONST_PI_BY_EIGHT, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tmpTensor, tmpTensor, constData.CONST_PI_BY_FOUR, calCount);
    PipeBarrier<PIPE_V>();
    Min(dstTensor, dstTensor, tmpTensor, calCount);
    PipeBarrier<PIPE_V>();
}

  __aicore__ inline void AtanCompute(LocalTensor<yType> &dstTensor, LocalTensor<yType> &srcTensor,
                                     LocalTensor<yType> &tmpTensor, uint32_t calCount) {
    AtanImpl(dstTensor, srcTensor, tmpTensor, calCount);

    // get sign of src
    Sign(tmpTensor, srcTensor, tempTensor2, calCount);

    // dst = sign(x) * dst.
    Mul(dstTensor, dstTensor, tmpTensor, calCount);
    PipeBarrier<PIPE_V>();
}

 private:
  TPipe pipe;
  ConstData constData;
  uint8_t repeatTimes;
  uint32_t complexTailLength = DATA_PER_REPEAT_B32;

  const float factorList[7] = {1, NEG_ONE_BY_THREE, ONE_BY_FIVE, NEG_ONE_BY_SEVEN, ONE_BY_NINE,
                               NEG_ONE_BY_ELEVEN, ONE_BY_THIRTEEN};
  GlobalTensor<yType> xGm;
  GlobalTensor<yType> yGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TBuf<TPosition::VECCALC> maskBuf1;
  TBuf<TPosition::VECCALC> tempBuf1;
  TBuf<TPosition::VECCALC> tempBuf2;
  TBuf<TPosition::VECCALC> tempBuf3;
  TBuf<TPosition::VECCALC> maskBuf2;
  TBuf<TPosition::VECCALC> tempBuf4;
  TBuf<TPosition::VECCALC> tempBuf5;
  TBuf<TPosition::VECCALC> oneBuf;
  TBuf<TPosition::VECCALC> zeroBuf;
  TBuf<TPosition::VECCALC> realBuf;
  TBuf<TPosition::VECCALC> imagBuf;

  LocalTensor<yType> realLocal;
  LocalTensor<yType> imagLocal;
  LocalTensor<yType> zeroTensor;
  LocalTensor<yType> oneTensor;
  LocalTensor<yType> tempTensor1;
  LocalTensor<yType> tempTensor2;
  LocalTensor<yType> tempTensor3;
  LocalTensor<yType> tempTensor4;
  LocalTensor<yType> tempTensor5;
  LocalTensor<uint8_t> mask1;
  LocalTensor<uint8_t> mask2;
};
} // AngleV2N
#endif  // _ANGLE_V2_COMPLEX_H_
