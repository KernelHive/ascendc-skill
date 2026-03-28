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
 * \file dequant_swiglu_quant.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_H
#define DEQUANT_SWIGLU_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DequantSwigluQuantOps {
using namespace AscendC;
constexpr static int64_t DB_BUFFER = 1;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t MASK_NUM_T32 = 256 / sizeof(float);
constexpr static int64_t MASK_BLK_STRIDE = 8;
constexpr static int64_t SWI_FACTOR = 2;
constexpr static float DYNAMIC_QUANT_FACTOR = 1.0 / 127.0;

template <typename T, typename TQuantScale, typename TGroup>
class DequantSwigluQuantBase {
 public:
  static constexpr bool hasGroupIndex_ = !IsSameType<TGroup, float>::value;
  __aicore__ inline DequantSwigluQuantBase(TPipe* pipe) {
    pipe_ = pipe;
  };

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale,
                              GM_ADDR quantOffset, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                              const DequantSwigluQuantBaseTilingData* tilingData);

  __aicore__ inline void Process();

 private:
  __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& tempRes, int32_t calCount);
  __aicore__ inline void ProcessSingleGroup(int64_t groupIdx);

 protected:
  /* global memory address */
  // input global mem
  GlobalTensor<int32_t> xGm_;
  GlobalTensor<float> weightScaleGm_;
  GlobalTensor<float> activationScaleGm_;
  GlobalTensor<float> biasGm_;
  GlobalTensor<TQuantScale> quantScaleGm_;
  GlobalTensor<float> quantOffsetGm_;
  GlobalTensor<TGroup> groupIndexGm_;

  // output global mem
  GlobalTensor<int8_t> yGm_;
  GlobalTensor<float> scaleGm_;

  /* ascendc variable */
  TPipe* pipe_ = nullptr;
  TQue<QuePosition::VECIN, DB_BUFFER> xActQueue_;
  TQue<QuePosition::VECIN, 1> inScaleQueue_;

  TQue<QuePosition::VECOUT, 1> outQueue_;

  TBuf<TPosition::VECCALC> tmpBuf1_;

  uint32_t blockIdx_ = GetBlockIdx();
  uint32_t realCoreDim_ = 0;
  int64_t realDimx_ = 0;
  int64_t groupOffset_ = 0;

  const DequantSwigluQuantBaseTilingData* tl_ = nullptr;
};
// 公共函数实现

template <typename T, typename TQuantScale, typename TGroup>
__aicore__ inline void DequantSwigluQuantBase<T, TQuantScale, TGroup>::Init(
    GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, const DequantSwigluQuantBaseTilingData* tilingData) {
  tl_ = tilingData;
  xGm_.SetGlobalBuffer((__gm__ int32_t*)x);
  weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
  activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale);
  quantScaleGm_.SetGlobalBuffer((__gm__ TQuantScale*)quantScale);
  if constexpr (hasGroupIndex_) {
    groupIndexGm_.SetGlobalBuffer((__gm__ TGroup*)groupIndex);
  }
  yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
  scaleGm_.SetGlobalBuffer((__gm__ float*)scale);

  // init buffer
  pipe_->InitBuffer(xActQueue_, DB_BUFFER,
                    (tl_->UbFactorDimx * tl_->outDimy * SWI_FACTOR + tl_->UbFactorDimx * BLOCK_ELEM) * sizeof(int32_t));
  pipe_->InitBuffer(inScaleQueue_, 1, (tl_->outDimy * SWI_FACTOR + tl_->outDimy) * sizeof(float));

  pipe_->InitBuffer(outQueue_, 1,
                    tl_->UbFactorDimx * tl_->outDimy * sizeof(int8_t) + tl_->UbFactorDimx * sizeof(float) + BLOCK_SIZE);

  pipe_->InitBuffer(tmpBuf1_, tl_->UbFactorDimx * tl_->outDimy * SWI_FACTOR * sizeof(float));
}

template <typename T, typename TQuantScale, typename TGroup>
__aicore__ inline void DequantSwigluQuantBase<T, TQuantScale, TGroup>::Process() {
  if constexpr (!hasGroupIndex_) {
    realDimx_ = tl_->inDimx;
    // do protect realDimx_ < 0, ignore this group
    realDimx_ = (realDimx_ < 0) ? 0 : realDimx_;
    ProcessSingleGroup(0);
    return;
  }

  groupOffset_ = 0;
  for (int32_t groupIdx = 0; groupIdx < tl_->inGroupNum; ++groupIdx) {
    realDimx_ = static_cast<int64_t>(groupIndexGm_(groupIdx));
    // do protect realDimx_ < 0, ignore this group
    realDimx_ = (realDimx_ < 0) ? 0 : realDimx_;
    if (realDimx_ > 0 && groupOffset_ < tl_->inDimx) {
      ProcessSingleGroup(groupIdx);
      groupOffset_ += realDimx_;
    }
  }
}

template <typename T, typename TQuantScale, typename TGroup>
__aicore__ inline void DequantSwigluQuantBase<T, TQuantScale, TGroup>::ProcessSingleGroup(int64_t groupIdx) {
  // do block tiling again
  int32_t blockDimxFactor = (realDimx_ + tl_->maxCoreNum - 1) / tl_->maxCoreNum;
  realCoreDim_ = (realDimx_ + blockDimxFactor - 1) / blockDimxFactor;
  if (blockIdx_ < realCoreDim_) {
    DataCopyPadParams padParams{false, 0, 0, 0};
    // copy weight scale [1, 2H] offset:0
    LocalTensor<float> inScaleLocal = inScaleQueue_.AllocTensor<float>();
    DataCopyParams dataCopyWeightScaleParams;
    dataCopyWeightScaleParams.blockCount = 1;
    dataCopyWeightScaleParams.blockLen = tl_->inDimy * sizeof(float);
    dataCopyWeightScaleParams.srcStride = 0;
    dataCopyWeightScaleParams.dstStride = 0;
    DataCopyPad(inScaleLocal, weightScaleGm_[groupIdx * tl_->inDimy], dataCopyWeightScaleParams, padParams);

    // copy quant scale [1, H] offset:tl_->inDimy
    DataCopyParams dataCopyQuantScaleParams;
    dataCopyQuantScaleParams.blockCount = 1;
    dataCopyQuantScaleParams.blockLen = tl_->outDimy * sizeof(TQuantScale);
    dataCopyQuantScaleParams.srcStride = 0;
    dataCopyQuantScaleParams.dstStride = 0;
    if constexpr (std::is_same_v<TQuantScale, float>) {
      DataCopyPad(inScaleLocal[tl_->inDimy], quantScaleGm_[groupIdx * tl_->outDimy], dataCopyQuantScaleParams,
                  padParams);
    } else {
      LocalTensor<TQuantScale> quantScaleLocalT16 = inScaleLocal.template ReinterpretCast<TQuantScale>();
      DataCopyPad(quantScaleLocalT16[SWI_FACTOR * tl_->inDimy + tl_->outDimy], quantScaleGm_[groupIdx * tl_->outDimy],
                  dataCopyQuantScaleParams, padParams);
    }
    inScaleQueue_.EnQue(inScaleLocal);
    inScaleLocal = inScaleQueue_.DeQue<float>();

    int32_t blockDimxTailFactor = realDimx_ - blockDimxFactor * (realCoreDim_ - 1);
    int32_t DimxCore = blockIdx_ == (realCoreDim_ - 1) ? blockDimxTailFactor : blockDimxFactor;
    // do ub tiling again
    int32_t ubDimxLoop = (DimxCore + tl_->UbFactorDimx - 1) / tl_->UbFactorDimx;
    int32_t ubDimxTailFactor = DimxCore - tl_->UbFactorDimx * (ubDimxLoop - 1);

    int64_t coreDimxOffset = blockDimxFactor * blockIdx_;
    int32_t actOffset = tl_->actRight * tl_->UbFactorDimy;
    int32_t gateOffset = tl_->UbFactorDimy - actOffset;

    LocalTensor<float> weightScaleLocal = inScaleLocal;
    LocalTensor<float> quantScaleLocal = inScaleLocal[tl_->inDimy];
    if constexpr (std::is_same_v<TQuantScale, half>) {
      LocalTensor<TQuantScale> quantScaleLocalT16 = inScaleLocal.template ReinterpretCast<TQuantScale>();
      Cast(quantScaleLocal, quantScaleLocalT16[SWI_FACTOR * tl_->inDimy + tl_->outDimy], RoundMode::CAST_NONE,
           tl_->outDimy);
    }
    for (uint32_t loopIdx = 0; loopIdx < ubDimxLoop; ++loopIdx) {
      int64_t xDimxOffset = (coreDimxOffset + loopIdx * tl_->UbFactorDimx) + groupOffset_;
      // copy in x: [x, y]
      int32_t proDimsx = loopIdx == (ubDimxLoop - 1) ? ubDimxTailFactor : tl_->UbFactorDimx;
      LocalTensor<int32_t> xActLocal = xActQueue_.AllocTensor<int32_t>();
      DataCopyParams dataCopyXParams;
      dataCopyXParams.blockCount = proDimsx;
      dataCopyXParams.blockLen = tl_->inDimy * sizeof(int32_t);
      dataCopyXParams.srcStride = 0;
      dataCopyXParams.dstStride = 0;
      DataCopyPad(xActLocal, xGm_[xDimxOffset * tl_->inDimy], dataCopyXParams, padParams);

      // copy act scale: [proDimsx,8] offset:tl_->UbFactorDimx * tl_->inDimy
      DataCopyParams dataCopyActScaleParams;
      dataCopyActScaleParams.blockCount = proDimsx;
      dataCopyActScaleParams.blockLen = sizeof(float);
      dataCopyActScaleParams.srcStride = 0;
      dataCopyActScaleParams.dstStride = 0;
      LocalTensor<float> xActLocalF32 = xActLocal.template ReinterpretCast<float>();
      DataCopyPad(xActLocalF32[tl_->UbFactorDimx * tl_->inDimy], activationScaleGm_[xDimxOffset],
                  dataCopyActScaleParams, padParams);
      xActQueue_.EnQue(xActLocal);
      xActLocal = xActQueue_.DeQue<int32_t>();
      // do int32 to fp32 cast
      // Cast from int32_t to float16
      LocalTensor<int32_t> xLocal = xActLocal;
      xActLocalF32 = xActLocal.template ReinterpretCast<float>();
      LocalTensor<float> xLocalF32 = xActLocalF32;
      LocalTensor<float> activationScaleLocal = xActLocalF32[tl_->UbFactorDimx * tl_->inDimy];
      Cast(xLocalF32, xLocal, RoundMode::CAST_NONE, SWI_FACTOR * proDimsx * tl_->UbFactorDimy);

      LocalTensor<float> tmpUbF32 = tmpBuf1_.AllocTensor<float>();
      // Copy weight scale: [1,2H] -> [proDimsx,2H]
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy * SWI_FACTOR);
      // params: dstStride: 1, srcStride: 1, dstRepStride: tl_->UbFactorDimy * 2 / 8, srcRepStride: 0
      Copy<float, false>(tmpUbF32, weightScaleLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
                         {1, 1, static_cast<uint16_t>((tl_->UbFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 0});
      SetMaskNorm();
      ResetMask();
      PipeBarrier<PIPE_V>();
      // Calc dequant: xLocalF32 = weightScaleLocal * xLocalF32
      Mul(xLocalF32, tmpUbF32, xLocalF32, tl_->UbFactorDimy * SWI_FACTOR * proDimsx);
      PipeBarrier<PIPE_V>();
      // Copy act scale: [proDimsx,8] -> [proDimsx,2H]
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy * SWI_FACTOR);
      Copy<float, false>(tmpUbF32, activationScaleLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
                         {1, 0, static_cast<uint16_t>((tl_->UbFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 1});
      SetMaskNorm();
      ResetMask();
      PipeBarrier<PIPE_V>();
      // Calc dequant: xLocalF32 = activationScaleLocal * xLocalF32
      Mul(xLocalF32, tmpUbF32, xLocalF32, tl_->UbFactorDimy * SWI_FACTOR * proDimsx);
      PipeBarrier<PIPE_V>();
      // do swi pre
      LocalTensor<float> tmpUbF32Act = tmpUbF32;
      LocalTensor<float> tmpUbF32Gate = tmpUbF32[tl_->UbFactorDimy * proDimsx];
      // Copy dequant result: xLocalF32[actOffset] -> tmpUbF32Act, [proDimsx,H]
      // Copy dequant result: xLocalF32[gateOffset] -> tmpUbF32Gate, [proDimsx,H]
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
      Copy<float, false>(tmpUbF32Act, xLocalF32[actOffset], AscendC::MASK_PLACEHOLDER, proDimsx,
                         {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM),
                          static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM * SWI_FACTOR)});
      Copy<float, false>(tmpUbF32Gate, xLocalF32[gateOffset], AscendC::MASK_PLACEHOLDER, proDimsx,
                         {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM),
                          static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM * SWI_FACTOR)});
      SetMaskNorm();
      ResetMask();
      PipeBarrier<PIPE_V>();
      Muls(xLocalF32, tmpUbF32Act, static_cast<float>(-1.0), tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();
      Exp(xLocalF32, xLocalF32, tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();
      Adds(xLocalF32, xLocalF32, static_cast<float>(1.0), tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();
      Div(tmpUbF32Act, tmpUbF32Act, xLocalF32, tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();
      // x compute done, free
      xActQueue_.FreeTensor(xActLocal);
      Mul(tmpUbF32Act, tmpUbF32Gate, tmpUbF32Act, tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();

      // Copy quant scale: [1,H] -> [proDimsx,H]
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
      Copy<float, false>(tmpUbF32Gate, quantScaleLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
                         {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM), 0});
      SetMaskNorm();
      ResetMask();
      PipeBarrier<PIPE_V>();
      // Calc quant: xLocalF32 = tmpUbF32Act * quantScaleLocal
      Mul(tmpUbF32Act, tmpUbF32Gate, tmpUbF32Act, tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();
      // Calc quant: tmpUbF32Gate = abs(tmpUbF32Act)
      Abs(tmpUbF32Gate, tmpUbF32Act, tl_->UbFactorDimy * proDimsx);

      LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
      LocalTensor<float> scaleOut = outLocal[tl_->UbFactorDimx * tl_->outDimy * sizeof(int8_t) / sizeof(float)];
      LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();
      PipeBarrier<PIPE_V>();
      // Calc quant: proDimsx * tl_->UbFactorDimy -> proDimsx * 64
      for (uint32_t i = 0; i < proDimsx; ++i) {
        ComputeReduceMax(tmpUbF32Gate[i * tl_->UbFactorDimy], tl_->UbFactorDimy);
      }
      // Calc quant: proDimsx * 64 -> proDimsx
      // repeatTimes:proDimsx, dstRepStride:1(dtype), srcBlkStride:1, srcRepStride:tl_->UbFactorDimy / 64 * 8
      WholeReduceMax(tmpUbF32Gate, tmpUbF32Gate, MASK_NUM_T32, proDimsx, 1, 1,
                     tl_->UbFactorDimy / MASK_NUM_T32 * MASK_BLK_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
      PipeBarrier<PIPE_V>();
      // Calc quant: scaleOut / 127.0
      Muls(scaleOut, tmpUbF32Gate, DYNAMIC_QUANT_FACTOR, proDimsx);
      PipeBarrier<PIPE_V>();
      // Calc Broadcast: proDimsx -> proDimsx,8
      int64_t blockCount = (proDimsx + BLOCK_ELEM - 1) / BLOCK_ELEM;
      Brcb(outLocal, scaleOut, blockCount, {1, MASK_BLK_STRIDE});
      PipeBarrier<PIPE_V>();
      // Copy scale: [proDimsx,8] -> [proDimsx,H]
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
      Copy<float, false>(tmpUbF32Gate, outLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
                         {1, 0, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM), 1});
      SetMaskNorm();
      ResetMask();
      PipeBarrier<PIPE_V>();
      // Calc y: tmpUbF32Act = tmpUbF32Act / scaleOut
      Div(tmpUbF32Act, tmpUbF32Act, tmpUbF32Gate, tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();

      LocalTensor<int32_t> tmpUbF32ActI32 = tmpUbF32Act.ReinterpretCast<int32_t>();
      Cast(tmpUbF32ActI32, tmpUbF32Act, RoundMode::CAST_RINT, tl_->UbFactorDimy * proDimsx);
      SetDeqScale((half)1.000000e+00f);

      LocalTensor<half> tmpUbF32Gate16 = tmpUbF32Gate.template ReinterpretCast<half>();
      Cast(tmpUbF32Gate16, tmpUbF32ActI32, RoundMode::CAST_ROUND, tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();

      Cast(yOut, tmpUbF32Gate16, RoundMode::CAST_TRUNC, tl_->UbFactorDimy * proDimsx);
      PipeBarrier<PIPE_V>();
      tmpBuf1_.FreeTensor(tmpUbF32);
      outQueue_.EnQue<float>(outLocal);
      // copy out
      outLocal = outQueue_.DeQue<float>();
      scaleOut = outLocal[tl_->UbFactorDimx * tl_->outDimy * sizeof(int8_t) / sizeof(float)];
      yOut = outLocal.template ReinterpretCast<int8_t>();
      DataCopyParams dataCopyOutScaleParams;
      dataCopyOutScaleParams.blockCount = 1;
      dataCopyOutScaleParams.blockLen = proDimsx * sizeof(float);
      dataCopyOutScaleParams.srcStride = 0;
      dataCopyOutScaleParams.dstStride = 0;
      DataCopyPad(scaleGm_[xDimxOffset], scaleOut, dataCopyOutScaleParams);

      DataCopyParams dataCopyOutyParams;
      dataCopyOutyParams.blockCount = 1;
      dataCopyOutyParams.blockLen = proDimsx * tl_->outDimy * sizeof(int8_t);
      dataCopyOutyParams.srcStride = 0;
      dataCopyOutyParams.dstStride = 0;
      DataCopyPad(yGm_[xDimxOffset * tl_->outDimy], yOut, dataCopyOutyParams);
      outQueue_.FreeTensor(outLocal);
    }
    inScaleQueue_.FreeTensor(inScaleLocal);
  }
}

template <typename T, typename TQuantScale, typename TGroup>
__aicore__ inline void DequantSwigluQuantBase<T, TQuantScale, TGroup>::ComputeReduceMax(
    const LocalTensor<float>& tempRes, int32_t calCount) {
  uint32_t vectorCycles = calCount / MASK_NUM_T32;
  uint32_t remainElements = calCount % MASK_NUM_T32;

  BinaryRepeatParams repeatParams;
  repeatParams.dstBlkStride = 1;
  repeatParams.src0BlkStride = 1;
  repeatParams.src1BlkStride = 1;
  repeatParams.dstRepStride = 0;
  repeatParams.src0RepStride = MASK_BLK_STRIDE;
  repeatParams.src1RepStride = 0;

  if (vectorCycles > 0 && remainElements > 0) {
    Max(tempRes, tempRes, tempRes[vectorCycles * MASK_NUM_T32], remainElements, 1, repeatParams);
    PipeBarrier<PIPE_V>();
  }

  if (vectorCycles > 1) {
    Max(tempRes, tempRes[MASK_NUM_T32], tempRes, MASK_NUM_T32, vectorCycles - 1, repeatParams);
    PipeBarrier<PIPE_V>();
  }
}
}  // namespace DequantSwigluQuantOps
#endif  // DEQUANT_SWIGLU_QUANT_H
