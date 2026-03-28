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
 * @file pre_layer_norm.cpp
 */
#include "kernel_operator.h"

constexpr float epsilon = 1e-5f;
constexpr int32_t BUFFER_NUM = 1;

class KernelPreLayerNorm {
 public:
  __aicore__ inline KernelPreLayerNorm() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR gamma, GM_ADDR beta,
                              GM_ADDR z, uint32_t lastDim, uint32_t tileNum) {
    this->lastDim = lastDim;
    int32_t initLength = 0;
    if (this->lastDim == 2048) {
      this->rowNum = tileNum;
      this->rowLength = lastDim;
      gemmaGm.SetGlobalBuffer((__gm__ float *)gamma);
      betaGm.SetGlobalBuffer((__gm__ float *)beta);
      xGm.SetGlobalBuffer(
          (__gm__ float *)x + this->rowNum * this->rowLength * AscendC::GetBlockIdx(),
          this->rowNum * this->rowLength);
      yGm.SetGlobalBuffer(
          (__gm__ float *)y + this->rowNum * this->rowLength * AscendC::GetBlockIdx(),
          this->rowNum * this->rowLength);
      zGm.SetGlobalBuffer(
          (__gm__ float *)z + this->rowNum * this->rowLength * AscendC::GetBlockIdx(),
          this->rowNum * this->rowLength);
      initLength = this->rowLength;
    } else {
      this->rowNum = 512 * 4;
      this->rowLength = lastDim;
      this->tileLength = 2560;
      this->rowPerTile = 1;
      this->tilePerRow = this->rowLength / this->tileLength;
      this->blockOffset = AscendC::GetBlockIdx() * this->rowLength;
      this->blockStride = AscendC::GetBlockNum() * this->rowLength;
      int32_t totalTileNum = this->rowNum / this->rowPerTile;
      this->tileNum = totalTileNum / AscendC::GetBlockNum();
      int32_t leftTileNum = totalTileNum - AscendC::GetBlockNum() * this->tileNum;
      if (AscendC::GetBlockIdx() < leftTileNum) {
        this->tileNum += 1;
      }
      this->rowNum = this->tileNum * this->rowPerTile;
      if (AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 &&
          this->rowNum % this->rowPerTile != 0) {
        this->rowNum =
            this->rowNum - this->rowPerTile + (this->rowNum % this->rowPerTile);
      }
      xGm.SetGlobalBuffer((__gm__ float *)x, this->rowNum * this->rowLength);
      yGm.SetGlobalBuffer((__gm__ float *)y, this->rowNum * this->rowLength);
      zGm.SetGlobalBuffer((__gm__ float *)z, this->rowNum * this->rowLength);
      gemmaGm.SetGlobalBuffer((__gm__ float *)gamma, this->rowLength);
      betaGm.SetGlobalBuffer((__gm__ float *)beta, this->rowLength);
      initLength = this->tileLength;
    }
    pipe.InitBuffer(queueX, BUFFER_NUM, initLength * sizeof(float));
    pipe.InitBuffer(queueY, BUFFER_NUM, initLength * sizeof(float));
    pipe.InitBuffer(queueZ, BUFFER_NUM, initLength * sizeof(float));

    pipe.InitBuffer(queueGamma, BUFFER_NUM, initLength * sizeof(float));
    pipe.InitBuffer(queueBeta, BUFFER_NUM, initLength * sizeof(float));
    pipe.InitBuffer(tempBuf1, initLength * sizeof(float));
    pipe.InitBuffer(tempBuf2, initLength * sizeof(float));
    pipe.InitBuffer(tempBuf3, initLength * sizeof(float));
    pipe.InitBuffer(tempBuf4, initLength * sizeof(float));
  }
  __aicore__ inline void Process() {
    if (this->lastDim == 2048) {
      for (int32_t i = 0; i < this->rowNum; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
      }
    } else {
      int32_t baseOffset = this->blockOffset;
      for (int32_t i = 0; i < this->rowNum; i++) {
        ProcessTileInRow(baseOffset);
        baseOffset += this->blockStride;
      }
    }
  }

 private:
  __aicore__ inline void ProcessTileInRow(int32_t baseOffset) {
    AscendC::LocalTensor<float> mean = tempBuf1.Get<float>();
    AscendC::Duplicate(mean, 0.0f, this->tileLength);
    AscendC::LocalTensor<float> variance = tempBuf2.Get<float>();
    AscendC::Duplicate(variance, 0.0f, this->tilePerRow * 8);

    for (int32_t tileIdx = 0; tileIdx < this->tilePerRow; ++tileIdx) {
      int32_t inputOffset = baseOffset + tileIdx * this->tileLength;
      int32_t copyLength = this->tileLength;
      CopyInInput(inputOffset, copyLength);
      AscendC::LocalTensor<float> xLocal = queueX.DeQue<float>();
      ComputeAdd(xLocal);
      int32_t tileOffset = tileIdx * 8;

      AscendC::LocalTensor<float> curMean = mean[tileOffset];
      AscendC::LocalTensor<float> curVariance = variance[tileOffset];
      ComputeMeanVar(curMean, curVariance, xLocal, this->tileLength);
      queueX.FreeTensor(xLocal);
    }
    MergeM2(mean, variance, this->tileLength, this->tilePerRow);
    ComputeDenominator(variance, 1);
    for (int32_t tileIdx = 0; tileIdx < this->tilePerRow; ++tileIdx) {
      int32_t inputOffset = baseOffset + tileIdx * this->tileLength;
      int32_t copyLength = this->tileLength;
      CopyInInput(inputOffset, copyLength);
      AscendC::LocalTensor<float> xLocal = queueX.DeQue<float>();
      ComputeAdd(xLocal);
      AscendC::LocalTensor<float> zLocal = queueZ.AllocTensor<float>();
      CopyInBeta(tileIdx * this->tileLength, this->tileLength);
      AscendC::LocalTensor<float> gammaLocal = queueGamma.DeQue<float>();
      AscendC::LocalTensor<float> betaLocal = queueBeta.DeQue<float>();
      ComputeLayerNorm(zLocal, xLocal, mean, variance, gammaLocal, betaLocal,
                       this->tileLength);

      queueX.FreeTensor(xLocal);
      queueGamma.FreeTensor(gammaLocal);
      queueBeta.FreeTensor(betaLocal);
      queueZ.EnQue<float>(zLocal);
      CopyOut(inputOffset, copyLength);
    }
  }

  __aicore__ inline void ComputeAdd(AscendC::LocalTensor<float> &x) {
    AscendC::LocalTensor<float> yLocal = queueY.DeQue<float>();
    AscendC::Add(x, x, yLocal, this->tileLength);
    queueY.FreeTensor(yLocal);
  }

  __aicore__ inline void ComputeMeanVar(AscendC::LocalTensor<float> &mean,
                                        AscendC::LocalTensor<float> &variance,
                                        AscendC::LocalTensor<float> &x, int32_t length) {
    AscendC::LocalTensor<float> workspace = tempBuf3.Get<float>();
    AscendC::LocalTensor<float> workspace1 = tempBuf4.Get<float>();
    AscendC::ReduceSum<float>(mean, x, workspace, length);
    float meanValue = 0;
    if (length != 0){
      meanValue = mean.GetValue(0) / length;
    }
    mean.SetValue(0, meanValue);
    float negativeMean = -meanValue;
    AscendC::Adds(workspace, x, negativeMean, length);
    AscendC::Mul(workspace, workspace, workspace, length);
    AscendC::Muls(workspace, workspace, 1.0f / (float)length, length);
    AscendC::ReduceSum<float>(variance, workspace, workspace1, length);
  }

  __aicore__ inline void MergeM2(AscendC::LocalTensor<float> &mean,
                                 AscendC::LocalTensor<float> &variance,
                                 int32_t mergeUnitSize, int32_t mergeNum) {
    auto curMergeSize = mergeUnitSize;
    auto remainMergeNum = mergeNum;
    AscendC::LocalTensor<float> delta = tempBuf3.Get<float>();
    while (remainMergeNum > 1) {
      remainMergeNum = remainMergeNum / 2;
      int32_t stride = remainMergeNum * 8;
      AscendC::Sub(delta, mean[0], mean[stride], stride);
      AscendC::Add(mean, mean[0], mean[stride], stride);
      AscendC::Muls(mean, mean, 0.5f, stride);
      AscendC::Mul(delta, delta, delta, stride);
      AscendC::Muls(delta, delta, 0.25f, stride);
      AscendC::Muls(variance, variance, 0.5f, stride * 2);
      AscendC::Add(variance, variance, variance[stride], stride);
      AscendC::Add(variance, variance, delta, stride);
      curMergeSize = curMergeSize * 2;
    }
  }

  __aicore__ inline void ComputeDenominator(AscendC::LocalTensor<float> &variance,
                                            int32_t length) {
    AscendC::Adds(variance, variance, epsilon, length);
    AscendC::Ln(variance, variance, length);
    AscendC::Muls(variance, variance, -0.5f, length);
    AscendC::Exp(variance, variance, length);
  }

  __aicore__ inline void ComputeLayerNorm(
      AscendC::LocalTensor<float> &z, AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &mean,
      AscendC::LocalTensor<float> &denominator, AscendC::LocalTensor<float> &gamma,
      AscendC::LocalTensor<float> &beta, int32_t length) {
    float negativeMean = -mean.GetValue(0);
    AscendC::Adds(x, x, negativeMean, length);
    AscendC::Muls(z, x, denominator.GetValue(0), length);
    AscendC::Mul(z, z, gamma, length);
    AscendC::Add(z, z, beta, length);
  }

  __aicore__ inline void CopyInInput(int32_t offset, int32_t length) {
    AscendC::LocalTensor<float> xLocal = queueX.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = queueY.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[offset], length);
    AscendC::DataCopy(yLocal, yGm[offset], length);
    queueX.EnQue(xLocal);
    queueY.EnQue(yLocal);
  }

  __aicore__ inline void CopyInBeta(int32_t offset, int32_t length) {
    AscendC::LocalTensor<float> gammaLocal = queueGamma.AllocTensor<float>();
    AscendC::LocalTensor<float> betaLocal = queueBeta.AllocTensor<float>();
    AscendC::DataCopy(gammaLocal, gemmaGm[offset], length);
    AscendC::DataCopy(betaLocal, betaGm[offset], length);
    queueGamma.EnQue(gammaLocal);
    queueBeta.EnQue(betaLocal);
  }

  __aicore__ inline void CopyOut(int32_t offset, int32_t length) {
    AscendC::LocalTensor<float> zLocal = queueZ.DeQue<float>();
    AscendC::DataCopy(zGm[offset], zLocal, length);
    queueZ.FreeTensor(zLocal);
  }

  __aicore__ inline void CopyIn(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = queueX.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = queueY.AllocTensor<float>();
    AscendC::LocalTensor<float> gammaLocal = queueGamma.AllocTensor<float>();
    AscendC::LocalTensor<float> betaLocal = queueBeta.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[progress * this->rowLength], this->rowLength);
    AscendC::DataCopy(yLocal, yGm[progress * this->rowLength], this->rowLength);
    AscendC::DataCopy(gammaLocal, gemmaGm, this->rowLength);
    AscendC::DataCopy(betaLocal, betaGm, this->rowLength);
    queueX.EnQue(xLocal);
    queueY.EnQue(yLocal);
    queueGamma.EnQue(gammaLocal);
    queueBeta.EnQue(betaLocal);
  }

  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = queueX.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = queueY.DeQue<float>();
    AscendC::LocalTensor<float> gammaLocal = queueGamma.DeQue<float>();
    AscendC::LocalTensor<float> betaLocal = queueBeta.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = queueZ.AllocTensor<float>();

    AscendC::LocalTensor<float> tempOneLocal = tempBuf1.Get<float>();
    AscendC::LocalTensor<float> tempTwoLocal = tempBuf2.Get<float>();
    AscendC::LocalTensor<float> tempThreeLocal = tempBuf3.Get<float>();
    AscendC::Add(xLocal, xLocal, yLocal, this->rowLength);
    AscendC::ReduceSum<float>(tempOneLocal, xLocal, tempTwoLocal, this->rowLength);
    AscendC::Muls(tempOneLocal, tempOneLocal, 1.0f / this->rowLength, 1);

    float average = tempOneLocal.GetValue(0) * -1.0f;

    AscendC::Adds(xLocal, xLocal, average, this->rowLength);
    tempOneLocal = xLocal * xLocal;
    AscendC::ReduceSum<float>(tempTwoLocal, tempOneLocal, tempThreeLocal,
                     this->rowLength);

    AscendC::Muls(tempTwoLocal, tempTwoLocal, 1.0f / this->rowLength, 1);
    AscendC::Adds(tempTwoLocal, tempTwoLocal, epsilon, 1);
    AscendC::Sqrt(tempTwoLocal, tempTwoLocal, 1);
    tempOneLocal = xLocal * gammaLocal;

    float temp = 1.0f / tempTwoLocal.GetValue(0);

    AscendC::Muls(tempOneLocal, tempOneLocal, temp, this->rowLength);
    zLocal = tempOneLocal + betaLocal;
    queueZ.EnQue<float>(zLocal);
    queueGamma.FreeTensor(gammaLocal);
    queueBeta.FreeTensor(betaLocal);

    queueY.FreeTensor(yLocal);
    queueX.FreeTensor(xLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<float> zLocal = queueZ.DeQue<float>();
    AscendC::DataCopy(zGm[progress * this->rowLength], zLocal, this->rowLength);
    queueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> queueX, queueY, queueGamma, queueBeta;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> queueZ;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> yGm;
  AscendC::GlobalTensor<float> gemmaGm;
  AscendC::GlobalTensor<float> betaGm;
  AscendC::GlobalTensor<float> zGm;
  AscendC::TBuf<> tempBuf1;
  AscendC::TBuf<> tempBuf2;
  AscendC::TBuf<> tempBuf3;
  AscendC::TBuf<> tempBuf4;
  uint32_t lastDim = 2048;
  uint32_t rowNum = 415;
  uint32_t rowLength = 2048;
  uint32_t tileLength;
  uint32_t tilePerRow;
  uint32_t blockOffset;
  uint32_t blockStride;
  uint32_t tileNum;
  uint32_t rowPerTile;
};

extern "C" __global__ __aicore__ void pre_layer_norm(
    GM_ADDR x, GM_ADDR y, GM_ADDR gamma, GM_ADDR beta, GM_ADDR res_out,
    GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);

  KernelPreLayerNorm op;
  op.Init(x, y, gamma, beta, res_out, tiling_data.lastDim,
          tiling_data.tileNum);
  if (TILING_KEY_IS(1)) {
    op.Process();
  }
}