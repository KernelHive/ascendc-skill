/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_with_large_head_dim_s1s2_bn2gs1.h
 * \brief
 */

 #ifndef FLASH_ATTENTION_SCORE_S1S2_BN2GS1_H
 #define FLASH_ATTENTION_SCORE_S1S2_BN2GS1_H
 
 /**
  * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
  * This file is a part of the CANN Open Software.
  * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */
 
 /*!
  * \file flash_attention_score_with_large_head_dim_common.h
  * \brief
  */
 
  #ifndef FLASH_ATTENTION_SCORE_COMMON_H
  #define FLASH_ATTENTION_SCORE_COMMON_H
  
  #include "kernel_operator.h"
  #include "kernel_tiling/kernel_tiling.h"
  #include "lib/matmul_intf.h"
  #include "lib/matrix/matmul/tiling.h"
  #include "stdarg.h"
  
  using AscendC::LocalTensor;
  using AscendC::GlobalTensor;
  using AscendC::DataFormat;
  using AscendC::ShapeInfo;
  using AscendC::DataCopyParams;
  using AscendC::DataCopyPadParams;
  using AscendC::HardEvent;
  using AscendC::SetFlag;
  using AscendC::WaitFlag;
  using AscendC::BinaryRepeatParams;
  using AscendC::Cast;
  using AscendC::Div;
  using AscendC::Duplicate;
  using AscendC::GetBlockIdx;
  using AscendC::RoundMode;
  using AscendC::SelectWithBytesMask;
  using AscendC::SelectWithBytesMaskShapeInfo;
  using AscendC::SoftmaxFlashV2;
  using AscendC::SoftMaxShapeInfo;
  using AscendC::TBuf;
  using AscendC::TPipe;
  using AscendC::TPosition;
  
  constexpr MatmulConfig CFG_EXCEED = GetNormalConfig(true);
  constexpr static uint64_t BLOCK_BYTE = 32;
  constexpr static int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
  constexpr static int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
  constexpr int32_t blockBytes = 32;
  constexpr static int32_t blockSize = blockBytes / 4; // 4 means sizeof(T)
  constexpr static int32_t repeatMaxBytes = 256;
  constexpr static int32_t repeatMaxSize = repeatMaxBytes / 4; // 4 means sizeof(T)
  
  // 0级接口的block间隔范围需要满足32B对齐
  constexpr static int32_t fp32BaseSize = 8;
  
  namespace math {
  template <typename T> __aicore__ inline T Ceil(T a, T b)
  {
      if (b == 0) {
          return 0;
      }
      return (a + b - 1) / b;
  }
  
  template <typename T> __aicore__ inline T Align(T a, T b)
  {
      if (b == 0) {
          return 0;
      }
      return (a + b - 1) / b * b;
  }
  }
  
  template <typename T1, typename T2>
  __aicore__ inline T1 CeilDiv(T1 a, T2 b)
  {
      if (b == 0) {
          return 0;
      }
      return (a + b - 1) / b;
  }
  
  template <typename T1, typename T2>
  __aicore__ inline T1 Min(T1 a, T2 b)
  {
      return (a > b) ? (b) : (a);
  }
  
  __aicore__ inline int32_t Align(int32_t shape)
  {
      int32_t alignFactor = 16;
      int32_t alignedSize = CeilDiv(shape, alignFactor) * alignFactor;
      return alignedSize;
  }
  
  __aicore__ inline bool IsBasicBlockInSoftMax(int32_t srcM, int32_t srcK)
  {
      return srcM % SOFTMAX_M_ALIGNED_SIZE == 0 && srcK % SOFTMAX_K_ALIGNED_SIZE == 0;
  }
  
  template <typename T>
  __aicore__ inline void DataCopy2D(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal, const uint32_t d0,
                                    const uint32_t d1, const uint32_t orgD1, uint64_t paddingValue = 0)
  {
      if (d1 % (BLOCK_BYTE / sizeof(T)) == 0 && orgD1 % (BLOCK_BYTE / sizeof(T)) == 0) {
          auto d1Blocks = math::Ceil(d1 * sizeof(T), BLOCK_BYTE);
          auto orgD1Blocks = math::Ceil(orgD1 * sizeof(T), BLOCK_BYTE);
          DataCopyParams copyParams(d0, d1Blocks, orgD1Blocks - d1Blocks, 0);
          DataCopy(dstLocal, srcGlobal, copyParams);
      } else {
          auto d1Bytes = d1 * sizeof(T);
          auto d1Aligned = math::Align(static_cast<int64_t>(d1), static_cast<int64_t>(BLOCK_BYTE / sizeof(T)));
          DataCopyParams copyParams(static_cast<uint16_t>(d0), static_cast<uint16_t>(d1Bytes),
                                    orgD1 * sizeof(T) - d1Bytes, 0);
          DataCopyPadParams padParams(true, 0, static_cast<uint8_t>(d1Aligned - d1), paddingValue);
          DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
      }
  }
  
  #endif // FLASH_ATTENTION_SCORE_COMMON_H
  
  
 #include "kernel_operator.h"
 #include "kernel_tiling/kernel_tiling.h"
 #include "lib/matmul_intf.h"
 
 using matmul::MatmulType;
 
 struct SplitExtraInfo {
     int64_t s2StartIdx;
     int64_t s2EndIdx;
     int64_t s2LoopCount;
     int64_t s1oIdx;
     int64_t boIdx;
     int64_t n2oIdx;
     int64_t goIdx;
     int64_t taskId;
     int8_t taskIdMod2;
     int8_t multiCoreInnerIdxMod2;
     bool lastNotPair;
     int32_t s1RealSize;
     int32_t s2RealSize;
     int32_t s2AlignedSize;
     int32_t vec1S1BaseSize;
     int32_t vec1S1RealSize;
     int32_t vec2S1BaseSize;
     int32_t vec2S1RealSize;
     int32_t realSplitN;
     int32_t s2LoopLimit;
     int64_t multiCoreInnerIdx;
     int64_t qCoreOffset;
     int64_t s1Size;
     int64_t s2Size;
     int64_t softmaxMaxOffset;
 };
 
 constexpr int64_t GM_DOUBLE_BUFFER = 2;
 constexpr int64_t INVALID_OFFSET = INT64_MIN;
 constexpr AscendC::SoftmaxConfig SOFTMAX_DEFAULT_CFG = {false};
 
 __aicore__ const constexpr MatmulConfig &GetMmCfg()
 {
     return CFG_EXCEED;
 }
 
 class FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1 {
 public:
     __aicore__ inline FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1(){};
 
     __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                 __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                 __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                 const FlashAttentionScoreWithLargeHeadDimTilingData *__restrict tiling, TPipe *tPipe);
     __aicore__ inline void Process();
 
     // define matmul
     using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, half>;
     using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, half, true, LayoutMode::NONE, false>;
     using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
     using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
     matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, GetMmCfg()> bmm1;
     // define batchmatmul
     using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, half>;
     using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NONE, false>;
     using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
     using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
     using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, float>;
     using modeTypemm2 = typename AscendC::Conditional<
           false,
           matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, GetMmCfg()>,
           matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, GetMmCfg()>>::type;
 
     modeTypemm2 bmm2;
 
 protected:
     __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                      __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut,
                                      __gm__ uint8_t *workspace, const FlashAttentionScoreWithLargeHeadDimTilingData *__restrict tiling, TPipe *tPipe);
     __aicore__ inline void WaitBmm1Result(SplitExtraInfo &extraInfo);
     __aicore__ inline void WaitBmm2Result();
     __aicore__ inline void IterateBmm2(SplitExtraInfo &extraInfo);
     __aicore__ inline void SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                         int64_t s2LoopLimit, int64_t multiCoreInnerIdx, bool lastNotPair);
     __aicore__ inline void InitBuffer();
     __aicore__ inline void ComputeConstexpr();
     __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
     template <typename T2, const MatmulConfig &MM_CFG>
     __aicore__ inline void IterateBmm1(SplitExtraInfo &extraInfo,
                                        matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
     template <typename T2, const MatmulConfig &MM_CFG>
     __aicore__ inline void Bmm1SetTensorA(SplitExtraInfo &extraInfo,
                                           matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
     template <typename T2, const MatmulConfig &MM_CFG>
     __aicore__ inline void SetBmm1TensorB(SplitExtraInfo &extraInfo,
                                           matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
     __aicore__ inline void ComputeBmm1Tail(SplitExtraInfo &extraInfo);
     __aicore__ inline void ProcessVec1(SplitExtraInfo &extraInfo);
     __aicore__ inline void GetBmm1Result(SplitExtraInfo &extraInfo, LocalTensor<float> &bmm1ResUb, int64_t loopIdx);
     __aicore__ inline void SoftMaxCompute(SplitExtraInfo &extraInfo, LocalTensor<float> &srcTensor, int64_t loopIdx);
     __aicore__ inline void ProcessVec2(SplitExtraInfo &extraInfo);
     __aicore__ inline void Bmm2ResultMul(SplitExtraInfo &extraInfo, LocalTensor<float> &bmm2ResUb, int64_t s1oIdx);
     __aicore__ inline void Bmm2ResultDiv(SplitExtraInfo &extraInfo, int64_t s1oIdx);
     __aicore__ inline void Bmm2DataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx, int64_t mm2ResCalcSize);
     __aicore__ inline void SoftmaxDataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx);
 
     uint32_t s1BaseSize;
     uint32_t s2BaseSize;
     uint32_t dSize;
     int64_t dSizeAlign16;
     int64_t s1Size;
     int64_t s2Size;
     int64_t s1OuterSize;
 
     // sparse 用参�?
     int64_t s2StartIdx;
     int64_t s2EndIdx;
     int64_t nextS2EndIdx;
 
     // s2方向的尾块，包含N:1配比
     int64_t bmm2LastS2RealSize = INVALID_OFFSET;
     int64_t qCoreOffset;
 
     // 资源分配
     // TBuf<> maskTBufPing;
     // TBuf<> maskTBufPong;
     TBuf<> pseTBuf;
     TBuf<> stage1PingBuf;
     TBuf<> stage1PongBuf;
     TBuf<> stage2TBuf;
     TBuf<> softmaxSumBuf[2];
     TBuf<> softmaxExpBuf[2];
     TBuf<> softmaxMaxBuf;
     TBuf<> commonTBuf;
     GlobalTensor<float> mm1Res[2];
     GlobalTensor<float> mm2Res[2];
     GlobalTensor<float> vec2Res[2];
     GlobalTensor<half> stage1Res[2];
 
     // 轴的乘积
     int64_t gS1o;
     int64_t n2GS1o;
     int64_t s1D;
     int64_t gS1D;
     int64_t n2GS1D;
     int64_t s2D;
     int64_t n2S2D;
     int64_t s1S2;
     int64_t gS1S2;
     int64_t gS1;
     int64_t n2GS1;
     int64_t gD;
     int64_t n2D;
     int64_t n2G;
     int64_t n2GD;
     int64_t gS2;
 
     // s2base*N之后的长度
     // cube计算的s2长度
     int64_t s2BaseNratioSize;
 
     int64_t s2BaseN2D;
     int64_t s1BaseN2GD;
     int64_t s2BaseNratioN2D;
     int64_t bN2G;
     int64_t mm1Ka;
     int64_t mm1Kb;
     int64_t mm2Kb;
     int32_t blockIdx;
     const FlashAttentionScoreWithLargeHeadDimTilingData *__restrict tilingData;
 
     int64_t boIdx;
     int64_t n2oIdx;
     int64_t goIdx;
     int64_t s1oIdx;
 
     TPipe *pipe;
 
     GlobalTensor<half> queryGm;
     GlobalTensor<half> keyGm;
     GlobalTensor<half> valueGm;
     GlobalTensor<half> attentionOutGm;
     GlobalTensor<float> softmaxMaxGm;
     GlobalTensor<float> softmaxSumGm;
 };
 
 __aicore__ inline void
 FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FlashAttentionScoreWithLargeHeadDimTilingData *__restrict tiling,
                                     TPipe *tPipe)
 {
     this->InitInput(query, key, value, softmaxMax, softmaxSum, attentionOut, workspace, tiling, tPipe); // gm设置
 
     this->ComputeConstexpr();
     this->InitBuffer();
     LocalTensor<float> apiTmpBuffer = this->commonTBuf.template Get<float>();
 }
 
 __aicore__ inline void
 FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                          __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                          __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                          const FlashAttentionScoreWithLargeHeadDimTilingData *__restrict tiling,
                                          TPipe *tPipe)
 {
     this->blockIdx = GetBlockIdx();
     this->pipe = tPipe;
     // copy base params
     this->tilingData = tiling;
     this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
     this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
     this->dSize = this->tilingData->inputParams.dSize;
     this->dSizeAlign16 = CeilDiv(this->tilingData->inputParams.dSize, 16) * 16;
 
     // init global buffer
     this->queryGm.SetGlobalBuffer((__gm__ half *)query);
     this->keyGm.SetGlobalBuffer((__gm__ half *)key);
     this->valueGm.SetGlobalBuffer((__gm__ half *)value);
     this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
     this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
     this->attentionOutGm.SetGlobalBuffer((__gm__ half *)attentionOut);
 
     int64_t mm1ResultSize = s1BaseSize * s2BaseSize;
     int64_t mmNRatioOffset = CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(float);
     int64_t mm2ResultSize = s1BaseSize * dSizeAlign16;
     int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * 4;
     int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
     int64_t vector1OffsetPing = 0;
     int64_t vector1OffsetPong = mmNRatioOffset;
 
     // 每个核在gm上都占用这么大的workspace
     int64_t totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * GM_DOUBLE_BUFFER;
     if (dSizeAlign16 > 64) {
         totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;
     }
 
     // workspace上找到当前core要使用的地址空间
     this->mm1Res[0].SetGlobalBuffer((__gm__ float *)(workspace + this->blockIdx * totalOffset));
     this->mm1Res[1].SetGlobalBuffer((__gm__ float *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset));
     // vec1阶段输出复用cube1输出bmm1Result的地址空间
     this->stage1Res[0].SetGlobalBuffer(
         (__gm__ half *)(workspace + this->blockIdx * totalOffset + vector1OffsetPing));
     this->stage1Res[1].SetGlobalBuffer(
         (__gm__ half *)(workspace + this->blockIdx * totalOffset + vector1OffsetPong));
 
     // bmm2Result
     this->mm2Res[0].SetGlobalBuffer(
         (__gm__ float *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio));
     this->mm2Res[1].SetGlobalBuffer(
         (__gm__ float *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset));
 
     // vec2阶段，D轴>64时，占用2倍mmOffset空间
     if (dSizeAlign16 > 64) {
         this->vec2Res[0].SetGlobalBuffer(
             (__gm__ float *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2));
         this->vec2Res[1].SetGlobalBuffer(
             (__gm__ float *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 3));
     }
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::InitBuffer()
 {
     uint64_t stage1Size = 8 * 1024;
     // uint64_t stage1AttenSize = 9 * 1024;
     uint64_t stage1PongSize = 35 * 1024;
     uint64_t stage2Size = 64 * 128;
     // uint64_t maskTBufPongSize = 16 * 1024;
 
     // 可选输入的buffer空间，保持和stage1处理的size一致
     // this->pipe->InitBuffer(this->maskTBufPing, stage1AttenSize); // 可以给attenmask 9k
     // this->pipe->InitBuffer(this->maskTBufPong, maskTBufPongSize); // 可以给dropoutmask 16k
     this->pipe->InitBuffer(this->pseTBuf, 16384); // pse 16k
 
     this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(float)); // t.a 32k
     this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(float));    // t.c 32k
     this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(float));    // t.b 32k
 
     this->pipe->InitBuffer(this->softmaxSumBuf[0], s1BaseSize * blockBytes); // 4k
     this->pipe->InitBuffer(this->softmaxSumBuf[1], s1BaseSize * blockBytes); // 4k
     this->pipe->InitBuffer(this->softmaxMaxBuf, s1BaseSize * blockBytes);    // 4k
     this->pipe->InitBuffer(this->softmaxExpBuf[0], s1BaseSize * blockBytes); // 4k
     this->pipe->InitBuffer(this->softmaxExpBuf[1], s1BaseSize * blockBytes); // 4k
     this->pipe->InitBuffer(this->stage1PongBuf, stage1Size * sizeof(float)); // i.a 32k
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::ComputeConstexpr()
 {
     // 计算轴的乘积
     this->s1D = this->tilingData->inputParams.s1Size * dSize;
     this->s2D = this->tilingData->inputParams.s2Size * dSize;
     this->gD = this->tilingData->inputParams.gSize * dSize;
     this->n2D = this->tilingData->inputParams.n2Size * dSize;
     this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
     this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
     this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
     this->gS1o = this->tilingData->inputParams.gSize * this->tilingData->coreParams.s1OuterSize;
 
     this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;
     this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
     this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
     this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
     this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
     this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;
     this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
 
     // 计算切分轴的乘积
     this->s2BaseN2D = this->s2BaseSize * this->n2D;
     this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;
     this->s1BaseN2GD = this->s1BaseSize * this->n2GD;
     this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;
     this->mm1Ka = this->n2GD;
     this->mm1Kb = this->n2D;
     this->mm2Kb = this->n2D;
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::Process()
 {
     // 确定核内切分起点
     int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
     int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
     if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
         multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
     }
     // 计算sparse场景下s1的循环范�?
     SplitExtraInfo extraInfo[3];
     int64_t taskId = 0;
     event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
 
     bool notSecondLast = true;
     bool notLast = true;
     multiCoreInnerLimit += 2;
     for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
             multiCoreInnerIdx++) {
         if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
             notSecondLast = false;
         } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
             notLast = false;
         }
 
         int64_t s2LoopLimit;
         bool notLastTwoLoop = notSecondLast && notLast;
         if (notLastTwoLoop) {
             this->ComputeAxisIdx(multiCoreInnerIdx);            
             this->s2StartIdx = 0;
             this->s2EndIdx = this->s2Size;
             s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, s2BaseNratioSize) - 1;
         } else {
             s2LoopLimit = 0;
         }
         for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
             if (taskId >= 1 && notLast) {
                 // 对应extraInfo[(i+2)%3]
                 WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
             }
 
             if (notLastTwoLoop) {
                 this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx,
                                     false);
                 this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
             }
 
             if (taskId > 0 && notLast) {
                 this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                 SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
             }
 
             if (taskId > 1) {
                 // 对应extraInfo[(i+1)%3]
                 WaitBmm2Result();
             }
 
             if (taskId > 0 && notLast) {
                 WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                 this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
             }
 
             if (taskId > 1) {
                 this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
             }
             taskId++;
         }
     }
 };
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::ComputeAxisIdx(int64_t multiCoreInnerIdx)
 {
     // 计算轴的idx
     this->boIdx = multiCoreInnerIdx / this->n2GS1o;
     this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
     this->goIdx = multiCoreInnerIdx % this->gS1o / this->tilingData->coreParams.s1OuterSize;
     this->s1oIdx = multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
     this->s1Size = this->tilingData->inputParams.s1Size;
     this->s2Size = this->tilingData->inputParams.s2Size;
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::WaitBmm1Result(SplitExtraInfo &extraInfo)
 {
     this->bmm1.WaitIterateAll();
     this->bmm1.End();
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId,
                                                                     int64_t s2LoopCount, int64_t s2LoopLimit,
                                                                     int64_t multiCoreInnerIdx, bool lastNotPair)
 {
     extraInfo.s2StartIdx = this->s2StartIdx;
     extraInfo.s2EndIdx = this->s2EndIdx;
     extraInfo.s2LoopCount = s2LoopCount;
     extraInfo.s1oIdx = this->s1oIdx;
     extraInfo.boIdx = this->boIdx;
     extraInfo.n2oIdx = this->n2oIdx;
     extraInfo.goIdx = this->goIdx;
     extraInfo.taskId = taskId;
     extraInfo.taskIdMod2 = taskId % 2;
     extraInfo.s2LoopLimit = s2LoopLimit;
     extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
     extraInfo.multiCoreInnerIdxMod2 = multiCoreInnerIdx % 2;
     extraInfo.s1Size = this->tilingData->inputParams.s1Size;
     extraInfo.s2Size = this->tilingData->inputParams.s2Size;
     extraInfo.s1RealSize = Min(s1BaseSize, this->tilingData->inputParams.s1Size - extraInfo.s1oIdx * s1BaseSize);
     this->ComputeBmm1Tail(extraInfo);
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::ComputeBmm1Tail(SplitExtraInfo &extraInfo)
 {
     if (this->tilingData->inputParams.s1Size < (extraInfo.s1oIdx + 1) * this->s1BaseSize) {
         extraInfo.s1RealSize = this->tilingData->inputParams.s1Size - extraInfo.s1oIdx * this->s1BaseSize;
     }
     extraInfo.s2RealSize = this->s2BaseNratioSize;
     extraInfo.s2AlignedSize = extraInfo.s2RealSize;
     if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
         extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
         extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
     }
 
     extraInfo.vec1S1BaseSize = Min(s2BaseNratioSize / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize);
     extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);
 
     if (dSizeAlign16 > 64) {
         extraInfo.vec2S1BaseSize = 64 * 128 / dSizeAlign16;
     } else {
         extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
     }
     return;
 }
 
 template <typename T2, const MatmulConfig &MM_CFG>
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::IterateBmm1(SplitExtraInfo &extraInfo,
                                                                     matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1)
 {
     bmm1.SetOrgShape(extraInfo.s1RealSize, this->mm1Kb, this->mm1Ka, this->mm1Kb, extraInfo.s2RealSize);
 
     this->Bmm1SetTensorA(extraInfo, bmm1);
     this->SetBmm1TensorB(extraInfo, bmm1);
     bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], 0, false, true);
 }
 
 template <typename T2, const MatmulConfig &MM_CFG>
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::Bmm1SetTensorA(SplitExtraInfo &extraInfo,
                                                                         matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1)
 {
     // 计算gm上的offset
     int64_t bOffset = extraInfo.boIdx * this->n2GS1D;
     // s1需要考虑inner轴的影响
     int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
     int64_t n2Offset = extraInfo.n2oIdx * this->gD;
     int64_t gOffset = extraInfo.goIdx * dSize;
     extraInfo.qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
     bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);
 }
 
 template <typename T2, const MatmulConfig &MM_CFG>
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::SetBmm1TensorB(SplitExtraInfo &extraInfo,
                                                                         matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG>
                                                                         &bmm1)
 {
     // 计算gm上的offset
     int64_t bOffset = extraInfo.boIdx * this->n2S2D;
     int64_t n2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
     int64_t s2Offset = extraInfo.n2oIdx * dSize;
     int64_t kCoreOffset = bOffset + n2Offset + s2Offset;
     bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
     bmm1.SetTail(extraInfo.s1RealSize, extraInfo.s2RealSize);
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::ProcessVec1(SplitExtraInfo &extraInfo)
 {
     LocalTensor<float> stage1PingTensor = this->stage1PingBuf.template Get<float>(); // t.a 32k
     LocalTensor<float> stage1PongTensor = this->stage1PongBuf.template Get<float>(); // i.a 32k
     LocalTensor<float> commonTBuf = this->commonTBuf.template Get<float>(); // t.b 32k
 
     event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
     event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
     event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
     event_t eventIdVToMte2C = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
     event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
     event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
     extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
     for (int32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; loopIdx++) {
         if (loopIdx == extraInfo.realSplitN - 1) {
             extraInfo.vec1S1RealSize = extraInfo.s1RealSize - loopIdx * extraInfo.vec1S1BaseSize;
         }
         if (loopIdx > 0) {
             WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
         }
         this->GetBmm1Result(extraInfo, stage1PongTensor, loopIdx);
 
         // mul需要等bmm结果搬完
         SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
         WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
         pipe_barrier(PIPE_V);
         Muls(stage1PingTensor, stage1PongTensor, static_cast<float>(this->tilingData->inputParams.scaleValue),
         extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
         if (loopIdx < extraInfo.realSplitN - 1) {
             SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
         }
 
         if (loopIdx > 0) {
             WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
         }
 
         this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);
         if (loopIdx < extraInfo.realSplitN - 1) {
             SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
         }
 
         if (loopIdx > 0) {
             WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
         }
         pipe_barrier(PIPE_V);
         LocalTensor<half> stage1CastTensor;
         stage1CastTensor = this->pseTBuf.template Get<half>();
         Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                 extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
         SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
         WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
 
         DataCopy(
             this->stage1Res[extraInfo.taskIdMod2][loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2AlignedSize],
             stage1CastTensor, extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
         if (loopIdx < extraInfo.realSplitN - 1) {
             SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
         }
     }
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2C);
     return;
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::GetBmm1Result(SplitExtraInfo &extraInfo, LocalTensor<float> &bmm1ResUb,
                                                                     int64_t loopIdx)
 {
     if (likely(extraInfo.s2AlignedSize == extraInfo.s2RealSize)) {
         DataCopy2D(bmm1ResUb,
                    this->mm1Res[extraInfo.taskIdMod2][loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                    extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);
 
     } else {
         DataCopyParams dataCopyParams;
         dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
         dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(float);
         dataCopyParams.srcStride = 0;
         dataCopyParams.dstStride = 0;
         DataCopyPadParams dataCopyPadParams;
         dataCopyPadParams.isPad = true;
         dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
         if (dataCopyPadParams.rightPadding > blockSize) {
             dataCopyPadParams.rightPadding -= blockSize;
             dataCopyParams.dstStride = 1;
             int32_t s2BlockAlignedSize = CeilDiv(extraInfo.s2RealSize, blockSize) * blockSize;
             Duplicate<float>(bmm1ResUb[s2BlockAlignedSize], 0, blockSize, extraInfo.vec1S1RealSize, 0,
                          extraInfo.s2AlignedSize * sizeof(float) / blockBytes);
         }
         dataCopyPadParams.paddingValue = 0;
         DataCopyPad(bmm1ResUb,
                     this->mm1Res[extraInfo.taskIdMod2][loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                     dataCopyParams, dataCopyPadParams);
     }
     uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                  static_cast<uint32_t>(extraInfo.s2AlignedSize)};
     bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::SoftMaxCompute(SplitExtraInfo &extraInfo, LocalTensor<float> &srcTensor,
                                                                         int64_t loopIdx)
 {
     uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                  static_cast<uint32_t>(extraInfo.s2AlignedSize)};
     uint32_t bmm1ResUbOrgShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                     static_cast<uint32_t>(extraInfo.s2RealSize)};
     srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOrgShape, DataFormat::ND));
 
     uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(fp32BaseSize)};
     LocalTensor<float> sumUb = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2]
                 .template Get<float>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];
     LocalTensor<float> maxUb = this->softmaxMaxBuf.template Get<float>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];
 
     sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
     maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
     LocalTensor<float> expUb = this->softmaxExpBuf[extraInfo.taskIdMod2]
                 .template Get<float>()[loopIdx * extraInfo.vec1S1BaseSize * blockBytes / sizeof(float)];
 
     expUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
     LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
     pipe_barrier(PIPE_V);
     if (unlikely(extraInfo.s2LoopCount == 0)) {
         if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
             SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                             extraInfo.s2AlignedSize, sizeof(float),
                                                                             sizeof(float),
                                                                             apiTmpBuffer.GetSize() / sizeof(float),
                                                                             false, true);
             SoftmaxFlashV2<float, false, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                              sumUb, maxUb, apiTmpBuffer, newTiling);
         } else {
             SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                             extraInfo.s2AlignedSize, sizeof(float),
                                                                             sizeof(float),
                                                                             apiTmpBuffer.GetSize() / sizeof(float),
                                                                             false, false);
             SoftmaxFlashV2<float, false, true, false, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                              sumUb, maxUb, apiTmpBuffer, newTiling);
         }
     } else {
         if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
             SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                             extraInfo.s2AlignedSize, sizeof(float),
                                                                             sizeof(float),
                                                                             apiTmpBuffer.GetSize() / sizeof(float),
                                                                             true, true);
             SoftmaxFlashV2<float, true, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                              sumUb, maxUb, apiTmpBuffer, newTiling);
         } else {
             SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                             extraInfo.s2AlignedSize, sizeof(float),
                                                                             sizeof(float),
                                                                             apiTmpBuffer.GetSize() / sizeof(float),
                                                                             true, false);
             SoftmaxFlashV2<float, true, true, false, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                              sumUb, maxUb, apiTmpBuffer, newTiling);
         }
     }
     if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
         extraInfo.softmaxMaxOffset =
             (extraInfo.boIdx * extraInfo.s1Size * this->n2G +
              extraInfo.n2oIdx * this->tilingData->inputParams.gSize * extraInfo.s1Size +
              extraInfo.goIdx * extraInfo.s1Size + extraInfo.s1oIdx * static_cast<int64_t>(s1BaseSize)) *
             static_cast<int64_t>(fp32BaseSize);
         int64_t calculateSize = extraInfo.s1RealSize * fp32BaseSize;
         LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();
         event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
         SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
         WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
         DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset], maxTensor, calculateSize);
     }
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::WaitBmm2Result()
 {
     this->bmm2.WaitIterateAll();
     this->bmm2.End();
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::IterateBmm2(SplitExtraInfo &extraInfo)
 {
     int64_t bOffset = 0;
     int64_t n2Offset = 0;
     int64_t s2Offset = 0;
 
     // BSH/BSND
     bOffset = extraInfo.boIdx * this->n2S2D;
     s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * s2BaseNratioSize * this->n2D;
     n2Offset = extraInfo.n2oIdx * dSize;
     int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
     if (extraInfo.s2AlignedSize != bmm2LastS2RealSize) {
         this->bmm2.SetOrgShape(extraInfo.s1Size, this->mm2Kb, extraInfo.s2AlignedSize, this->mm2Kb, this->dSize);
         bmm2LastS2RealSize = extraInfo.s2AlignedSize;
     }
 
     this->bmm2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2]);
 
     this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
     this->bmm2.SetTail(extraInfo.s1RealSize, this->dSize, extraInfo.s2RealSize);
     this->bmm2.template IterateAll<false>(this->mm2Res[extraInfo.taskIdMod2], false, false, true);
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::ProcessVec2(SplitExtraInfo &extraInfo)
 {
     // 获取缓存bmm2的计算结�?
     LocalTensor<float> bmm2ResUb = this->stage2TBuf.template Get<float>();
     LocalTensor<float> stage2BufTensor = this->commonTBuf.template Get<float>();
     int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);
 
     event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
     event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
     event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
     event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
     extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;
     for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; s1oIdx++) {
         if (s1oIdx == vec2LoopLimit - 1) {
             extraInfo.vec2S1RealSize = extraInfo.s1RealSize - s1oIdx * extraInfo.vec2S1BaseSize;
         }
         int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * dSize;
         int64_t mm2ResOffset = s1oIdx * extraInfo.vec2S1BaseSize * dSize;
         SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
         WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
         int64_t dAlign8 = (this->dSize + 7) / 8 * 8;
         if (likely(this->dSizeAlign16 == this->dSize)) {
             DataCopy(stage2BufTensor, this->mm2Res[extraInfo.taskIdMod2][mm2ResOffset], mm2ResCalcSize);
         } else {
             DataCopyParams dataCopyParams;
             DataCopyPadParams dataCopyPadParams;
             dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
             dataCopyParams.dstStride = 0;
             dataCopyParams.srcStride = 0;
             dataCopyParams.blockLen = this->dSize * 4;
             dataCopyPadParams.rightPadding = this->dSizeAlign16 - this->dSize;
             dataCopyPadParams.paddingValue = 0;
             if (dataCopyPadParams.rightPadding > blockSize) {
                 // 8对齐场景，内部vector需�?6对齐，我们在data copy的时候需要手动补0
                 dataCopyPadParams.rightPadding -= blockSize;
                 dataCopyParams.dstStride = 1;
                 Duplicate<float>(stage2BufTensor[dAlign8], 0, blockSize, extraInfo.vec2S1RealSize, 0,
                                 this->dSizeAlign16 * sizeof(float) / blockBytes);
             }
             DataCopyPad(stage2BufTensor, this->mm2Res[extraInfo.taskIdMod2][mm2ResOffset], dataCopyParams,
                         dataCopyPadParams);
             mm2ResCalcSize = extraInfo.vec2S1RealSize * dSizeAlign16;
             mm2ResOffset = s1oIdx * extraInfo.vec2S1BaseSize * dSizeAlign16;
         }
 
         if (vec2LoopLimit > 1) {
             SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
             WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
             DataCopy(bmm2ResUb, this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], mm2ResCalcSize);
         }
 
         SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
         WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
         if (unlikely(extraInfo.s2LoopCount == 0)) {
             DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
         } else {
             this->Bmm2ResultMul(extraInfo, bmm2ResUb, s1oIdx);
             pipe_barrier(PIPE_V);
             Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
         }
 
         if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
             Bmm2ResultDiv(extraInfo, s1oIdx);
             Bmm2DataCopyOut(extraInfo, s1oIdx, mm2ResCalcSize);
             SoftmaxDataCopyOut(extraInfo, s1oIdx);
             event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
             SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
             WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
         } else if (vec2LoopLimit > 1) {
             SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
             WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
             SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
             WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
             DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
             SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
             WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
         }
     }
     return;
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::Bmm2ResultMul(SplitExtraInfo &extraInfo, LocalTensor<float> &bmm2ResUb,
                                                             int64_t s1oIdx)
 {
     pipe_barrier(PIPE_V);
     LocalTensor<float> expUb;
     expUb = softmaxExpBuf[extraInfo.taskIdMod2].Get<float>();
 
     BinaryRepeatParams repeatParams;
     repeatParams.src0BlkStride = 0;
     repeatParams.src0RepStride = 1;
     repeatParams.src1RepStride = dSizeAlign16 / blockSize;
     repeatParams.dstRepStride = dSizeAlign16 / blockSize;
 
     // s1长度可能会超�?55限制，修改成双重循环
     // 根据一次最多计算的byte数量，对bmm2Res分组mul
     int32_t loop = dSizeAlign16 / repeatMaxSize;
     int32_t remain = dSizeAlign16 % repeatMaxSize;
     for (int i = 0; i < loop; i++) {
         Mul(bmm2ResUb[i * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], bmm2ResUb[i * repeatMaxSize],
             repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
     }
     if (likely(remain)) {
         Mul(bmm2ResUb[loop * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8],
             bmm2ResUb[loop * repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
     }
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::Bmm2ResultDiv(SplitExtraInfo &extraInfo, int64_t s1oIdx)
 {
     LocalTensor<float> bmm2ResUb = this->stage2TBuf.template Get<float>();
 
     BinaryRepeatParams repeatParams;
     repeatParams.src0BlkStride = 1;
     repeatParams.src0RepStride = dSizeAlign16 / blockSize;
     repeatParams.src1BlkStride = 0;
     repeatParams.src1RepStride = 1;
     repeatParams.dstRepStride = dSizeAlign16 / blockSize;
     int32_t loop = dSizeAlign16 / repeatMaxSize;
     int32_t remain = dSizeAlign16 % repeatMaxSize;
 
     LocalTensor<float> sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].Get<float>();
 
     int32_t calcSize = sumUb.GetSize();
     // 用optionalInputQueue的queue
     pipe_barrier(PIPE_V);
     for (int i = 0; i < loop; i++) {
         Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize],
             sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
     }
     if (likely(remain)) {
         Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
             sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
     }
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::Bmm2DataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx,
                                                               int64_t mm2ResCalcSize)
 {
     LocalTensor<float> bmm2ResUb = this->stage2TBuf.template Get<float>();
     LocalTensor<half> attenOut = this->stage2TBuf.template Get<half>();
     bmm2ResUb.SetSize(mm2ResCalcSize);
     pipe_barrier(PIPE_V);
     Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, mm2ResCalcSize);
 
     event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
     SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
     WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
 
     DataCopyParams dataCopyParams;
     dataCopyParams.blockLen = this->dSize * sizeof(half);
     dataCopyParams.srcStride = 0;
     int64_t dstStride = 0;
     int64_t attenOutOffset = this->dSize;
     int64_t datacopyOffset = this->dSize;
 
     datacopyOffset = this->n2GD;
     attenOutOffset = this->n2GD;
     dstStride = (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->dSize *
                 sizeof(half);
     if (likely(dstStride <= 65535)) {
         dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
         dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
         DataCopyPad(this->attentionOutGm[extraInfo.qCoreOffset + s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset],
                     attenOut, dataCopyParams);
     } else {
         dataCopyParams.blockCount = 1;
         dataCopyParams.dstStride = 0;
 
         for (int32_t i = 0; i < extraInfo.vec2S1RealSize; i++) {
             DataCopyPad(this->attentionOutGm[extraInfo.qCoreOffset +
                                              s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset + i * datacopyOffset],
                         attenOut[i * this->dSizeAlign16], dataCopyParams);
         }
     }
 }
 
 __aicore__ inline void FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::SoftmaxDataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx)
 {
     int64_t vec2S1Offset = s1oIdx * extraInfo.vec2S1BaseSize * fp32BaseSize;
     LocalTensor<float> sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>()[vec2S1Offset];
     DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset + vec2S1Offset], sumTensor,
              extraInfo.vec2S1RealSize * fp32BaseSize);
 }
 
 #endif // FLASH_ATTENTION_SCORE_S1S2_BN2GS1_H
 
 
using namespace AscendC;

extern "C" __global__ __aicore__ void flash_attention_score_with_large_head_dim(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_out, GM_ADDR workspace, GM_ADDR tiling) {
    
    TPipe tPipe;
    set_mask_norm();    
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreWithLargeHeadDimTilingData, tilingDataIn, tiling);
    const FlashAttentionScoreWithLargeHeadDimTilingData *__restrict tilingData = &tilingDataIn;
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);
    FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1 op;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);
    op.Init(query, key, value, softmax_max, softmax_sum, attention_out, user, tilingData, &tPipe);
    op.Process();
}