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
 * \file upsample_bicubic2d_grad_base.h
 * \brief
 */
#ifndef _ASCENDC_UPSAMPLE_BICUBIC2D_GRAD_BASE_H_
#define _ASCENDC_UPSAMPLE_BICUBIC2D_GRAD_BASE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_SIZE = 160 * 1024;
constexpr uint32_t NUM_FRACTAL = 16;
constexpr uint32_t NUM_PER_BLOCK_FLOAT16 = 16;
constexpr uint32_t NUM_PER_BLOCK_FLOAT32 = 8;
constexpr uint32_t SPECIAL_INIT_VAL = 210715;

template <typename T>
class UpsampleBicubic2dGradBase {
 public:
  TPipe pipe;

  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
      MMH;

  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
      MMW;
  __aicore__ inline UpsampleBicubic2dGradBase() = default;
  __aicore__ inline void Process();
  __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR usrWorkspace, const UpsampleBicubic2dGradTilingData* __restrict tiling_data);
 protected:
  __aicore__ inline uint32_t GetNumPerBlock();
  __aicore__ inline void InitMMW();
  __aicore__ inline void InitEventId();
  __aicore__ inline void InitGlobalTensors(GM_ADDR grad_output, GM_ADDR grad_input);
  __aicore__ inline void InitWorkspaceTensors(GM_ADDR usrWorkspace);
  __aicore__ inline void InitLocalTensors();
  __aicore__ inline void computeCoeff(int32_t offset, float scale, uint32_t scaleN, int32_t idx[16]);
  __aicore__ inline void fillAndCastCoeffW(int32_t offset, int32_t base[2], int32_t idx[16]);
  __aicore__ inline void fillAndCastCoeffH(int32_t offset, int32_t base[2], int32_t idx[16]);
  __aicore__ inline void ProcessW();
  __aicore__ inline void ProcessH();
  __aicore__ inline void ClearGM(const GlobalTensor<T> &dstGlobal, uint32_t loop,
                                uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum);
  const UpsampleBicubic2dGradTilingData* __restrict tilingData;

  uint32_t block_id;
  uint32_t block_h;
  uint32_t block_inner_h;
  uint32_t block_w;
  uint32_t block_inner_w;

  GlobalTensor<T> inGm;
  GlobalTensor<T> outGm;
  GlobalTensor<T> interGm;
  GlobalTensor<T> coeffW;
  GlobalTensor<T> coeffH;
  GlobalTensor<float> coeffWFloat;
  GlobalTensor<float> coeffHFloat;
  
  event_t eventIdMTE3ToS;
  event_t eventIdVToS;
  event_t eventIdVToMTE3;
  event_t eventIdSToV;
  event_t eventIdSToMTE3;

  TBuf<TPosition::VECCALC> UbBuf;
  LocalTensor<T> clearUb;
  LocalTensor<float> coeffUbBuff;
  LocalTensor<float> coeffUbRes;
  LocalTensor<T> coeffUbRes_;
};
#endif