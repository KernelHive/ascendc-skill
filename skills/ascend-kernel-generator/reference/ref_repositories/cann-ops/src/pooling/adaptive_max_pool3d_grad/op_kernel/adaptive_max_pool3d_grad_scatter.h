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
 * \file adaptive_max_pool3d_grad_scatter.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_H
#define ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "adaptive_max_pool3d_grad_common.h"
#include "adaptive_max_pool3d_grad_scatter_base.h"

namespace AdaptiveMaxPool3DGrad {
using namespace AscendC;
using namespace AdaptiveMaxPool3DGradComm;

template <typename TX, typename TGrad, typename TArgmax, typename TY>
class AdaptiveMaxPool3DGradScatter : public AdaptiveMaxPool3DGradScatterBase<TX, TGrad, TArgmax, TY> {
 public:
  __aicore__ inline AdaptiveMaxPool3DGradScatter(TPipe *pipe) : AdaptiveMaxPool3DGradScatterBase<TX, TGrad, TArgmax, TY>(pipe){
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace,
                              const AdaptiveMaxPool3DGradTilingData* tiling) {
    // load tiling data
    this->InitParams(tiling);
    // set global buffer
    this->InitInputsOutputs(x, grad, argmax, y, usrWorkspace);
    // init global memory
    InitYGMGlobalMemory(x, grad, argmax, y, usrWorkspace);
    // ub buffer init
    this->InitUbBuffer();
  }

  __aicore__ inline void InitYGMGlobalMemory(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace) {
    InitGlobalMemory(this->yGm, this->params_.initLen, static_cast<TY>(0));
    event_t eventMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    set_flag(PIPE_MTE3, PIPE_S, eventMTE3);
    wait_flag(PIPE_MTE3, PIPE_S, eventMTE3);
  }

  __aicore__ inline void CalcOutOffset() {
    LocalTensor<TArgmax> argmaxUb = this->argmaxQue.template DeQue<TArgmax>();  //  need free in the end
    LocalTensor<TGrad> gradUb = this->gradQue.template DeQue<TGrad>();          //  need free in the end

    event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    set_flag(PIPE_MTE2, PIPE_S, eventMTE2S);
    wait_flag(PIPE_MTE2, PIPE_S, eventMTE2S);

    uint64_t basedhwLen = this->block_.doShape * this->block_.hoShape * this->block_.woShape;
    for (uint64_t ncIdx = 0; ncIdx < this->block_.ncShape; ncIdx++) {
      uint64_t ncOffset = this->block_.baseNcOffset * this->params_.diHiWiLen;
      for (uint64_t dhwIdx = 0; dhwIdx < basedhwLen; dhwIdx++) {
        uint64_t ubOffset = ncIdx * basedhwLen + dhwIdx;
        uint64_t outOffset = ncOffset + (uint64_t)argmaxUb.GetValue(ubOffset);
        TGrad gradValue = gradUb.GetValue(ubOffset);
        this->yGm.SetValue(outOffset, gradValue);
        DataCacheCleanAndInvalid<TGrad, CacheLine::SINGLE_CACHE_LINE>(this->yGm[outOffset]);
      }
      this->block_.ShapeSum += basedhwLen;
      if (this->block_.ShapeSum == this->params_.doDim * this->params_.hoDim * this->params_.woDim) {
        this->block_.baseNcOffset += 1;
        this->block_.ShapeSum = 0;
      }
    }
    this->gradQue.FreeTensor(gradUb);
    this->argmaxQue.FreeTensor(argmaxUb);
  }

  __aicore__ inline void CalcBlock() {
    this->CopyInGrad();
    this->CopyInArgmax();
    CalcOutOffset();
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Process() {
    uint64_t ncIndex = this->params_.ncIndex;
    for (uint64_t i = 0; i < this->params_.ncRealRound; i++) {
      if (ncIndex < this->params_.ncCnt) {
        this->block_.ncCntIndex = ncIndex;
        this->block_.ncShape =
            this->block_.ncCntIndex >= (this->params_.ncCnt - 1) ? this->params_.ncTail : this->params_.baseNc;
        for (uint64_t j = 0; j < this->params_.dCnt; j++) {
          this->block_.doCntIndex = j;
          this->block_.doShape =
              this->block_.doCntIndex >= (this->params_.dCnt - 1) ? this->params_.doTail : this->params_.baseDo;
          for (uint64_t k = 0; k < this->params_.hCnt; k++) {
            this->block_.hoCntIndex = k;
            this->block_.hoShape =
                this->block_.hoCntIndex >= (this->params_.hCnt - 1) ? this->params_.hoTail : this->params_.baseHo;
            for (uint64_t l = 0; l < this->params_.wCnt; l++) {
              this->block_.woCntIndex = l;
              this->block_.woShape =
                  this->block_.woCntIndex >= (this->params_.wCnt - 1) ? this->params_.woTail : this->params_.baseWo;
              this->block_.offsetGrad =
                  this->block_.ncCntIndex * this->params_.baseNc * this->params_.doDim * this->params_.hoDim *
                      this->params_.woDim +
                  this->block_.doCntIndex * this->params_.baseDo * this->params_.hoDim * this->params_.woDim +
                  this->block_.hoCntIndex * this->params_.baseHo * this->params_.woDim +
                  this->block_.woCntIndex * this->params_.baseWo;
              this->block_.offsetArgmax = this->block_.offsetGrad;
              CalcBlock();
            }
          }
        }
        ncIndex += 1;  // 当前ncCntIndex
      }
    }
  }
};
}  // namespace AdaptiveMaxPool3DGrad
#endif  // ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_H