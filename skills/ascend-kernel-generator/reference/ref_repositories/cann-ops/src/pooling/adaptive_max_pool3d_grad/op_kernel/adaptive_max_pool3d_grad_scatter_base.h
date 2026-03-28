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
 * \file adaptive_max_pool3d_grad_scatter_base.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_BASE_H
#define ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "adaptive_max_pool3d_grad_common.h"

namespace AdaptiveMaxPool3DGrad {
using namespace AscendC;
using namespace AdaptiveMaxPool3DGradComm;

const uint64_t REPEAT_BASE_BLOCK_LEN = 32;

template <typename TX, typename TGrad, typename TArgmax, typename TY>
class AdaptiveMaxPool3DGradScatterBase {
 public:
  __aicore__ inline AdaptiveMaxPool3DGradScatterBase(TPipe *pipe) {
    pipe_ = pipe;
  }

  __aicore__ inline void InitParams(const AdaptiveMaxPool3DGradTilingData* tiling) {
    params_.ncDim = tiling->ncDim;  // 池化正向输出结果NC维度
    params_.doDim = tiling->doDim;  // 池化正向输出结果d维度
    params_.hoDim = tiling->hoDim;  // 池化正向输出结果h维度
    params_.woDim = tiling->woDim;  // 池化正向输出结果w维度
    params_.diDim = tiling->diDim;  // 池化正向输入d维度
    params_.hiDim = tiling->hiDim;  // 池化正向输入h维度
    params_.wiDim = tiling->wiDim;  // 池化正向输入w维度

    params_.baseNc = tiling->baseNc;  // 每次CalcOutOffset矩阵的NC维度
    params_.baseDo = tiling->baseDo;  // 每次CalcOutOffset矩阵的d维度
    params_.baseHo = tiling->baseHo;  // 每次CalcOutOffset矩阵的h维度
    params_.baseWo = tiling->baseWo;  // 每次CalcOutOffset矩阵的w维度
    params_.ncTail = tiling->ncTail;
    params_.doTail = tiling->doTail;
    params_.hoTail = tiling->hoTail;
    params_.woTail = tiling->woTail;
    params_.ncCnt = tiling->ncCnt;  // nc方向base矩阵个数
    params_.dCnt = tiling->doCnt;   // h方向base矩阵个数
    params_.hCnt = tiling->hoCnt;   // h方向base矩阵个数
    params_.wCnt = tiling->woCnt;   // w方向base矩阵个数
    params_.usedCoreNum = tiling->usedCoreNum;

    params_.totalCnt = tiling->totalCnt;       // 需要处理base矩阵个数
    params_.ncCntRound = tiling->ncRound;      // 多核切nc，先分nc,向上取整,
    params_.preCoreNum = tiling->preCoreNum;  // 每个核均分完后剩余nce由前preCoreNum个核进行填充
    params_.diHiWiLen = params_.diDim * params_.hiDim * params_.wiDim;
    params_.ncRealRound = 0;
    params_.ubSize = tiling->totalUBSize;
    uint64_t blockId = GetBlockIdx();
    if (params_.preCoreNum == 0 || blockId < params_.preCoreNum) {  // 前preCoreNum个核
      params_.ncIndex = blockId * params_.ncCntRound;  // 由于轮数为向上取整，所以当前核填充数的起始位置为 填充数*核数
      params_.ncRealRound = params_.ncCntRound;
    } else {
      params_.ncIndex = params_.preCoreNum * (params_.ncCntRound) + (blockId - params_.preCoreNum) *  tiling->ncRoundTail;
      params_.ncRealRound = tiling->ncRoundTail;
    }
    params_.realRound =
        params_.ncRealRound * params_.dCnt * params_.hCnt * params_.wCnt;  // 处理的base矩阵个数，包括base矩阵和tail矩阵
  }

  __aicore__ inline void InitInputsOutputs(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace) {
    gradGm.SetGlobalBuffer((__gm__ TGrad*)grad);
    argmaxGm.SetGlobalBuffer((__gm__ TArgmax*)argmax);

    // 起始地址
    uint64_t initOffset = params_.ncIndex * params_.baseNc * params_.diHiWiLen;
    uint64_t initLen = 0;
    uint64_t ncIndex = params_.ncIndex;
    for (uint64_t j = 0; j < params_.ncRealRound; j++) {
      block_.ncShape = ncIndex >= (params_.ncCnt - 1) ? params_.ncTail : params_.baseNc;
      initLen += block_.ncShape * params_.diHiWiLen;
      ncIndex += 1;  // 当前ncCntIndex
    }
    params_.initLen = initLen;
    params_.initOffset = initOffset;

    yGm.SetGlobalBuffer((__gm__ TY*)y + initOffset, initLen);
    workspaceGm.SetGlobalBuffer((__gm__ float*)usrWorkspace + initOffset, initLen);
  }

  __aicore__ inline void InitUbBuffer() {
    pipe_->InitBuffer(gradQue, 1, params_.baseNc * params_.baseDo * params_.baseHo * params_.baseWo * sizeof(TGrad));
    pipe_->InitBuffer(argmaxQue, 1, params_.baseNc * params_.baseDo * params_.baseHo * params_.baseWo * sizeof(TArgmax));
  }

  __aicore__ inline void CopyInGrad() {
    LocalTensor<TGrad> gradUb = gradQue.AllocTensor<TGrad>();
    if (params_.baseWo == params_.woDim) {
      // repeat = VL, blockLen = baseHo * baseWo
      uint32_t baseblockLen = block_.doShape * block_.hoShape * block_.woShape * sizeof(TGrad);
      if (baseblockLen > REPEAT_BASE_BLOCK_LEN && baseblockLen % REPEAT_BASE_BLOCK_LEN == 0) {
        DataCopyExtParams copyParamsGrad;
        copyParamsGrad.blockCount = block_.ncShape;
        copyParamsGrad.blockLen = baseblockLen;
        copyParamsGrad.srcStride = 0;
        copyParamsGrad.dstStride = 0;
        DataCopyPadExtParams<TGrad> padGrad{false, 0, 0, 0};

        DataCopyPad(gradUb, gradGm[block_.offsetGrad], copyParamsGrad, padGrad);
      } else {
        DataCopyExtParams copyParamsGrad;
        uint32_t totalBlockLen = block_.ncShape * baseblockLen;
        uint16_t baseBlockCount = totalBlockLen / REPEAT_BASE_BLOCK_LEN;
        copyParamsGrad.blockCount = baseBlockCount;
        copyParamsGrad.blockLen = REPEAT_BASE_BLOCK_LEN;
        copyParamsGrad.srcStride = 0;
        copyParamsGrad.dstStride = 0;
        DataCopyPadExtParams<TGrad> padGrad{false, 0, 0, 0};
        DataCopyPad(gradUb, gradGm[block_.offsetGrad], copyParamsGrad, padGrad);

        uint32_t tailBlockLen = totalBlockLen % REPEAT_BASE_BLOCK_LEN;
        if (tailBlockLen != 0) {
          copyParamsGrad.blockCount = 1;
          copyParamsGrad.blockLen = tailBlockLen;
          padGrad.isPad = true;
          DataCopyPad(gradUb[baseBlockCount * (REPEAT_BASE_BLOCK_LEN / sizeof(TGrad))],
                      gradGm[block_.offsetGrad + baseBlockCount * (REPEAT_BASE_BLOCK_LEN / sizeof(TGrad))], copyParamsGrad,
                      padGrad);
        }
      }
    } else {
      // VL is for loop, repeat = baseHo, blockLen = baseWo
      for (uint64_t loopidx = 0; loopidx < block_.ncShape; loopidx++) {
        DataCopyExtParams copyParamsGrad;
        copyParamsGrad.blockCount = params_.baseHo;
        copyParamsGrad.blockLen = params_.baseWo * sizeof(TGrad);
        copyParamsGrad.srcStride = (params_.woDim - block_.woShape) * sizeof(TGrad);
        copyParamsGrad.dstStride = 0;
        DataCopyPadExtParams<TGrad> padGrad{false, 0, 0, 0};
        DataCopyPad(gradUb[loopidx * block_.hoShape * block_.woShape],
                    gradGm[block_.offsetGrad + loopidx * params_.hoDim * params_.woDim], copyParamsGrad, padGrad);
      }
    }
    gradQue.EnQue(gradUb);
  }

  __aicore__ inline void CopyInArgmax() {
    LocalTensor<TArgmax> argmaxUb = argmaxQue.AllocTensor<TArgmax>();
    if (params_.baseWo == params_.woDim) {
      uint32_t baseblockLen = block_.doShape * block_.hoShape * block_.woShape * sizeof(TArgmax);
      if (baseblockLen > REPEAT_BASE_BLOCK_LEN && baseblockLen % REPEAT_BASE_BLOCK_LEN == 0) {
        // repeat = VL, blockLen = baseHo * baseWo
        DataCopyExtParams copyParamsArgmax;
        copyParamsArgmax.blockCount = block_.ncShape;
        copyParamsArgmax.blockLen = baseblockLen;
        copyParamsArgmax.srcStride = 0;
        copyParamsArgmax.dstStride = 0;
        DataCopyPadExtParams<TArgmax> padArgmax{false, 0, 0, 0};
        DataCopyPad(argmaxUb, argmaxGm[block_.offsetArgmax], copyParamsArgmax, padArgmax);
      } else {
        DataCopyExtParams copyParamsArgmax;
        uint32_t totalBlockLen = block_.ncShape * baseblockLen;
        uint16_t baseBlockCount = totalBlockLen / REPEAT_BASE_BLOCK_LEN;
        copyParamsArgmax.blockCount = baseBlockCount;
        copyParamsArgmax.blockLen = REPEAT_BASE_BLOCK_LEN;
        copyParamsArgmax.srcStride = 0;
        copyParamsArgmax.dstStride = 0;
        DataCopyPadExtParams<TArgmax> padArgmax{false, 0, 0, 0};
        DataCopyPad(argmaxUb, argmaxGm[block_.offsetArgmax], copyParamsArgmax, padArgmax);

        uint16_t tailBlockLen = totalBlockLen % REPEAT_BASE_BLOCK_LEN;
        if (tailBlockLen != 0) {
          copyParamsArgmax.blockCount = 1;
          copyParamsArgmax.blockLen = tailBlockLen;
          copyParamsArgmax.srcStride = 0;
          copyParamsArgmax.dstStride = 0;
          padArgmax.isPad = true;
          DataCopyPad(argmaxUb[baseBlockCount * (REPEAT_BASE_BLOCK_LEN / sizeof(TArgmax))],
                      argmaxGm[block_.offsetArgmax + baseBlockCount * (REPEAT_BASE_BLOCK_LEN / sizeof(TArgmax))],
                      copyParamsArgmax, padArgmax);
        }
      }
    } else {
      // VL is for loop, repeat = baseHo, blockLen = baseWo
      for (uint64_t loopidx = 0; loopidx < block_.ncShape; loopidx++) {
        DataCopyExtParams copyParamsArgmax;
        copyParamsArgmax.blockCount = params_.baseHo;
        copyParamsArgmax.blockLen = params_.baseWo * sizeof(TArgmax);
        copyParamsArgmax.srcStride = (params_.woDim - block_.woShape) * sizeof(TArgmax);
        copyParamsArgmax.dstStride = 0;
        DataCopyPadExtParams<TArgmax> padGrad{false, 0, 0, 0};
        DataCopyPad(argmaxUb[loopidx * block_.hoShape * block_.woShape],
                    argmaxGm[block_.offsetArgmax + loopidx * params_.hoDim * params_.woDim], copyParamsArgmax, padGrad);
      }
    }
    argmaxQue.EnQue(argmaxUb);
  }

 public:
  TilingParams params_;
  BlockParams block_;
  TPipe *pipe_ = nullptr;

  GlobalTensor<TGrad> gradGm;
  GlobalTensor<TArgmax> argmaxGm;

  GlobalTensor<TY> yGm;
  GlobalTensor<float> workspaceGm;

  TQue<QuePosition::VECIN, 1> gradQue;
  TQue<QuePosition::VECIN, 1> argmaxQue;
  TQue<QuePosition::VECIN, 1> wsQue;
  TQue<QuePosition::VECOUT, 1> yQue;
};
}  // namespace AdaptiveMaxPool3DGrad
#endif  // ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_BASE_H
