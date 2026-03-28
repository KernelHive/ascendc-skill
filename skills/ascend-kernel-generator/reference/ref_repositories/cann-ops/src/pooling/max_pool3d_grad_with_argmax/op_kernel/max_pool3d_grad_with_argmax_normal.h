/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file max_pool3d_grad_with_argmax_normal.h
 * \brief
 */
#ifndef MAX_POOL_GRAD3D_WITH_ARGMAX_NORMAL_H
#define MAX_POOL_GRAD3D_WITH_ARGMAX_NORMAL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "max_pool3d_grad_with_argmax_common.h"

namespace MaxPool3DGradWithArgmax {
using namespace AscendC;
using namespace MaxPool3DGradWithArgmaxComm;

struct OffsetParams
{
    uint64_t xIndex = 0;
    uint64_t argmax;
    uint64_t grad;
    uint64_t yTran;
    uint64_t argmaxTran;
    uint64_t gradTran;
    uint64_t mask;
    uint64_t tmpIndex;
    uint64_t tmpGrad;
    uint64_t y;
};

template <typename TX, typename TGrad, typename TArgmax, typename TY, bool IsOverlap>
class MaxPool3DGradWithArgmaxNormal {
public:
  __aicore__ inline MaxPool3DGradWithArgmaxNormal(TPipe *pipe) {
    pipe_ = pipe;
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                              GM_ADDR usrWorkspace, const MaxPool3DGradWithArgmaxTilingData* tiling) {
    // load tiling data
    InitParams(tiling);
    // set global buffer
    InitInputsOutputs(x, grad, argmax, y, usrWorkspace);
    // ub buffer init
    InitUbBuffer();
}

__aicore__ inline void InitParams(const MaxPool3DGradWithArgmaxTilingData* tiling) {
    para_.ncDim = tiling->ncDim;
    para_.diDim = tiling->diDim;
    para_.hiDim = tiling->hiDim;
    para_.wiDim = tiling->wiDim;
    para_.doDim = tiling->doDim;
    para_.hoDim = tiling->hoDim;
    para_.woDim = tiling->woDim;
    para_.kd = tiling->kd;
    para_.kh = tiling->kh;
    para_.kw = tiling->kw;
    para_.sd = tiling->sd;
    para_.sh = tiling->sh;
    para_.sw = tiling->sw;
    para_.padDTop = tiling->padDTop;
    para_.padHTop = tiling->padHTop;
    para_.padWTop = tiling->padWTop;
    para_.padDBottom = tiling->padDBottom;
    para_.padHBottom = tiling->padHBottom;
    para_.padWBottom = tiling->padWBottom;
    para_.singleCoreNc = tiling->singleCoreNc;
    para_.singleCoreDo = tiling->singleCoreDo;
    para_.singleCoreHo = tiling->singleCoreHo;
    para_.singleCoreWo = tiling->singleCoreWo;
    para_.baseNc = tiling->baseNc;
    para_.baseDo = tiling->baseDo;
    para_.baseHo = tiling->baseHo;
    para_.baseWo = tiling->baseWo;
    para_.ncCnt = tiling->ncCnt;
    para_.doCnt = tiling->doCnt;
    para_.hoCnt = tiling->hoCnt;
    para_.woCnt = tiling->woCnt;
    para_.totalCnt = tiling->totalCnt;
    para_.ncTail = tiling->ncTail;
    para_.doTail = tiling->doTail;
    para_.hoTail = tiling->hoTail;
    para_.woTail = tiling->woTail;
    para_.ubSize = tiling->totalUBSize;
    para_.padGmOffset = para_.padDTop * para_.hiDim * para_.wiDim +
                        para_.padHTop * para_.wiDim + 
                        para_.padWTop;
    para_.baseDoHoWo = para_.baseDo * para_.baseHo * para_.baseWo;
    para_.baseDoHoWoAlign8 = AlignUp(para_.baseDoHoWo, BLOCK_NUM_32);
    para_.baseDoHoWoAlign16 = AlignUp(para_.baseDoHoWo, BLOCK_NUM_16);
    para_.baseDi = (para_.baseDo - 1) * para_.sd + para_.kd;
    para_.baseHi = (para_.baseHo - 1) * para_.sh + para_.kh;
    para_.baseWi = (para_.baseWo - 1) * para_.sw + para_.kw;
    para_.baseDiHiWi = para_.baseDi * para_.baseHi * para_.baseWi;
    if constexpr (is_same<TY, float>::value || IsOverlap) {
        para_.baseWiAlign = AlignUp(para_.baseWi, BLOCK_NUM_32);
    } else {
        para_.baseWiAlign = AlignUp(para_.baseWi, BLOCK_NUM_16);
    }
    para_.baseDiHiWiAlign8 = para_.baseDi * para_.baseHi * AlignUp(para_.baseWi, BLOCK_NUM_32);
    para_.baseDiHiWiAlign16 = para_.baseDi * para_.baseHi * AlignUp(para_.baseWi, BLOCK_NUM_16);
    para_.baseDiHiWiAlign = para_.baseDi * para_.baseHi * para_.baseWiAlign;
}

__aicore__ inline void InitInputsOutputs(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace) {
    // GlobalBuffer is full, each core use offset to get real addr
    xGm.SetGlobalBuffer((__gm__ TX*)x);
    gradGm.SetGlobalBuffer((__gm__ TGrad*)grad);
    argmaxGm.SetGlobalBuffer((__gm__ TArgmax*)argmax);
    yGm.SetGlobalBuffer((__gm__ TY*)y);
    workspaceGm.SetGlobalBuffer((__gm__ float*)usrWorkspace);
}

__aicore__ inline void InitUbBuffer() {
    pipe_->InitBuffer(totalBuf, para_.ubSize);
    LocalTensor<uint8_t> totalUb = totalBuf.Get<uint8_t>();

    uint64_t xIndexSize = para_.baseDiHiWi * para_.baseNc * sizeof(int32_t);
    uint64_t argmaxSize = para_.baseNc * para_.baseDoHoWoAlign8 * sizeof(int32_t);
    uint64_t argmaxTranSize = para_.baseDoHoWoAlign8 * para_.baseNc * sizeof(int32_t);
    uint64_t gradSize;
    uint64_t gradTranSize;
    if constexpr (is_same<TGrad, float>::value) {
        gradSize = para_.baseNc * para_.baseDoHoWoAlign8 * sizeof(float);
        gradTranSize = para_.baseNc * para_.baseDoHoWoAlign8 * sizeof(float);
    } else {
        gradSize = para_.baseNc * para_.baseDoHoWoAlign16 * sizeof(half);
        gradTranSize = para_.baseNc * para_.baseDoHoWoAlign16 * sizeof(half);
    }
    uint64_t yTranSize;
    uint64_t ySize;
    if constexpr (is_same<TY, float>::value || IsOverlap) {
        yTranSize = para_.baseNc * para_.baseDiHiWiAlign8 * sizeof(float);
        ySize = para_.baseNc * para_.baseDiHiWiAlign8 * sizeof(float);
    } else {
        yTranSize = para_.baseNc * para_.baseDiHiWiAlign16 * sizeof(float);
        ySize = para_.baseNc * para_.baseDiHiWiAlign16 * sizeof(half);
    }
    uint64_t maskSize = AlignUp(para_.baseDoHoWo * para_.baseNc , BLOCK_SIZE);

    //  | xIndex | argmax   | grad    | yTran | argmaxTran | gradTran | mask | y |
    //  |        | tmpIndex |         |       |            |                 |   |
    //  |        | tmpGrad  |         |       |            |                 |   |
    ubOffset_.argmax = xIndexSize;
    ubOffset_.grad = ubOffset_.argmax + argmaxSize;
    ubOffset_.yTran = ubOffset_.grad + gradSize;
    ubOffset_.argmaxTran = ubOffset_.yTran + yTranSize;
    ubOffset_.gradTran = ubOffset_.argmaxTran + argmaxTranSize;
    ubOffset_.mask = ubOffset_.gradTran + gradTranSize;
    ubOffset_.tmpIndex = ubOffset_.argmax;
    ubOffset_.tmpGrad = ubOffset_.argmax;
    ubOffset_.y = ubOffset_.mask + maskSize;

    LocalTensor<int32_t> argmaxUb = totalUb[ubOffset_.argmax].ReinterpretCast<int32_t>();
    LocalTensor<int32_t> argmaxTranUb = totalUb[ubOffset_.argmaxTran].ReinterpretCast<int32_t>();
    LocalTensor<TGrad> gradUb = totalUb[ubOffset_.grad].ReinterpretCast<TGrad>();
    LocalTensor<TGrad> gradTranUb = totalUb[ubOffset_.gradTran].ReinterpretCast<TGrad>();
    LocalTensor<TY> yTranUb = totalUb[ubOffset_.yTran].ReinterpretCast<TY>();
    LocalTensor<float> yTranUbFp32 = totalUb[ubOffset_.yTran].ReinterpretCast<float>();
    LocalTensor<TY> yUb = totalUb[ubOffset_.y].ReinterpretCast<TY>();
    LocalTensor<float> yUbFp32 = totalUb[ubOffset_.y].ReinterpretCast<float>();

    for (uint64_t r = 0; r < para_.baseNc / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
          argmaxAddrList_[r][i] =
              (uint64_t)(argmaxUb[r * TRANS_ADDR_LEN * para_.baseDoHoWoAlign8 + i * para_.baseDoHoWoAlign8]
                             .GetPhyAddr());
          argmaxTranAddrList_[r][i] =
              (uint64_t)(argmaxTranUb[r * TRANS_ADDR_LEN + i / NUM_TWO * para_.baseNc + i % NUM_TWO * BLOCK_NUM_32].GetPhyAddr());

          if constexpr (is_same<TGrad, float>::value) {
            gradAddrList_[r][i] =
                (uint64_t)(gradUb[r * TRANS_ADDR_LEN * para_.baseDoHoWoAlign8 + i * para_.baseDoHoWoAlign8]
                               .GetPhyAddr());
            gradTranAddrList_[r][i] =
                (uint64_t)(gradTranUb[r * TRANS_ADDR_LEN + i / NUM_TWO * para_.baseNc + i % NUM_TWO * BLOCK_NUM_32].GetPhyAddr());
            } else {
              gradAddrList_[r][i] =
                  (uint64_t)(gradUb[r * TRANS_ADDR_LEN * para_.baseDoHoWoAlign16 + i * para_.baseDoHoWoAlign16]
                                 .GetPhyAddr());
              gradTranAddrList_[r][i] = (uint64_t)(gradTranUb[r * TRANS_ADDR_LEN + i * para_.baseNc].GetPhyAddr());
            }
            // copy out
            if constexpr (is_same<TY, float>::value || IsOverlap) {
                yTranAddrList_[r][i] = (uint64_t)(yTranUbFp32[r * TRANS_ADDR_LEN + i % BLOCK_NUM_32 * para_.baseNc +
                                                              i / BLOCK_NUM_32 * BLOCK_NUM_32]
                                                      .GetPhyAddr());
                yAddrList_[r][i] = (uint64_t)(yUbFp32[r * TRANS_ADDR_LEN * para_.baseDiHiWiAlign +
                                                      (i % NUM_TWO * BLOCK_NUM_32 + i / NUM_TWO) * para_.baseDiHiWiAlign]
                                                  .GetPhyAddr());
            } else {
                yTranAddrList_[r][i] = (uint64_t)(yTranUb[r * TRANS_ADDR_LEN + i * para_.baseNc].GetPhyAddr());
                yAddrList_[r][i] =
                    (uint64_t)(yUb[r * TRANS_ADDR_LEN * para_.baseDiHiWiAlign + i * para_.baseDiHiWiAlign]
                                   .GetPhyAddr());
            }
        }
    }
}

__aicore__ inline void Process() {
    bool needInitOutput = (para_.doDim * para_.kd < para_.diDim + para_.padDTop + para_.padDBottom) ||
                          (para_.hoDim * para_.kh < para_.hiDim + para_.padHTop + para_.padHBottom) ||
                          (para_.woDim * para_.kw < para_.wiDim + para_.padWTop + para_.padWBottom) ||
                          IsOverlap;
    if (needInitOutput) {
        if constexpr (is_same<TY, float>::value || !IsOverlap) {
            ProcessInit0(yGm);
        } else {
            ProcessInit0(workspaceGm);
        }
      SyncAll();
    }
    // gen initX
    LocalTensor<uint8_t> totalUb = totalBuf.Get<uint8_t>();
    LocalTensor<int32_t> xIndexUb = totalUb[ubOffset_.xIndex].ReinterpretCast<int32_t>();
    GenXInitIndex(xIndexUb);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint64_t totalIndex = 0; totalIndex < para_.totalCnt; totalIndex++) {
        if (GetBlockIdx() == totalIndex % GetBlockNum()) {
            core_.ncCntIndex = totalIndex / (para_.doCnt * para_.hoCnt * para_.woCnt);
            core_.doCntIndex = totalIndex / (para_.hoCnt * para_.woCnt) % para_.doCnt;
            core_.hoCntIndex = totalIndex / para_.woCnt % para_.hoCnt;
            core_.woCntIndex = totalIndex % para_.woCnt;
            core_.ncShape = core_.ncCntIndex == para_.ncCnt - 1 ? para_.ncTail : para_.singleCoreNc;
            core_.doShape = core_.doCntIndex == para_.doCnt - 1 ? para_.doTail : para_.singleCoreDo;
            core_.hoShape = core_.hoCntIndex == para_.hoCnt - 1 ? para_.hoTail : para_.singleCoreHo;
            core_.woShape = core_.woCntIndex == para_.woCnt - 1 ? para_.woTail : para_.singleCoreWo;
            core_.offsetArgmax =
                core_.ncCntIndex * para_.singleCoreNc * para_.doDim * para_.hoDim * para_.woDim +
                core_.doCntIndex * para_.singleCoreDo * para_.hoDim * para_.woDim +
                core_.hoCntIndex * para_.singleCoreHo * para_.woDim +
                core_.woCntIndex * para_.singleCoreWo;
            core_.offsetGrad = core_.offsetArgmax;
            core_.offsetYD = core_.doCntIndex * para_.singleCoreDo * para_.sd;
            core_.offsetYH = core_.hoCntIndex * para_.singleCoreHo * para_.sh;
            core_.offsetYW = core_.woCntIndex * para_.singleCoreWo * para_.sw;
            core_.offsetY = core_.ncCntIndex * para_.singleCoreNc * para_.diDim * para_.hiDim * para_.wiDim +
                            core_.offsetYD * para_.hiDim * para_.wiDim +
                            core_.offsetYH * para_.wiDim +
                            core_.offsetYW;  // no pad
            SubProcess();
        }
    }
    if constexpr (!is_same<TY, float>::value && IsOverlap) {
        SyncAll();
        InitCastUbBuffer();
        ProcessCast();
    }
}

template <typename T>
__aicore__ inline void ProcessInit0(const GlobalTensor<T>& outGm) {
    uint64_t maxCalcNum = para_.ubSize / sizeof(T);
    uint64_t totalNum = para_.ncDim * para_.diDim * para_.hiDim * para_.wiDim;
    uint64_t totalLoops = CeilDiv(totalNum, maxCalcNum);
    uint64_t calcTail = totalNum - (totalLoops - 1) * maxCalcNum;
    LocalTensor<T> totalUb = totalBuf.Get<T>();
    Duplicate(totalUb, (T)0, maxCalcNum);  //也可以根据实际数据dump对应值
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    for (uint64_t loopIndex = 0; loopIndex < totalLoops; loopIndex++) {
        if (GetBlockIdx() == loopIndex % GetBlockNum()) {
            uint64_t calcNum = (loopIndex == totalLoops - 1) ? calcTail : maxCalcNum;
            CopyOut0(outGm[loopIndex * maxCalcNum], calcNum);
        }
    }
}

template <typename T>
__aicore__ inline void CopyOut0(const GlobalTensor<T>& outGm, uint64_t calcNum) {
    LocalTensor<T> totalUb = totalBuf.Get<T>();
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = calcNum * sizeof(T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyPad(outGm, totalUb, copyOutParams);
}

__aicore__ inline void SubProcess() {
    uint64_t incoreDCnt = CeilDiv(core_.doShape, para_.baseDo);
    uint64_t incoreHCnt = CeilDiv(core_.hoShape, para_.baseHo);
    uint64_t incoreWCnt = CeilDiv(core_.woShape, para_.baseWo);
    uint64_t incoreDoTail = core_.doShape - (incoreDCnt - 1) * para_.baseDo;
    uint64_t incoreHoTail = core_.hoShape - (incoreHCnt - 1) * para_.baseHo;
    uint64_t incoreWoTail = core_.woShape - (incoreWCnt - 1) * para_.baseWo;
    block_.ncShape = core_.ncShape;
    for (uint64_t doCntIndex = 0; doCntIndex < incoreDCnt; doCntIndex++) {
        block_.doCntIndex = doCntIndex;
        block_.doShape = doCntIndex == (incoreDCnt - 1) ? incoreDoTail : para_.baseDo;
        block_.diShape = (block_.doShape - 1) * para_.sd + para_.kd;
        block_.offsetYD = block_.doCntIndex * para_.baseDo * para_.sd;
        block_.padDTop =
            (core_.offsetYD + block_.offsetYD > para_.padDTop) ? 0 : para_.padDTop - core_.offsetYD - block_.offsetYD;
        block_.padDBottom = (core_.offsetYD + block_.offsetYD + block_.diShape > para_.padDTop + para_.diDim)
                                ? core_.offsetYD + block_.offsetYD + block_.diShape - para_.padDTop - para_.diDim
                                : 0;
        block_.diValid = block_.diShape - block_.padDTop - block_.padDBottom;
        for (uint64_t hoCntIndex = 0; hoCntIndex < incoreHCnt; hoCntIndex++) {
            block_.hoCntIndex = hoCntIndex;
            block_.hoShape = hoCntIndex == (incoreHCnt - 1) ? incoreHoTail : para_.baseHo;
            block_.hiShape = (block_.hoShape - 1) * para_.sh + para_.kh;
            block_.offsetYH = block_.hoCntIndex * para_.baseHo * para_.sh;
            block_.padHTop = (core_.offsetYH + block_.offsetYH > para_.padHTop)
                                    ? 0
                                    : para_.padHTop - core_.offsetYH - block_.offsetYH;
            block_.padHBottom =
                (core_.offsetYH + block_.offsetYH + block_.hiShape > para_.padHTop + para_.hiDim)
                    ? core_.offsetYH + block_.offsetYH + block_.hiShape - para_.padHTop - para_.hiDim
                    : 0;
            block_.hiValid = block_.hiShape - block_.padHTop - block_.padHBottom;
            for (uint64_t woCntIndex = 0; woCntIndex < incoreWCnt; woCntIndex++) {
                block_.woCntIndex = woCntIndex;
                block_.woShape = woCntIndex == (incoreWCnt - 1) ? incoreWoTail : para_.baseWo;
                block_.wiShape = (block_.woShape - 1) * para_.sw + para_.kw;

                block_.offsetYW = block_.woCntIndex * para_.baseWo * para_.sw;
                block_.padWTop = (core_.offsetYW + block_.offsetYW > para_.padWTop)
                                     ? 0
                                     : para_.padWTop - core_.offsetYW - block_.offsetYW;
                block_.padWBottom =
                    (core_.offsetYW + block_.offsetYW + block_.wiShape > para_.padWTop + para_.wiDim)
                        ? core_.offsetYW + block_.offsetYW + block_.wiShape - para_.padWTop - para_.wiDim
                        : 0;
                block_.wiValid = block_.wiShape - block_.padWTop - block_.padWBottom;

                block_.dohowoShape = block_.doShape * block_.hoShape * block_.woShape;

                block_.offsetArgmax = block_.doCntIndex * para_.baseDo * para_.hoDim * para_.woDim + 
                                      block_.hoCntIndex * para_.baseHo * para_.woDim + 
                                      block_.woCntIndex * para_.baseWo; // 和offsetGrad是一致的
                block_.offsetGrad = block_.offsetArgmax;
                block_.offsetY = block_.offsetYD * para_.hiDim * para_.wiDim +
                                 block_.offsetYH * para_.wiDim +
                                 block_.offsetYW;
                CalcBlock();
            }
        }
    }
}

__aicore__ inline void CalcBlock() {
    LocalTensor<uint8_t> totalUb = totalBuf.Get<uint8_t>();
    LocalTensor<int32_t> xIndexUb = totalUb[ubOffset_.xIndex].ReinterpretCast<int32_t>();

    CopyInArgmax();
    event_t eventArgmaxMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventArgmaxMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventArgmaxMTE2ToV);

    LocalTensor<int32_t> argmaxUb = totalUb[ubOffset_.argmax].ReinterpretCast<int32_t>();
    LocalTensor<int32_t> argmaxTranUb = totalUb[ubOffset_.argmaxTran].ReinterpretCast<int32_t>();
    uint64_t dohowoAlign8 = AlignUp(block_.dohowoShape, BLOCK_NUM_32);
    TransposeAddrBase16M8(argmaxTranAddrList_, argmaxAddrList_, para_.baseNc, dohowoAlign8);

    CopyInGrad();
    event_t eventGradMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventGradMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventGradMTE2ToV);

    LocalTensor<TGrad> gradUb = totalUb[ubOffset_.grad].ReinterpretCast<TGrad>();
    LocalTensor<TGrad> gradTranUb = totalUb[ubOffset_.gradTran].ReinterpretCast<TGrad>();
    if constexpr (is_same<TGrad, float>::value) {
        TransposeAddrBase16M8(gradTranAddrList_, gradAddrList_, para_.baseNc, dohowoAlign8);
    } else {
        uint64_t dohowoAlign16 = AlignUp(block_.dohowoShape, BLOCK_NUM_16);
        TransposeAddrBase16M16(gradTranAddrList_, gradAddrList_, para_.baseNc, dohowoAlign16);
    }

    LocalTensor<int32_t> tmpIndexUb = totalUb[ubOffset_.tmpIndex].ReinterpretCast<int32_t>();
    LocalTensor<uint8_t> maskUb = totalUb[ubOffset_.mask];
    LocalTensor<TGrad> tmpGradUb = totalUb[ubOffset_.tmpGrad].ReinterpretCast<TGrad>();
    LocalTensor<float> tmpGradUbFp32 = totalUb[ubOffset_.tmpGrad].ReinterpretCast<float>();
    LocalTensor<TY> yTranUb = totalUb[ubOffset_.yTran].ReinterpretCast<TY>();
    LocalTensor<float> yTranUbFp32 = totalUb[ubOffset_.yTran].ReinterpretCast<float>();
    Duplicate(yTranUbFp32, 0.0f, para_.baseDiHiWiAlign * para_.baseNc);

    for (uint64_t kdId = 0; kdId < para_.kd; kdId++) {
        for (uint64_t khId = 0; khId < para_.kh; khId++) {
            for (uint64_t kwId = 0; kwId < para_.kw; kwId++) {
                uint64_t offsetIn = kdId * para_.baseHi * para_.baseWi * para_.baseNc +
                                    khId * para_.baseWi * para_.baseNc +
                                    kwId * para_.baseNc;
                Img2ColPart(tmpIndexUb, xIndexUb[offsetIn]);
                AscendC::PipeBarrier<PIPE_V>();
                Compare(maskUb, tmpIndexUb, argmaxTranUb, CMPMODE::EQ,
                        AlignUp(block_.dohowoShape * para_.baseNc, B32_VECTOR_MASK));
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (is_same<TGrad, float>::value) {
                    SelectGrad(tmpGradUb, maskUb, gradTranUb);
                } else {
                    SelectGrad(tmpGradUb[block_.dohowoShape * para_.baseNc], maskUb, gradTranUb);
                    AscendC::PipeBarrier<PIPE_V>();
                    Cast(tmpGradUbFp32, tmpGradUb[block_.dohowoShape * para_.baseNc], RoundMode::CAST_NONE,
                         block_.dohowoShape * para_.baseNc);
                }
                AscendC::PipeBarrier<PIPE_V>();
                // ((doShape * hoShape * woShape), 64) -> (diShape * hiShape * Align(wiShape), 64)
                uint64_t offsetOut = kdId * para_.baseHi * para_.baseWiAlign * para_.baseNc +
                                     khId * para_.baseWiAlign * para_.baseNc + 
                                     kwId * para_.baseNc;
                Col2ImgPart(yTranUbFp32[offsetOut], tmpGradUbFp32);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);

    DePad(yTranUbFp32);  // 可以考虑转移到cast后
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (!IsOverlap) {
        if constexpr (is_same<TY, half>::value) {
            Cast(yTranUb, yTranUbFp32, RoundMode::CAST_NONE, para_.baseDiHiWiAlign * para_.baseNc);  // 也可以只cast valid
        } else if constexpr (is_same<TY, bfloat16_t>::value) {
            Cast(yTranUb, yTranUbFp32, RoundMode::CAST_RINT, para_.baseDiHiWiAlign * para_.baseNc);
        }
    }
    event_t eventGradMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventGradMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventGradMTE3ToV);  // 考虑不与y复用，容易阻塞
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    if constexpr (is_same<TY, float>::value) {
        TransposeAddrBase8M16(yAddrList_, yTranAddrList_, para_.baseDiHiWiAlign, para_.baseNc);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        CopyOut(yGm);
    } else if constexpr (IsOverlap) {
        TransposeAddrBase8M16(yAddrList_, yTranAddrList_, para_.baseDiHiWiAlign, para_.baseNc);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        CopyOut(workspaceGm);
    } else {
        TransposeAddrBase16M16B(yAddrList_, yTranAddrList_, para_.baseDiHiWiAlign, para_.baseNc);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        CopyOut(yGm);
    }
}

__aicore__ inline void GenXInitIndex(LocalTensor<int32_t>& dstLocal) {
    // 1.Dup first Value
    Duplicate(dstLocal, (int32_t)0, para_.baseNc);
    AscendC::PipeBarrier<PIPE_V>();

    // 2. Gen wiShape * vL
    uint64_t wIdx = 1;
    for (; wIdx * NUM_TWO <= para_.baseWi; wIdx = wIdx * NUM_TWO) {
        Adds(dstLocal[wIdx * para_.baseNc], dstLocal, (int32_t)wIdx, wIdx * para_.baseNc);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (wIdx != para_.baseWi) {
        Adds(dstLocal[wIdx * para_.baseNc], dstLocal, (int32_t)wIdx, (para_.baseWi - wIdx) * para_.baseNc);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 3. Gen hiShape * wiShape * vL
    uint64_t hIdx = 1;
    for (; hIdx * NUM_TWO <= para_.baseHi; hIdx = hIdx * NUM_TWO) {
        Adds(dstLocal[hIdx * para_.baseWi * para_.baseNc], dstLocal, (int32_t)(hIdx * para_.wiDim),
             hIdx * para_.baseWi * para_.baseNc); // offset hIdx * para_.baseWi * para_.baseNc
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (hIdx != para_.baseHi) {
        Adds(dstLocal[hIdx * para_.baseWi * para_.baseNc], dstLocal, (int32_t)(hIdx * para_.wiDim),
             (para_.baseHi - hIdx) * para_.baseWi * para_.baseNc);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // 4. Gen diShape * hiShape * wiShape * vL
    uint64_t dIdx = 1;
    for (; dIdx * NUM_TWO <= para_.baseDi; dIdx = dIdx * NUM_TWO) {
        Adds(dstLocal[dIdx * para_.baseHi * para_.baseWi * para_.baseNc], dstLocal,
             (int32_t)(dIdx * para_.hiDim * para_.wiDim), dIdx * para_.baseHi * para_.baseWi * para_.baseNc);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (dIdx != para_.baseDi) {
      Adds(dstLocal[dIdx * para_.baseHi * para_.baseWi * para_.baseNc], dstLocal,
           (int32_t)(dIdx * para_.hiDim * para_.wiDim),
           (para_.baseDi - dIdx) * para_.baseHi * para_.baseWi * para_.baseNc);
      AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void CopyInArgmax() {
    LocalTensor<uint8_t> totalUb = totalBuf.Get<uint8_t>();
    LocalTensor<TArgmax> argmaxUb = totalUb[ubOffset_.argmax].ReinterpretCast<TArgmax>();
    DataCopyExtParams copyParamsArgmax;
    copyParamsArgmax.blockCount = block_.ncShape;
    copyParamsArgmax.blockLen = block_.dohowoShape * sizeof(TArgmax);
    copyParamsArgmax.srcStride = (para_.doDim * para_.hoDim * para_.woDim - block_.dohowoShape) * sizeof(TArgmax);
    // dstStride equal with (para_.baseDoHoWoAlign8 - block_.dohowoAlign8) / BLOCK_NUM_32;
    copyParamsArgmax.dstStride = (para_.baseDoHoWoAlign8 - block_.dohowoShape) / BLOCK_NUM_32;
    DataCopyPadExtParams<TArgmax> padArgmax{false, 0, 0, 0};
    DataCopyPad(argmaxUb, argmaxGm[core_.offsetArgmax + block_.offsetArgmax], copyParamsArgmax, padArgmax);
}

__aicore__ inline void CopyInGrad() {
    LocalTensor<uint8_t> totalUb = totalBuf.Get<uint8_t>();
    LocalTensor<TGrad> gradUb = totalUb[ubOffset_.grad].ReinterpretCast<TGrad>();
    DataCopyExtParams copyParamsGrad;
    copyParamsGrad.blockCount = block_.ncShape;
    copyParamsGrad.blockLen = block_.dohowoShape * sizeof(TGrad);
    copyParamsGrad.srcStride = (para_.doDim * para_.hoDim * para_.woDim - block_.dohowoShape) * sizeof(TGrad);
    if constexpr (is_same<TY, float>::value) {
        copyParamsGrad.dstStride = (para_.baseDoHoWoAlign8 - block_.dohowoShape) / BLOCK_NUM_32;
    } else {
        copyParamsGrad.dstStride = (para_.baseDoHoWoAlign16 - block_.dohowoShape) / BLOCK_NUM_16;
    }

    DataCopyPadExtParams<TGrad> padGrad{false, 0, 0, 0};
    DataCopyPad(gradUb, gradGm[core_.offsetGrad + block_.offsetGrad], copyParamsGrad, padGrad);
}

template <typename T>
__aicore__ inline void Img2ColPart(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal) {
    T firstValue = core_.offsetY + block_.offsetY -
                   core_.ncCntIndex * para_.singleCoreNc * para_.diDim * para_.hiDim * para_.wiDim -
                   para_.padGmOffset;
    uint64_t mask = block_.ncShape;  // 256 / sizeof(T)
    if (likely(para_.sw < 32)) {  // para_.sw * 8 <= 255, 255 is maximum of uint8
        uint8_t baseRepStride = para_.baseNc / 8;
        for (uint64_t doIdx = 0; doIdx < block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < block_.hoShape; hoIdx++) {
                uint64_t dstOffset = doIdx * para_.baseHo * para_.baseWo * para_.baseNc +
                                     hoIdx * para_.baseWo * para_.baseNc;
                uint64_t srcOffset = doIdx * para_.sd * para_.baseHi * para_.baseWi * para_.baseNc +
                                     hoIdx * para_.sh * para_.baseWi * para_.baseNc;
                Adds(dstLocal[dstOffset], srcLocal[srcOffset], firstValue, mask, block_.woShape,
                    {1, 1, baseRepStride, static_cast<uint8_t>(para_.sw * baseRepStride)});
            }
        }
    } else {
        for (uint64_t doIdx = 0; doIdx < block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < block_.hoShape; hoIdx++) {
                for (uint64_t woIdx = 0; woIdx < block_.woShape; woIdx++) {
                    uint64_t dstOffset = doIdx * para_.baseHo * para_.baseWo * para_.baseNc +
                                         hoIdx * para_.baseWo * para_.baseNc +
                                         woIdx * para_.baseNc;
                    uint64_t srcOffset = doIdx * para_.sd * para_.baseHi * para_.baseWi * para_.baseNc +
                                         hoIdx * para_.sh * para_.baseWi * para_.baseNc +
                                         woIdx * para_.sw * para_.baseNc;
                    Adds(dstLocal[dstOffset], srcLocal[srcOffset], firstValue, mask, 1, {1, 1, 0, 0});
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Col2ImgPart(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal) {
    uint64_t mask = block_.ncShape;  // 256 / sizeof(T)
    if (likely(para_.sw < 32)) {  // para_.sw * 8 <= 255, 255 is maximum of uint8
        uint8_t baseRepStride = para_.baseNc / 8;
        for (uint64_t doIdx = 0; doIdx < block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < block_.hoShape; hoIdx++) {
                // ((doShape * hoShape * woShape), 64) -> (diShape * hiShape * Align(wiShape), 64)
                uint64_t dstOffset = doIdx * para_.sd * para_.baseHi * para_.baseWiAlign * para_.baseNc +
                                     hoIdx * para_.sh * para_.baseWiAlign * para_.baseNc;

                uint64_t srcOffset = doIdx * para_.baseHo * para_.baseWo * para_.baseNc +
                                     hoIdx * para_.baseWo * para_.baseNc;
                Add(dstLocal[dstOffset], dstLocal[dstOffset], srcLocal[srcOffset], mask, block_.woShape,
                    {1, 1, 1, static_cast<uint8_t>(para_.sw * baseRepStride),
                     static_cast<uint8_t>(para_.sw * baseRepStride), baseRepStride});
            }
        }
    } else {
        for (uint64_t doIdx = 0; doIdx < block_.doShape; doIdx++) {
            for (uint64_t hoIdx = 0; hoIdx < block_.hoShape; hoIdx++) {
                for (uint64_t woIdx = 0; woIdx < block_.woShape; woIdx++) {
                    // ((doShape * hoShape * woShape), 64) -> (diShape * hiShape * Align(wiShape), 64)
                    uint64_t dstOffset = doIdx * para_.sd * para_.baseHi * para_.baseWiAlign * para_.baseNc +
                                         hoIdx * para_.sh * para_.baseWiAlign * para_.baseNc +
                                         woIdx * para_.sw * para_.baseNc;

                    uint64_t srcOffset = doIdx * para_.baseHo * para_.baseWo * para_.baseNc +
                                         hoIdx * para_.baseWo * para_.baseNc +
                                         woIdx * para_.baseNc;
                    Add(dstLocal[dstOffset], dstLocal[dstOffset], srcLocal[srcOffset], mask, 1, {1, 1, 1, 0, 0, 0});
                }
            }
        }
    }
}

__aicore__ inline void SelectGrad(const LocalTensor<TGrad>& dstLocal, const LocalTensor<uint8_t>& maskUb,
                                  const LocalTensor<TGrad>& src0Local) {
    // 注意BF16做reinterpret
    if constexpr (is_same<TY, bfloat16_t>::value) {
        LocalTensor<half> dstLocalFp16 = dstLocal.template ReinterpretCast<half>();
        LocalTensor<half> src0LocalFp16 = src0Local.template ReinterpretCast<half>();
        Select(dstLocalFp16, maskUb, src0LocalFp16, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               block_.dohowoShape * para_.baseNc);
    } else {
        Select(dstLocal, maskUb, src0Local, static_cast<TGrad>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               block_.dohowoShape * para_.baseNc);
    }
}

__aicore__ inline void DePad(LocalTensor<float>& dstLocal) {
    // 到转置前di,hi,wi是连续的，需要在切分不同轴时，将每部分对齐到32byte对齐，需要进行增减pad

    // 经过Col2Img之后， 梯度结果储存在(diShape, hiShape, Align(wiShape), baseNc)中，当结果存在前pad时，需要将前pad消除
    // 
    if (block_.padDTop > 0 || block_.padHTop > 0 || block_.padWTop > 0) {
        uint64_t padOffset = block_.padDTop * para_.baseHi * para_.baseWiAlign +
                             block_.padHTop * para_.baseWiAlign +
                             block_.padWTop;
        // 暂时拷贝pad后面的所有数据
        Adds(dstLocal, dstLocal[padOffset * para_.baseNc], 0.0f, (para_.baseDiHiWiAlign - padOffset) * para_.baseNc);
    }
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& outGm) {
    LocalTensor<uint8_t> totalUb = totalBuf.Get<uint8_t>();
    LocalTensor<T> yUb = totalUb[ubOffset_.y].ReinterpretCast<T>();
    if constexpr (IsOverlap) {
        SetAtomicAdd<T>();
    }
    DataCopyExtParams copyParamsY;
    
    for (uint64_t diIdx = 0; diIdx < block_.diValid; diIdx++) {
        for (uint64_t hiIdx = 0; hiIdx < block_.hiValid; hiIdx++) {
            copyParamsY.blockCount = block_.ncShape;
            copyParamsY.blockLen = block_.wiValid * sizeof(T);
            if constexpr (is_same<T, float>::value) {
                // srcStride equal with para_.baseDiHiWiAlign / BLOCK_NUM_32 - CeilDiv(block_.wiValid, BLOCK_NUM_32)
                copyParamsY.srcStride = (para_.baseDiHiWiAlign - block_.wiValid) / BLOCK_NUM_32;
            } else {
                // srcStride  // equal with para_.baseDiHiWiAlign / BLOCK_NUM_16 - CeilDiv(block_.wiValid, BLOCK_NUM_16)
                copyParamsY.srcStride = (para_.baseDiHiWiAlign - block_.wiValid) / BLOCK_NUM_16;
            }

            copyParamsY.dstStride = (para_.diDim * para_.hiDim * para_.wiDim - block_.wiValid) * sizeof(T);
            uint64_t srcOffset = diIdx * para_.baseHi * para_.baseWiAlign + 
                                    hiIdx * para_.baseWiAlign;
            uint64_t blockPadOffset = block_.padDTop * para_.hiDim * para_.wiDim +
                                      block_.padHTop * para_.wiDim +
                                      block_.padWTop;
            uint64_t dstOffset = core_.offsetY + block_.offsetY + blockPadOffset - para_.padGmOffset;
            DataCopyPad(
                outGm[dstOffset + diIdx * para_.hiDim * para_.wiDim + hiIdx * para_.wiDim],
                yUb[srcOffset], copyParamsY);
        }
    }

    if constexpr (IsOverlap) {
        SetAtomicNone();
    }
}

__aicore__ inline void InitCastUbBuffer() {
    pipe_->Reset();
    uint64_t maxCalcNum = para_.ubSize / (sizeof(half) + sizeof(float));
    pipe_->InitBuffer(wsQue, 1, maxCalcNum * sizeof(float));
    pipe_->InitBuffer(yQue, 1, maxCalcNum * sizeof(half));
}

__aicore__ inline void ProcessCast() {
    uint64_t maxCalcNum = para_.ubSize / (sizeof(half) + sizeof(float));
    uint64_t totalLoops = CeilDiv(para_.ncDim * para_.diDim * para_.hiDim * para_.wiDim, maxCalcNum);
    uint64_t calcTail = para_.ncDim * para_.diDim * para_.hiDim * para_.wiDim - (totalLoops - 1) * maxCalcNum;
    for (uint64_t loopIndex = 0; loopIndex < totalLoops; loopIndex++) {
        if (GetBlockIdx() == loopIndex % GetBlockNum()) {
            uint64_t calcNum = (loopIndex == totalLoops - 1) ? calcTail : maxCalcNum;
            CopyInWorkspace(loopIndex * maxCalcNum, calcNum);
            ComputeCast(calcNum);
            CopyOutCast(loopIndex * maxCalcNum, calcNum);
        }
    }
}

__aicore__ inline void CopyInWorkspace(uint64_t gmOffset, uint64_t calcNum) {
    LocalTensor<float> fp32Ub = wsQue.AllocTensor<float>();

    DataCopyExtParams copyParamsWs;
    copyParamsWs.blockCount = 1;
    copyParamsWs.blockLen = calcNum * sizeof(float);
    copyParamsWs.srcStride = 0;
    copyParamsWs.dstStride = 0;
    DataCopyPadExtParams<float> padWs{false, 0, 0, 0};
    DataCopyPad(fp32Ub, workspaceGm[gmOffset], copyParamsWs, padWs);

    wsQue.EnQue(fp32Ub);
}

__aicore__ inline void ComputeCast(uint64_t calcNum) {
    LocalTensor<float> fp32Ub = wsQue.DeQue<float>();
    LocalTensor<TY> b16Ub = yQue.AllocTensor<TY>();
    if constexpr (is_same<TY, half>::value) {
        Cast(b16Ub, fp32Ub, RoundMode::CAST_NONE, calcNum);  // 也可以只cast valid
    } else if constexpr (is_same<TY, bfloat16_t>::value) {
        Cast(b16Ub, fp32Ub, RoundMode::CAST_RINT, calcNum);
    }
    wsQue.FreeTensor(fp32Ub);
    yQue.EnQue(b16Ub);
}

__aicore__ inline void CopyOutCast(uint64_t gmOffset, uint64_t calcNum) {
    LocalTensor<TY> yUb = yQue.DeQue<TY>();
    DataCopyExtParams copyParamsY;
    copyParamsY.blockCount = 1;
    copyParamsY.blockLen = calcNum * sizeof(TY);
    copyParamsY.srcStride = 0;
    copyParamsY.dstStride = 0;
    DataCopyPad(yGm[gmOffset], yUb, copyParamsY);
    yQue.FreeTensor(yUb);
}

public:
    TPipe *pipe_ = nullptr;
    BlockParams block_;
    BlockParams core_;
    TilingParams para_;
    OffsetParams ubOffset_;

    GlobalTensor<TX> xGm;
    GlobalTensor<TGrad> gradGm;
    GlobalTensor<TArgmax> argmaxGm;
    GlobalTensor<TY> yGm;
    GlobalTensor<float> workspaceGm;

    TQue<QuePosition::VECIN, 1> wsQue;
    TQue<QuePosition::VECOUT, 1> yQue;

    TBuf<TPosition::VECCALC> totalBuf;

    uint64_t argmaxAddrList_[MAX_LIST_NUM][TRANS_ADDR_LEN];
    uint64_t argmaxTranAddrList_[MAX_LIST_NUM][TRANS_ADDR_LEN];
    uint64_t gradAddrList_[MAX_LIST_NUM][TRANS_ADDR_LEN];
    uint64_t gradTranAddrList_[MAX_LIST_NUM][TRANS_ADDR_LEN];
    uint64_t yAddrList_[MAX_LIST_NUM][TRANS_ADDR_LEN];
    uint64_t yTranAddrList_[MAX_LIST_NUM][TRANS_ADDR_LEN];
};
}

#endif  // MAX_POOL_GRAD3D_WITH_ARGMAX_NORMAL_H
