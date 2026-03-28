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
 * \file max_pool3d_grad_with_argmax_cutk_base.h
 * \brief
 */
#ifndef MAX_POOL_GRAD3D_WITH_ARGMAX_CUTK_BASE
#define MAX_POOL_GRAD3D_WITH_ARGMAX_CUTK_BASE

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "max_pool3d_grad_with_argmax_common.h"

namespace MaxPool3DGradWithArgmax {
using namespace AscendC;
using namespace MaxPool3DGradWithArgmaxComm;

constexpr uint64_t VL_FP32_BLOCK = 8;
constexpr uint64_t RESERVED_UB_SIZE = 256;
constexpr uint64_t BINARY_MUL = 2;

struct GenIndicesParams{
    uint64_t dCount = 0;
    uint64_t colCount = 0;
    uint64_t rowCount = 0;

    uint64_t firstValue = 0;
    uint64_t increaseWValue = 1;
    uint64_t increaseHValue = 0;
    uint64_t increaseDValue = 0;
    uint64_t vlValue = 64;
};

struct DepadParams{
    uint64_t hiValid = 0;
    uint64_t hiStartOffset = 0;
    uint64_t padHStartOffset = 0;
    uint64_t padHEndOffset = 0;

    uint64_t wiValid = 0;
    uint64_t wiStartOffset = 0;
    uint64_t padWStartOffset = 0;
    uint64_t padWEndOffset = 0;
    uint64_t wiValidAlign = 0;

    uint64_t diValid = 0;
    uint64_t diStartOffset = 0;
    uint64_t padDStartOffset = 0;
    uint64_t padDEndOffset = 0;
};

template <typename T>
__aicore__ inline T CeilAlign(T x, T y) {
    return y == 0 ? x : (x + y - 1) / y * y;
}


template <typename TX, typename TGrad, typename TArgmax, typename TY, bool isOverlap>
class MaxPool3DGradWithArgmaxCutKBase {
public:
__aicore__ inline MaxPool3DGradWithArgmaxCutKBase() {}

__aicore__ inline void InitParams(const MaxPool3DGradWithArgmaxTilingData* tiling)
{   
    params_.ncDim = tiling->ncDim;
    params_.diDim = tiling->diDim;
    params_.hiDim = tiling->hiDim;
    params_.wiDim = tiling->wiDim;
    params_.doDim = tiling->doDim;
    params_.hoDim = tiling->hoDim;
    params_.woDim = tiling->woDim;
    params_.kd = tiling->kd;
    params_.kh = tiling->kh;
    params_.kw = tiling->kw;
    params_.sd = tiling->sd;
    params_.sh = tiling->sh;
    params_.sw = tiling->sw;
    params_.padDTop = tiling->padDTop;
    params_.padHTop = tiling->padHTop;
    params_.padWTop = tiling->padWTop;
    params_.padDBottom = tiling->padDBottom;
    params_.padHBottom = tiling->padHBottom;
    params_.padWBottom = tiling->padWBottom;
    params_.baseNc = tiling->baseNc;
    params_.baseDo = tiling->baseDo;
    params_.baseHo = tiling->baseHo;
    params_.baseWo = tiling->baseWo;
    params_.ncCnt = tiling->ncCnt;
    params_.doCnt = tiling->doCnt;
    params_.hoCnt = tiling->hoCnt;
    params_.woCnt = tiling->woCnt;
    params_.totalCnt = tiling->totalCnt;
    params_.ncTail = tiling->ncTail;
    params_.doTail = tiling->doTail;
    params_.hoTail = tiling->hoTail;
    params_.woTail = tiling->woTail;
    params_.usedCoreNum = tiling->usedCoreNum;
    params_.ubSize = tiling->totalUBSize;
    params_.outputDataSize = params_.ncDim * params_.diDim * params_.hiDim * params_.wiDim;
    bool ifKSEqual = (params_.kd != params_.sd) || (params_.kh != params_.sh) || (params_.kw != params_.sw);
    bool ifRedundant = (params_.doDim * params_.sd < (params_.diDim + params_.padDTop + params_.padDBottom)) ||
                       (params_.hoDim * params_.sh < (params_.hiDim + params_.padHTop + params_.padHBottom)) ||
                       (params_.woDim * params_.sw < (params_.wiDim + params_.padWTop + params_.padWBottom));
    params_.needInitOutput = ifKSEqual || ifRedundant;
}

__aicore__ inline void InitInputsOutputs(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace)
{
    // GlobalBuffer is full, each core use offset to get real addr
    xGm.SetGlobalBuffer((__gm__ TX*)x);
    gradGm.SetGlobalBuffer((__gm__ TGrad*)grad);
    argmaxGm.SetGlobalBuffer((__gm__ TArgmax*)argmax);
    yGm.SetGlobalBuffer((__gm__ TY*)y);
    if constexpr (isOverlap && (std::is_same<TGrad, half>::value || std::is_same<TGrad, bfloat16_t>::value)) {
        workspaceGm.SetGlobalBuffer((__gm__ float*)usrWorkspace);
        if (params_.needInitOutput){
            if (GetBlockIdx() == 0) {
                InitGlobalMemory(workspaceGm, params_.outputDataSize, 0.0f);
            }
            SyncAll();
        }
    } else if (params_.needInitOutput) {
        if (GetBlockIdx() == 0) {
            InitGlobalMemory(yGm, params_.outputDataSize, static_cast<TY>(0));
        }
        SyncAll();
    }
}

__aicore__ inline void InitUbBuffer()
{
    uint64_t singleBlock = params_.baseDo * params_.baseHo * params_.baseWo;
    uint64_t singleBlockSizeAlign = CeilAlign<uint64_t>(singleBlock * sizeof(TGrad), BLOCK_SIZE);
    uint64_t singleBlockArgmaxSizeAlign = CeilAlign<uint64_t>(singleBlock * sizeof(TArgmax), BLOCK_SIZE); 
    pipe->InitBuffer(gradQue, 1, params_.baseNc * singleBlockSizeAlign);
    pipe->InitBuffer(argmaxQue, 1, params_.baseNc * singleBlockArgmaxSizeAlign);
    pipe->InitBuffer(yQue, 1, baseNDHWpAlign * sizeof(float));
    pipe->InitBuffer(yTransposeBuf, baseNDHWpAlign * sizeof(float));
    pipe->InitBuffer(yTranDepadBuf, baseNDHWpAlign * sizeof(float));
    pipe->InitBuffer(argmaxTransposeBuf, params_.baseNc * singleBlockArgmaxSizeAlign);
    pipe->InitBuffer(gradTransposeBuf, params_.baseNc * singleBlockSizeAlign);
    pipe->InitBuffer(maskBuf, params_.baseNc * singleBlock * sizeof(float) / BLOCK_SIZE);
    pipe->InitBuffer(tmpGradBuf, params_.baseNc * singleBlock * sizeof(float));
    pipe->InitBuffer(indexImgBuf, params_.baseNc * baseDp * baseHp * baseWp * sizeof(TArgmax));
    pipe->InitBuffer(indexColBuf, params_.baseNc * singleBlock * sizeof(TArgmax));
}

__aicore__ inline void GenKernelIndicesWithTranspose(const LocalTensor<TArgmax>& dstLocal,
                                                     const GenIndicesParams& genParams)
{
    const uint64_t dCount = genParams.dCount;
    const uint64_t colCount = genParams.colCount;
    const uint64_t rowCount = genParams.rowCount;
    const uint64_t vlValue = genParams.vlValue;
    const int32_t firstValue = static_cast<int32_t>(genParams.firstValue);
    const int32_t increaseWValue = static_cast<int32_t>(genParams.increaseWValue);
    const int32_t increaseHValue = static_cast<int32_t>(genParams.increaseHValue);
    const int32_t increaseDValue = static_cast<int32_t>(genParams.increaseDValue);

    // 1. Dup (VL), full of firstValue
    Duplicate(dstLocal, firstValue, vlValue);
    AscendC::PipeBarrier<PIPE_V>();

    // 2. Adds to (countAligned, VL)
    // Binary split in the w direction
    int32_t i = 1;
    int32_t ivIndex = 1;
    for (; i * BINARY_MUL <= colCount; i = i * BINARY_MUL) {
        Adds(dstLocal[i * vlValue], dstLocal[0], increaseWValue * ivIndex, i * vlValue);
        ivIndex *= BINARY_MUL;
        AscendC::PipeBarrier<PIPE_V>();
    }
    
    if (i != colCount) {
        Adds(dstLocal[i * vlValue], dstLocal[0], increaseWValue * ivIndex, (colCount-i) * vlValue);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // Binary split in the h direction
    int32_t j = 1;
    int32_t jvIndex = 1;
    for (; j * BINARY_MUL <= rowCount; j = j * BINARY_MUL) {
        Adds(dstLocal[j * colCount * vlValue], dstLocal[0], increaseHValue * jvIndex, j * vlValue * colCount);
        jvIndex *= BINARY_MUL;
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (j != rowCount) {
        Adds(dstLocal[j * colCount * vlValue], dstLocal[0], increaseHValue * jvIndex,
             (rowCount-j) * vlValue * colCount);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::PipeBarrier<PIPE_V>();

    // Binary split in the d direction
    int32_t k = 1;
    int32_t kvIndex = 1;
    for (; k * BINARY_MUL <= dCount; k = k * BINARY_MUL) {
        Adds(dstLocal[k * rowCount * colCount * vlValue], dstLocal[0], increaseDValue * kvIndex,
             k * vlValue * colCount * rowCount);
        kvIndex *= BINARY_MUL;
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (k != dCount) {
        Adds(dstLocal[k * rowCount * colCount * vlValue], dstLocal[0], increaseDValue * kvIndex,
             (dCount-k) * vlValue * colCount * rowCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CopyInArgmax()
{
    LocalTensor<TArgmax> argmaxUb = argmaxQue.AllocTensor<TArgmax>();

    DataCopyExtParams copyParamsArgmax;
    copyParamsArgmax.blockCount = block_.ncShape;
    copyParamsArgmax.blockLen = block_.dohowoShape * sizeof(TArgmax);
    copyParamsArgmax.srcStride =
        (params_.doDim * params_.hoDim * params_.woDim - block_.dohowoShape) * sizeof(TArgmax);
    copyParamsArgmax.dstStride = 0;
    DataCopyPadExtParams<TArgmax> padArgmax{false, 0, 0, 0};
    DataCopyPad(argmaxUb, argmaxGm[block_.offsetArgmax], copyParamsArgmax, padArgmax);

    argmaxQue.EnQue(argmaxUb);
}

__aicore__ inline void CopyInGrad()
{
    LocalTensor<TGrad> gradUb = gradQue.AllocTensor<TGrad>();

    DataCopyExtParams copyParamsGrad;
    copyParamsGrad.blockCount = block_.ncShape;
    copyParamsGrad.blockLen = block_.dohowoShape * sizeof(TGrad);
    copyParamsGrad.srcStride = (params_.doDim * params_.hoDim * params_.woDim - block_.dohowoShape) * sizeof(TGrad);
    copyParamsGrad.dstStride = 0;
    DataCopyPadExtParams<TGrad> padGrad{false, 0, 0, 0};
    DataCopyPad(gradUb, gradGm[block_.offsetGrad], copyParamsGrad, padGrad);

    gradQue.EnQue(gradUb);
}

__aicore__ inline void CopyInData(LocalTensor<TArgmax> argmaxTranUb, LocalTensor<TGrad> gradTranUb) {
    this->CopyInArgmax();
    LocalTensor<TArgmax> argmaxUb = this->argmaxQue.template DeQue<TArgmax>();
    this->CopyInGrad();
    LocalTensor<TGrad> gradUb = this->gradQue.template DeQue<TGrad>();
    int32_t eventIdMte2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    TransposeBase16M8<int32_t>(argmaxTranUb, argmaxUb, this->params_.baseNc, this->block_.dohowoAlign8);
    if constexpr (std::is_same<TGrad, half>::value || std::is_same<TGrad, bfloat16_t>::value) {
        TransposeBase16M16<TGrad>(gradTranUb, gradUb, this->params_.baseNc, this->block_.dohowoAlign16);
    } else {
        TransposeBase16M8<TGrad>(gradTranUb, gradUb, this->params_.baseNc, this->block_.dohowoAlign8);
    }
    this->argmaxQue.FreeTensor(argmaxUb);
    this->gradQue.FreeTensor(gradUb);
}

__aicore__ inline void CalBlockParams(uint64_t index)
{
    block_.doCntIndex = index % (params_.doCnt * params_.hoCnt * params_.woCnt) / (params_.hoCnt * params_.woCnt);
    block_.hoCntIndex = index % (params_.hoCnt * params_.woCnt) / params_.woCnt;
    block_.woCntIndex = index % params_.woCnt;
    block_.ncShape = block_.ncCntIndex >= (params_.ncCnt - 1) ? params_.ncTail : params_.baseNc;
    block_.doShape = block_.doCntIndex >= (params_.doCnt - 1) ? params_.doTail : params_.baseDo;
    block_.hoShape = block_.hoCntIndex >= (params_.hoCnt - 1) ? params_.hoTail : params_.baseHo;
    block_.woShape = block_.woCntIndex >= (params_.woCnt - 1) ? params_.woTail : params_.baseWo;
    block_.diShape = (block_.doShape - 1) * params_.sd + params_.kd;
    block_.hiShape = (block_.hoShape - 1) * params_.sh + params_.kh;
    block_.wiShape = (block_.woShape - 1) * params_.sw + params_.kw;
    block_.dohowoShape = block_.doShape * block_.hoShape * block_.woShape;
    block_.dohowoAlign8 = (block_.doShape * block_.hoShape * block_.woShape + BLOCK_NUM_32 - 1) /
                          BLOCK_NUM_32 * BLOCK_NUM_32;
    block_.dohowoAlign16 = (block_.doShape * block_.hoShape * block_.woShape + BLOCK_NUM_16 - 1) /
                           BLOCK_NUM_16 * BLOCK_NUM_16;
    block_.offsetGrad = block_.ncCntIndex * params_.baseNc * params_.doDim * params_.hoDim * params_.woDim +
                        block_.doCntIndex * params_.baseDo * params_.hoDim  * params_.woDim +
                        block_.hoCntIndex * params_.baseHo * params_.woDim +
                        block_.woCntIndex * params_.baseWo;
    block_.offsetArgmax = block_.ncCntIndex * params_.baseNc * params_.doDim * params_.hoDim * params_.woDim +
                        block_.doCntIndex * params_.baseDo * params_.hoDim  * params_.woDim +
                        block_.hoCntIndex * params_.baseHo * params_.woDim +
                        block_.woCntIndex * params_.baseWo;
    block_.offsetY = block_.ncCntIndex * params_.doDim * params_.hiDim * params_.wiDim +
                     ((block_.doCntIndex * params_.baseDo) * params_.sd - params_.padDTop)
                     * params_.hiDim * params_.wiDim +
                     ((block_.hoCntIndex * params_.baseHo) * params_.sh - params_.padHTop) * params_.wiDim +
                     (block_.woCntIndex * params_.baseWo) * params_.sw - params_.padWTop;
    baseDataNum = params_.baseNc * block_.doShape * block_.hoShape * block_.woShape;
}

__aicore__ inline void InitCastUbBuffer()
{
    pipe->Reset();
    uint64_t singleBlockDataNum = params_.outputDataSize / params_.usedCoreNum;
    uint64_t tailBlockDataNum = params_.outputDataSize - singleBlockDataNum * (params_.usedCoreNum - 1);
    uint64_t curBlockData = GetBlockIdx() < (params_.usedCoreNum - 1) ? singleBlockDataNum : tailBlockDataNum;
    yCastNum = (params_.ubSize - RESERVED_UB_SIZE) / sizeof(float);
    yCastNumTail = curBlockData % yCastNum;
    carryLoop = CeilDiv(curBlockData, yCastNum);
    blockOffset = GetBlockIdx() * singleBlockDataNum;
    pipe->InitBuffer(yCastQue, 1, yCastNum * sizeof(float));
}

__aicore__ inline void CopyHalfOut()
{
    LocalTensor<float> yCastUb = yCastQue.AllocTensor<float>();
    DataCopyPadExtParams<float> padParams {false, 0, 0, 0.0f};
    for (uint64_t loopIdx = 0; loopIdx < carryLoop; loopIdx++) {
        uint64_t curCastNum = loopIdx < (carryLoop - 1) ? yCastNum : yCastNumTail;
        DataCopyExtParams copyParams {1,
                                      static_cast<uint32_t>(curCastNum * sizeof(float)),
                                      0,
                                      0,
                                      0};
        DataCopyPad(yCastUb, workspaceGm[blockOffset + loopIdx * yCastNum], copyParams, padParams);
        int32_t eventIdMte2ToV = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        if constexpr (std::is_same<TGrad, bfloat16_t>::value) {
            Cast(yCastUb.ReinterpretCast<TY>(), yCastUb, RoundMode::CAST_RINT, yCastNum);
        } else if constexpr (std::is_same<TGrad, half>::value) {
            Cast(yCastUb.ReinterpretCast<TY>(), yCastUb, RoundMode::CAST_NONE, yCastNum);
        }
        int32_t eventIdVToMte3 = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        copyParams = {1, static_cast<uint32_t>(curCastNum * sizeof(half)), 0, 0, 0};
        DataCopyPad(yGm[blockOffset + loopIdx * yCastNum], yCastUb.ReinterpretCast<TY>(), copyParams);
        int32_t eventIdMte3ToMte2 = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }
    yCastQue.FreeTensor(yCastUb);
}

__aicore__ inline void  SelectGradOut(const LocalTensor<TGrad>& gradTranUb,
                                      const LocalTensor<TArgmax>& argmaxTranUb,
                                      const LocalTensor<TArgmax>& indexColUb,
                                      LocalTensor<float>& tmpGradUb)
{
    LocalTensor<uint8_t> maskUb = maskBuf.Get<uint8_t>();
    Compare(maskUb, indexColUb, argmaxTranUb, CMPMODE::EQ, baseDataNum);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<TGrad, bfloat16_t>::value) {
        Select(tmpGradUb.template ReinterpretCast<half>()[baseDataNum], maskUb,
               gradTranUb.template ReinterpretCast<half>(), static_cast<half>(ZERO),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, baseDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tmpGradUb, tmpGradUb.template ReinterpretCast<bfloat16_t>()[baseDataNum], RoundMode::CAST_NONE,
             baseDataNum);
    } else if constexpr (std::is_same<TGrad, half>::value) {
        Select(tmpGradUb.template ReinterpretCast<half>()[baseDataNum], maskUb, gradTranUb, static_cast<half>(ZERO),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, baseDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tmpGradUb, tmpGradUb.ReinterpretCast<half>()[baseDataNum], RoundMode::CAST_NONE, baseDataNum);
    } else if constexpr (std::is_same<TGrad, float>::value) {
        Select(tmpGradUb, maskUb, gradTranUb, ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, baseDataNum);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CalcDepadParamsD(uint64_t kdIdx) {
    uint64_t dpStartOffset = block_.doCntIndex * params_.baseDo * params_.sd + kdIdx; 
    uint64_t dpEndOffset = block_.doCntIndex * params_.baseDo * params_.sd +
                           (block_.doShape - 1) * params_.sd + kdIdx;
    if (dpStartOffset >= params_.padDTop && dpEndOffset < (params_.diDim + params_.padDTop)) {
        // All data in dp is valid
        depad_.padDStartOffset = 0;
        depad_.padDEndOffset = block_.doShape - 1;
    } else if (dpStartOffset < params_.padDTop && dpEndOffset < (params_.diDim + params_.padDTop)) {
        // The left part of the data in dp should be removed
        depad_.padDStartOffset = (params_.padDTop - 1 - dpStartOffset) / params_.sd + 1;
        depad_.padDEndOffset = block_.doShape - 1;
    } else if (dpStartOffset < params_.padDTop && dpEndOffset >= (params_.diDim + params_.padDTop)) {
        // The left and right part of the data in dp should be removed
        depad_.padDStartOffset = (params_.padDTop - 1 - dpStartOffset) / params_.sd + 1;
        // Consider distinguishing whether dp is entirely in the right padding scenario
        depad_.padDEndOffset = dpStartOffset < (params_.padDTop + params_.diDim) ?
                               (params_.padDTop + params_.diDim - 1 - dpStartOffset) / params_.sd : 0;
    } else {
        // The right part of the data in dp should be removed
        depad_.padDStartOffset = 0;
        depad_.padDEndOffset = dpStartOffset < (params_.padDTop + params_.diDim) ?
                               (params_.padDTop + params_.diDim - 1 - dpStartOffset) / params_.sd : 0;
    }
    depad_.diStartOffset = (depad_.padDStartOffset + block_.doCntIndex * params_.baseDo) * params_.sd + kdIdx -
                           params_.padDTop;
    // Consider distinguishing whether dp is entirely in the left padding scenario
    depad_.diValid = depad_.padDEndOffset >= depad_.padDStartOffset ?
                     depad_.padDEndOffset + 1 - depad_.padDStartOffset : 0;
    depad_.diValid = depad_.diStartOffset > (params_.diDim - 1) ? 0 : depad_.diValid;
}

__aicore__ inline void CalcDepadParamsH(uint64_t khIdx) {
    uint64_t hpStartOffset = block_.hoCntIndex * params_.baseHo * params_.sh + khIdx;
    uint64_t hpEndOffset = block_.hoCntIndex * params_.baseHo * params_.sh +
                           (block_.hoShape - 1) * params_.sh + khIdx;
    if (hpStartOffset >= params_.padHTop && hpEndOffset < (params_.hiDim + params_.padHTop)) {
        depad_.padHStartOffset = 0; 
        depad_.padHEndOffset = block_.hoShape - 1;
    } else if (hpStartOffset < params_.padHTop && hpEndOffset < (params_.hiDim + params_.padHTop)) {
        depad_.padHStartOffset = (params_.padHTop - 1 - hpStartOffset) / params_.sh + 1;
        depad_.padHEndOffset = block_.hoShape - 1;
    } else if (hpStartOffset < params_.padHTop && hpEndOffset >= (params_.hiDim + params_.padHTop)) {
        depad_.padHStartOffset = (params_.padHTop - 1 - hpStartOffset) / params_.sh + 1;
        depad_.padHEndOffset = hpStartOffset < (params_.padHTop + params_.hiDim) ?
                               (params_.padHTop + params_.hiDim - 1 - hpStartOffset) / params_.sh : 0;
    } else {
        depad_.padHStartOffset = 0;
        depad_.padHEndOffset = hpStartOffset < (params_.padHTop + params_.hiDim) ?
                               (params_.padHTop + params_.hiDim - 1 - hpStartOffset) / params_.sh : 0;
    }
    depad_.hiStartOffset = (depad_.padHStartOffset + block_.hoCntIndex * params_.baseHo) * params_.sh + khIdx -
                           params_.padHTop;
    depad_.hiValid = depad_.padHEndOffset >= depad_.padHStartOffset ?
                     depad_.padHEndOffset + 1 - depad_.padHStartOffset : 0;
    depad_.hiValid = depad_.hiStartOffset > (params_.hiDim - 1) ? 0 : depad_.hiValid;
}

__aicore__ inline void CalcDepadParamsW(uint64_t kwIdx)
{
    uint64_t wpStartOffset = block_.woCntIndex * params_.baseWo * params_.sw + kwIdx;
    uint64_t wpEndOffset = block_.woCntIndex * params_.baseWo * params_.sw +
                           (block_.woShape - 1) * params_.sw + kwIdx;
    if ((wpStartOffset >= params_.padWTop) && (wpEndOffset < params_.wiDim + params_.padWTop)) {
        depad_.padWStartOffset = 0;
        depad_.padWEndOffset = block_.woShape - 1;
    } else if (wpStartOffset < params_.padWTop && wpEndOffset < (params_.wiDim + params_.padWTop)) {
        depad_.padWStartOffset = (params_.padWTop - 1 - wpStartOffset) / params_.sw + 1;
        depad_.padWEndOffset = block_.woShape - 1;
    } else if (wpStartOffset < params_.padWTop && wpEndOffset >= (params_.wiDim + params_.padWTop)) {
        depad_.padWStartOffset = (params_.padWTop - 1 - wpStartOffset) / params_.sw + 1;
        depad_.padWEndOffset = wpStartOffset < (params_.padWTop + params_.wiDim) ?
                               (params_.padWTop + params_.wiDim - 1 - wpStartOffset) / params_.sw : 0;
    } else {
        depad_.padWStartOffset = 0;
        depad_.padWEndOffset = wpStartOffset < (params_.padWTop + params_.wiDim) ?
                               (params_.padWTop + params_.wiDim - 1 - wpStartOffset) / params_.sw : 0;
    }
    depad_.wiStartOffset = (depad_.padWStartOffset + block_.woCntIndex * params_.baseWo) * params_.sw + kwIdx -
                           params_.padWTop;
    depad_.wiValid = depad_.padWEndOffset >= depad_.padWStartOffset ?
                     depad_.padWEndOffset + 1 - depad_.padWStartOffset : 0;
    depad_.wiValid = depad_.wiStartOffset > (params_.wiDim - 1) ? 0 : depad_.wiValid;
    depad_.wiValidAlign =
        (!isOverlap && (std::is_same<TGrad, half>::value || std::is_same<TGrad, bfloat16_t>::value)) ?
        depad_.wiValid * BLOCK_NUM_16 : depad_.wiValid * BLOCK_NUM_32;
}

__aicore__ inline void CalcDepadParamsH() {
    uint64_t hpStartOffset = block_.hoCntIndex * params_.baseHo * params_.sh;
    uint64_t hpEndOffset = block_.hoCntIndex * params_.baseHo * params_.sh +
                           (block_.hoShape - 1) * params_.sh + params_.kh - 1;
    if (hpStartOffset >= params_.padHTop && hpEndOffset < (params_.hiDim + params_.padHTop)) {
        depad_.hiValid = block_.hiShape;
        depad_.hiStartOffset = block_.hoCntIndex * params_.baseHo * params_.sh - params_.padHTop;
        depad_.padHStartOffset = 0;
    } else if (hpStartOffset < params_.padHTop && hpEndOffset < (params_.hiDim + params_.padHTop)) {
        depad_.hiValid = block_.hiShape - (params_.padHTop - hpStartOffset);
        depad_.hiStartOffset = 0;
        depad_.padHStartOffset = params_.padHTop - hpStartOffset;
    } else if (hpStartOffset < params_.padHTop && hpEndOffset >= (params_.hiDim + params_.padHTop)) {
        depad_.hiValid = params_.hiDim;
        depad_.hiStartOffset = 0;
        depad_.padHStartOffset = params_.padHTop - hpStartOffset;
    } else {
        depad_.hiValid = block_.hiShape - (hpEndOffset + 1 - (params_.hiDim + params_.padHTop));
        depad_.hiStartOffset = block_.hoCntIndex * params_.baseHo * params_.sh - params_.padHTop;
        depad_.padHStartOffset = 0;
    }
    depad_.padHEndOffset = depad_.padHStartOffset + depad_.hiValid - 1;
}

__aicore__ inline void CalcDepadParamsW()
{
    uint64_t wpStartOffset = block_.woCntIndex * params_.baseWo * params_.sw;
    uint64_t wpEndOffset = block_.woCntIndex * params_.baseWo * params_.sw +
                           (block_.woShape - 1) * params_.sw + params_.kw - 1;
    if ((wpStartOffset >= params_.padWTop) && (wpEndOffset < params_.wiDim + params_.padWTop)) {
        depad_.wiValid = block_.wiShape;
        depad_.wiStartOffset = block_.woCntIndex * params_.baseWo * params_.sw - params_.padWTop;
        depad_.padWStartOffset = 0;
    } else if (wpStartOffset < params_.padWTop && wpEndOffset < (params_.wiDim + params_.padWTop)) {
        depad_.wiValid = block_.wiShape - (params_.padWTop - wpStartOffset);
        depad_.wiStartOffset = 0;
        depad_.padWStartOffset = params_.padWTop - wpStartOffset;
    } else if (wpStartOffset < params_.padWTop && wpEndOffset >= (params_.wiDim + params_.padWTop)) {
        depad_.wiValid = params_.wiDim;
        depad_.wiStartOffset = 0;
        depad_.padWStartOffset = params_.padWTop - wpStartOffset;
    } else {
        depad_.wiValid = block_.wiShape - (wpEndOffset + 1 - (params_.wiDim + params_.padWTop));
        depad_.wiStartOffset = block_.woCntIndex * params_.baseWo * params_.sw - params_.padWTop;
        depad_.padWStartOffset = 0;
    }
    depad_.padWEndOffset = depad_.padWStartOffset + depad_.wiValid - 1;
    depad_.wiValidAlign = 
        (!isOverlap && (std::is_same<TGrad, half>::value || std::is_same<TGrad, bfloat16_t>::value)) ?
        CeilAlign<uint64_t>(depad_.wiValid, BLOCK_NUM_16) :
        CeilAlign<uint64_t>(depad_.wiValid, BLOCK_NUM_32);
}

__aicore__ inline void CastAndCarryOut() {
    if constexpr (isOverlap && (std::is_same<TGrad, half>::value || std::is_same<TGrad, bfloat16_t>::value)) {
        SyncAll();
        InitCastUbBuffer();
        CopyHalfOut();
    }
}

public:
    BlockParams block_;
    TilingParams params_;
    DepadParams depad_;
    TPipe *pipe;

    GlobalTensor<TX> xGm;
    GlobalTensor<TGrad> gradGm;
    GlobalTensor<TArgmax> argmaxGm;
    GlobalTensor<TY> yGm;
    GlobalTensor<float> workspaceGm;

    TQue<QuePosition::VECIN, 1> gradQue;
    TQue<QuePosition::VECIN, 1> argmaxQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECIN, 1> yCastQue;

    TBuf<TPosition::VECCALC> argmaxTransposeBuf;
    TBuf<TPosition::VECCALC> gradTransposeBuf;
    TBuf<TPosition::VECCALC> yTransposeBuf;
    TBuf<TPosition::VECCALC> yTranDepadBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> tmpGradBuf;
    TBuf<TPosition::VECCALC> indexImgBuf;
    TBuf<TPosition::VECCALC> indexColBuf;

    uint64_t yCastUbSize;
    uint64_t yCastUbSizeTail;
    uint64_t carryLoop;
    uint64_t blockOffset; // workspace offset

    uint64_t baseDp;
    uint64_t baseHp;
    uint64_t baseWp;
    uint64_t baseWp8Align;
    uint64_t baseNDHWpAlign;
    uint64_t baseDataNum;
    uint64_t yCastNum;
    uint64_t yCastNumTail;
};
}

#endif // MAX_POOL_GRAD3D_WITH_ARGMAX_CUTK_BASE