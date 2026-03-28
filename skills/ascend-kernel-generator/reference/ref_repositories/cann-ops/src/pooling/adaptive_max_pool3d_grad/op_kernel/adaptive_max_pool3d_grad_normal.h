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
 * \file adaptive_max_pool3d_grad_normal.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_NORMAL_H
#define ADAPTIVE_MAX_POOL3D_GRAD_NORMAL_H
 
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "adaptive_max_pool3d_grad_common.h"
 
namespace AdaptiveMaxPool3DGrad {
using namespace AscendC;
using namespace AdaptiveMaxPool3DGradComm;
 
template <typename TX, typename TGrad, typename TArgmax, typename TY, bool IsOverlap>
class AdaptiveMaxPool3DGradNormal {
public:
__aicore__ inline AdaptiveMaxPool3DGradNormal(TPipe* Pipe) {
    pipe = Pipe;
}
__aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace,
                            const AdaptiveMaxPool3DGradTilingData *tiling)
{
    InitParams(tiling);
    InitInputsOutputs(x, grad, argmax, y, usrWorkspace);
    InitUbBuffer();
}
 
__aicore__ inline void InitParams(const AdaptiveMaxPool3DGradTilingData *tiling)
{
    params_.ncDim = tiling->ncDim;
    params_.diDim = tiling->diDim;
    params_.hiDim = tiling->hiDim;
    params_.wiDim = tiling->wiDim;
    params_.doDim = tiling->doDim;
    params_.hoDim = tiling->hoDim;
    params_.woDim = tiling->woDim;
    params_.singleCoreNc = tiling->singleCoreNc;
    params_.singleCoreDo = tiling->singleCoreDo;
    params_.singleCoreHo = tiling->singleCoreHo;
    params_.singleCoreWo = tiling->singleCoreWo;
    params_.baseNc = tiling->baseNc;
    params_.baseDo = tiling->baseDo;
    params_.baseHo = tiling->baseHo;
    params_.baseWo = tiling->baseWo;
    params_.ncCnt = tiling->ncCnt;
    params_.dCnt = tiling->doCnt;
    params_.hCnt = tiling->hoCnt;
    params_.wCnt = tiling->woCnt;
    params_.totalCnt = tiling->totalCnt;
    params_.baseNcTail = tiling->ncTail;
    params_.doTail = tiling->doTail;
    params_.hoTail = tiling->hoTail;
    params_.woTail = tiling->woTail;
    params_.usedCoreNum = tiling->usedCoreNum;
    params_.maxKd = tiling->kdMax;
    params_.maxKh = tiling->khMax;
    params_.maxKw = tiling->kwMax;  
    params_.ubSize = tiling->totalUBSize;  
    params_.diHiWiLen = params_.diDim * params_.hiDim * params_.wiDim;
    params_.maxKdhwLen = params_.maxKd * params_.maxKh * params_.maxKw;
}
 
__aicore__ inline void InitInputsOutputs(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace)
{
    gradGm.SetGlobalBuffer((__gm__ TGrad*)grad);
    argmaxGm.SetGlobalBuffer((__gm__ TArgmax*)argmax);
    yGm.SetGlobalBuffer((__gm__ TY*)y);
    workspaceGm.SetGlobalBuffer((__gm__ float*)usrWorkspace);
    if (GetBlockIdx() == 0) {
        if constexpr (is_same<TY, float>::value) {
            InitGlobalMemory(yGm, params_.ncDim * params_.diHiWiLen, 0.0f);
        } else {
            if constexpr (IsOverlap) {
                InitGlobalMemory(workspaceGm, params_.ncDim * params_.diHiWiLen, 0.0f);
            } else {
                InitGlobalMemory(yGm, params_.ncDim * params_.diHiWiLen, (TY)0);
            }
        }
    }
    SyncAll();
}
 
__aicore__ inline void InitUbBuffer()
{
    uint64_t baseDoHoWo = params_.baseDo * params_.baseHo * params_.baseWo;
    uint64_t baseDoHoWoAlign8 = AlignUp(baseDoHoWo, BLOCK_NUM_32);
    uint64_t blockNumDtype = BLOCK_SIZE / sizeof(TGrad);
    uint64_t baseDoHoWoAlignDtype = AlignUp(baseDoHoWo, blockNumDtype);
 
    pipe->InitBuffer(gradQue, 1, params_.singleCoreNc * baseDoHoWoAlignDtype * sizeof(TGrad));
    pipe->InitBuffer(gradTransposeBuf, params_.singleCoreNc * baseDoHoWoAlignDtype * sizeof(TGrad));

    pipe->InitBuffer(indicesQue, 1, params_.singleCoreNc * baseDoHoWoAlign8 * sizeof(int32_t));
    pipe->InitBuffer(indicesTransposeBuf, params_.singleCoreNc * baseDoHoWoAlign8 * sizeof(int32_t));
    pipe->InitBuffer(indicesFloatBuf, params_.singleCoreNc * baseDoHoWoAlign8 * sizeof(float));
    pipe->InitBuffer(indicesDBuf, params_.singleCoreNc * baseDoHoWoAlign8 * sizeof(float));
    pipe->InitBuffer(indicesHBuf, params_.singleCoreNc * baseDoHoWoAlign8 * sizeof(float));
    pipe->InitBuffer(indicesWBuf, params_.singleCoreNc * baseDoHoWoAlign8 * sizeof(float));
    pipe->InitBuffer(tempBuf, params_.singleCoreNc * baseDoHoWoAlign8 * sizeof(float));
    pipe->InitBuffer(kernelIdxBuf, params_.singleCoreNc * params_.maxKdhwLen * sizeof(float));
    pipe->InitBuffer(maskBuf, params_.singleCoreNc * params_.maxKdhwLen / UINT8_BITS);  
    pipe->InitBuffer(tempGradBuf, params_.singleCoreNc * params_.maxKdhwLen * sizeof(float));

    if constexpr (IsOverlap) {
        uint64_t maxWAlign8 = AlignUp(params_.maxKw, BLOCK_NUM_32);
        pipe->InitBuffer(yQue, 1, params_.singleCoreNc * params_.maxKd * params_.maxKh * maxWAlign8 * sizeof(float)); 
        pipe->InitBuffer(yTransposeBuf, params_.singleCoreNc * params_.maxKd * params_.maxKh * maxWAlign8 * sizeof(float));
    } else {
        uint64_t maxWAlignDtype = AlignUp(params_.maxKw, blockNumDtype);
        pipe->InitBuffer(yQue, 1, params_.singleCoreNc * params_.maxKd * params_.maxKh * maxWAlignDtype * sizeof(TGrad)); 
        pipe->InitBuffer(yTransposeBuf, params_.singleCoreNc * params_.maxKd * params_.maxKh * maxWAlignDtype * sizeof(TGrad));
    }
}
 
__aicore__ inline void Process()
{
    LocalTensor<float> kernelIdx = kernelIdxBuf.Get<float>();
    GenkernelIndex(kernelIdx);  
    AscendC::PipeBarrier<PIPE_V>();
    core_.coeffD = 1.f * params_.hiDim * params_.wiDim;
    core_.coeffH = 1.f * params_.wiDim;
    core_.hwDims = 1.f * params_.hiDim * params_.wiDim;
    
    for (uint64_t totalIndex = 0; totalIndex < params_.totalCnt; totalIndex++) {
        if (GetBlockIdx() == totalIndex % GetBlockNum()) {
            core_.ncCntIndex = totalIndex / (params_.dCnt * params_.hCnt * params_.wCnt);
            core_.doCntIndex = totalIndex / (params_.hCnt * params_.wCnt) % params_.dCnt;
            core_.hoCntIndex = totalIndex / params_.wCnt % params_.hCnt;
            core_.woCntIndex = totalIndex % params_.wCnt;
            core_.ncShape = core_.ncCntIndex == params_.ncCnt - 1 ? params_.baseNcTail : params_.singleCoreNc;
            core_.doShape = core_.doCntIndex == params_.dCnt - 1 ? params_.doTail : params_.singleCoreDo;
            core_.hoShape = core_.hoCntIndex == params_.hCnt - 1 ? params_.hoTail : params_.singleCoreHo;
            core_.woShape = core_.woCntIndex == params_.wCnt - 1 ? params_.woTail : params_.singleCoreWo;
            SubProcess();
        }
    }
    if constexpr (!is_same<TY, float>::value && IsOverlap) {
        SyncAll();
        InitCastUbBuffer();
        ProcessCast();
    }
}
 
__aicore__ inline void SubProcess()
{
    uint64_t ncCoreIdx = core_.ncCntIndex * params_.singleCoreNc;
    uint64_t doCoreIdx = core_.doCntIndex * params_.singleCoreDo;
    uint64_t hoCoreIdx = core_.hoCntIndex * params_.singleCoreHo;
    uint64_t woCoreIdx = core_.woCntIndex * params_.singleCoreWo;
    uint64_t incoreDCnt = CeilDiv(core_.doShape, params_.baseDo);  
    uint64_t incoreHCnt = CeilDiv(core_.hoShape, params_.baseHo);  
    uint64_t incoreWCnt = CeilDiv(core_.woShape, params_.baseWo);
    block_.ncShape = params_.singleCoreNc;
    block_.doShape = params_.baseDo;
    block_.hoShape = params_.baseHo;
    for (uint64_t doCntIndex = 0; doCntIndex < incoreDCnt; doCntIndex++) {
        uint64_t doBlockIdx = doCoreIdx + doCntIndex;
        for (uint64_t hoCntIndex = 0; hoCntIndex < incoreHCnt; hoCntIndex++) {
            uint64_t hoBlockIdx = hoCoreIdx + hoCntIndex;
            block_.woShape = params_.baseWo;
            block_.dohowoShape = block_.doShape * block_.hoShape * block_.woShape;
            block_.dohowoAlign8 = AlignUp(block_.dohowoShape, BLOCK_NUM_32);
            block_.dohowoAlign16 = AlignUp(block_.dohowoShape, BLOCK_NUM_16);
            for (uint64_t woCntIndex = 0; woCntIndex < incoreWCnt; woCntIndex++) {
                uint64_t woBlockIdx = woCoreIdx + woCntIndex * block_.woShape;
                if (woCntIndex == incoreWCnt - 1) {
                    block_.woShape = core_.woShape - woCntIndex * params_.baseWo;
                    block_.dohowoShape = block_.doShape * block_.hoShape * block_.woShape;
                    block_.dohowoAlign8 = AlignUp(block_.dohowoShape, BLOCK_NUM_32); 
                    block_.dohowoAlign16 = AlignUp(block_.dohowoShape, BLOCK_NUM_16);
                }
                block_.startD = FloorDiv(doBlockIdx * params_.diDim, params_.doDim);
                block_.startH = FloorDiv(hoBlockIdx * params_.hiDim, params_.hoDim);
                block_.deltaD = CeilDiv((doBlockIdx + 1) * params_.diDim, params_.doDim) - block_.startD;
                block_.deltaH = CeilDiv((hoBlockIdx + 1) * params_.hiDim, params_.hoDim) - block_.startH;
                block_.offsetArgmax = ncCoreIdx * params_.doDim * params_.hoDim * params_.woDim + doBlockIdx * params_.hoDim * params_.woDim + hoBlockIdx * params_.woDim + woBlockIdx;
                block_.offsetGrad = block_.offsetArgmax;
                CalcBlock(woBlockIdx);
            }
        }
    }
}
 
__aicore__ inline void CalcBlock(uint64_t woBlockIdx) {
    CopyInArgmax();   
    LocalTensor<int32_t> argmaxUb = indicesQue.DeQue<int32_t>();
    LocalTensor<int32_t> argmaxTranUb = indicesTransposeBuf.Get<int32_t>();
    TransposeBase16M8(argmaxTranUb, argmaxUb, params_.singleCoreNc, block_.dohowoAlign8);
    indicesQue.FreeTensor(argmaxUb); 
 
    CopyInGrad();   
    LocalTensor<TGrad> gradUb = gradQue.DeQue<TGrad>();
    LocalTensor<TGrad> gradTranUb = gradTransposeBuf.Get<TGrad>();
    if constexpr (is_same<TGrad, float>::value) {
        TransposeBase16M8(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign8);  
    } else {
        TransposeBase16M16(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign16);
    }
    gradQue.FreeTensor(gradUb); 

    LocalTensor<float> indicesD = indicesDBuf.Get<float>();  
    LocalTensor<float> indicesH = indicesHBuf.Get<float>();  
    LocalTensor<float> indicesW = indicesWBuf.Get<float>();
    LocalTensor<float> indicesFloat = indicesFloatBuf.Get<float>();
    // 3. Calc IndicesD/H/W (Vector)
    CalcIndices(indicesD, indicesH, indicesW, indicesFloat, argmaxTranUb);
    // 4. Cal indices in k space.  k space is window
    CalcIndicesInWindow(indicesD, indicesH, indicesW, indicesFloat, woBlockIdx);

    LocalTensor<float> kernelIdx = kernelIdxBuf.Get<float>();  
    uint64_t ncCoreIdx = core_.ncCntIndex * params_.singleCoreNc;    
    for (uint64_t woCntIndex = 0; woCntIndex < block_.woShape; woCntIndex++) {
        uint64_t curWoBlockIdx = woBlockIdx + woCntIndex;
        block_.startW = FloorDiv(curWoBlockIdx * params_.wiDim, params_.woDim);
        block_.deltaW = CeilDiv((curWoBlockIdx + 1) * params_.wiDim, params_.woDim) - block_.startW;
        
        block_.offsetY = ncCoreIdx * params_.diHiWiLen + block_.startD * params_.hiDim * params_.wiDim +
                         block_.startH * params_.wiDim + block_.startW;
        // 5.1 Cal mask
        LocalTensor<uint8_t> maskUb = maskBuf.Get<uint8_t>();
        uint64_t mask = params_.singleCoreNc;   
        AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 0};
        Compare(maskUb, kernelIdx, indicesFloat[params_.singleCoreNc * woCntIndex], CMPMODE::EQ, mask, params_.maxKdhwLen, repeatParams); 
        AscendC::PipeBarrier<PIPE_V>();
        // 5.2 Select
        LocalTensor<TGrad> gradSelUb = tempGradBuf.Get<TGrad>();
        SelectGrad(gradSelUb, maskUb, gradTranUb, woCntIndex); 
        AscendC::PipeBarrier<PIPE_V>();
        // 5.3 Calc y
        LocalTensor<TGrad> yTranspose = yTransposeBuf.Get<TGrad>();
        LocalTensor<float> yTransposeFP32 = yTransposeBuf.Get<float>(); 
        CalcY(gradSelUb, yTranspose, yTransposeFP32);
        // 5.4 Y transpose
        YTransposeCopyOut(yTranspose, yTransposeFP32);
    }
}
 
__aicore__ inline void GenkernelIndex(LocalTensor<float>& dstLocal) {
  float firstValue = 0;
  uint64_t kW = params_.maxKw;
  uint64_t kH = params_.maxKh;
  uint64_t kD = params_.maxKd;

  // 1.Dup first Value
  Duplicate(dstLocal, firstValue, params_.singleCoreNc);
  AscendC::PipeBarrier<PIPE_V>();
 
  // 2. Gen kw * vL
  for (uint64_t wIdx = 1; wIdx < kW; wIdx++) {
    Adds(dstLocal[wIdx * params_.singleCoreNc], dstLocal, 1.f * wIdx, params_.singleCoreNc);
  }
  AscendC::PipeBarrier<PIPE_V>();

  // 3. Gen kh * kw * vL
  for (uint64_t hIdx = 1; hIdx < kH; hIdx++) {
    Adds(dstLocal[hIdx * kW * params_.singleCoreNc], dstLocal, 1.f * (hIdx * kW),
         kW * params_.singleCoreNc);
  }
  AscendC::PipeBarrier<PIPE_V>();

  // 4. Gen kd * kh * kw * vL
  for (uint64_t dIdx = 1; dIdx < kD; dIdx++) {
    Adds(dstLocal[dIdx * kH * kW * params_.singleCoreNc], dstLocal,
         1.f * (dIdx * kH * kW), kH * kW * params_.singleCoreNc);
  }
}
 
__aicore__ inline void CopyInArgmax() {
    LocalTensor<TArgmax> argmaxUb = indicesQue.AllocTensor<TArgmax>();
    DataCopyExtParams copyParamsArgmax;
    copyParamsArgmax.blockCount = core_.ncShape;
    copyParamsArgmax.blockLen = block_.dohowoShape * sizeof(TArgmax);
    copyParamsArgmax.srcStride = (params_.doDim * params_.hoDim * params_.woDim - block_.dohowoShape) * sizeof(TArgmax);
    copyParamsArgmax.dstStride = 0;
    DataCopyPadExtParams<TArgmax> padArgmax{false, 0, 0, 0};
    DataCopyPad(argmaxUb, argmaxGm[block_.offsetArgmax], copyParamsArgmax, padArgmax);
    indicesQue.EnQue(argmaxUb);
}
 
__aicore__ inline void CopyInGrad() {
    LocalTensor<TGrad> gradUb = gradQue.AllocTensor<TGrad>();
 
    DataCopyExtParams copyParamsGrad;
    copyParamsGrad.blockCount = core_.ncShape;
    copyParamsGrad.blockLen = block_.dohowoShape * sizeof(TGrad);
    copyParamsGrad.srcStride = (params_.doDim * params_.hoDim * params_.woDim - block_.dohowoShape) * sizeof(TGrad);
    copyParamsGrad.dstStride = 0;
    DataCopyPadExtParams<TGrad> padGrad{false, 0, 0, 0};
 
    DataCopyPad(gradUb, gradGm[block_.offsetGrad], copyParamsGrad, padGrad);
    gradQue.EnQue(gradUb);
}
 
__aicore__ inline void CalcIndices(LocalTensor<float>& indicesD, LocalTensor<float>& indicesH, LocalTensor<float>& indicesW, 
                                   LocalTensor<float>& indicesFloat, LocalTensor<int32_t>& argmaxTranUb) {
    // indicesD = indices / (hi * wi)
    // indicesH = indices % (hi * wi) / wi
    // indicesW = indices % wi
    LocalTensor<float> tempLocal = tempBuf.Get<float>();
    Cast(indicesFloat, argmaxTranUb, AscendC::RoundMode::CAST_NONE, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();

    // 3.1 indicesD = indices / (hi * wi)
    Duplicate<float>(tempLocal, core_.coeffD, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tempLocal, indicesFloat, tempLocal, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>(); 
    Cast(indicesD, tempLocal, AscendC::RoundMode::CAST_FLOOR, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();

    // 3.2 indicesH = indices % (hi * wi) / wi
    // indicesH = (indices - (hi * wi) * indicesD) / wi
    Muls(indicesH, indicesD, core_.hwDims, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Sub(indicesH, indicesFloat, indicesH, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate<float>(tempLocal, core_.coeffH, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tempLocal, indicesH, tempLocal, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(indicesH, tempLocal, AscendC::RoundMode::CAST_FLOOR, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();

    // 3.3 indicesW = indices % wi
    // indicesW = indices - indices / wi * wi
    Duplicate<float>(tempLocal, core_.coeffH, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tempLocal, indicesFloat, tempLocal, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(indicesW, tempLocal, AscendC::RoundMode::CAST_FLOOR, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(indicesW, indicesW, 1.f * (params_.wiDim), params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Sub(indicesW, indicesFloat, indicesW, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CalcIndicesInWindow(LocalTensor<float>& indicesD, LocalTensor<float>& indicesH, LocalTensor<float>& indicesW, 
                                           LocalTensor<float>& indicesFloat, uint64_t woBlockIdx) {
    Adds(indicesD, indicesD, -1.f * block_.startD, params_.singleCoreNc * block_.dohowoAlign8);
    Adds(indicesH, indicesH, -1.f * block_.startH, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();

    for (uint64_t woCntIndex = 0; woCntIndex < block_.woShape; woCntIndex++) {
        uint64_t curWoBlockIdx = woBlockIdx + woCntIndex;
        block_.startW = FloorDiv(curWoBlockIdx * params_.wiDim, params_.woDim);
        Adds(indicesW[params_.singleCoreNc * woCntIndex], indicesW[params_.singleCoreNc * woCntIndex], -1.f * block_.startW, params_.singleCoreNc);
    }
    AscendC::PipeBarrier<PIPE_V>();
 
    // indices_kernel = indices_d_kernel * khLen * kwLen + indices_h_kernel * kwLen + indices_w_kernel
    Muls(indicesFloat, indicesD, 1.f * params_.maxKh * params_.maxKw, params_.singleCoreNc * block_.dohowoAlign8);
    Muls(indicesH, indicesH, 1.f * params_.maxKw, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Add(indicesFloat, indicesFloat, indicesH, params_.singleCoreNc * block_.dohowoAlign8);
    AscendC::PipeBarrier<PIPE_V>();
    Add(indicesFloat, indicesFloat, indicesW, params_.singleCoreNc * block_.dohowoAlign8); 
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CalcY(LocalTensor<TGrad>& gradSelUb, LocalTensor<TGrad>& yTranspose, LocalTensor<float>& yTransposeFP32) {
    uint64_t kWAlignDtype = AlignUp(block_.deltaW, BLOCK_SIZE / sizeof(TGrad));
    uint64_t kWAlign8 = AlignUp(block_.deltaW, BLOCK_NUM_32);
    uint64_t kWLen = block_.deltaW * block_.ncShape;
    if constexpr (is_same<TGrad, float>::value || !IsOverlap) {
        for (uint64_t kD = 0; kD < block_.deltaD; kD++) {
            for (uint64_t kH = 0; kH < block_.deltaH; kH++) {
                if constexpr (is_same<TGrad, bfloat16_t>::value) {
                    LocalTensor<half> yTransposeFp16 = yTranspose.template ReinterpretCast<half>();
                    LocalTensor<half> gradSelUbFp16 = gradSelUb.template ReinterpretCast<half>();
                    AscendC::PipeBarrier<PIPE_V>();
                    Muls(yTransposeFp16[(kD * block_.deltaH + kH) * kWAlignDtype * block_.ncShape],
                         gradSelUbFp16[(kD * params_.maxKh + kH) * params_.maxKw * block_.ncShape], 
                         static_cast<half>(1), kWLen);
                } else {
                    Muls(yTranspose[(kD * block_.deltaH + kH) * kWAlignDtype * block_.ncShape],
                         gradSelUb[(kD * params_.maxKh + kH) * params_.maxKw * block_.ncShape], 
                         static_cast<TGrad>(1), kWLen);
                }
            }
        }
    } else {
        LocalTensor<float> gradSelFP32 = tempGradBuf.Get<float>();
        Cast(gradSelFP32, gradSelUb[params_.singleCoreNc * params_.maxKdhwLen], RoundMode::CAST_NONE, 
             params_.maxKdhwLen * params_.singleCoreNc);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint64_t kD = 0; kD < params_.maxKd; kD++) {    
            for (uint64_t kH = 0; kH < params_.maxKh; kH++) {
                Muls(yTransposeFP32[(kD * block_.deltaH + kH) * kWAlign8 * block_.ncShape],
                     gradSelFP32[(kD * params_.maxKh + kH) * params_.maxKw * block_.ncShape], 1.f, kWLen);
            }
        }
    }
    if constexpr (is_same<TGrad, float>::value || IsOverlap) {
        block_.dihiwiAlign = block_.deltaD * block_.deltaH * kWAlign8;
    } else {
        block_.dihiwiAlign = block_.deltaD * block_.deltaH * kWAlignDtype;
    }
}

__aicore__ inline void YTransposeCopyOut(LocalTensor<TGrad>& yTranspose, LocalTensor<float>& yTransposeFP32) {
    if constexpr (is_same<TY, float>::value) {
        LocalTensor<TY> yUb = yQue.AllocTensor<TY>();
        TransposeBase8M16(yUb, yTranspose, block_.dihiwiAlign, params_.singleCoreNc);
        yQue.EnQue(yUb);
        CopyOut(yGm);
    } else if constexpr (IsOverlap) {
        LocalTensor<float> yUbFp32 = yQue.AllocTensor<float>();
        TransposeBase8M16(yUbFp32, yTransposeFP32, block_.dihiwiAlign, params_.singleCoreNc);
        yQue.EnQue(yUbFp32);
        CopyOut(workspaceGm);
    } else {
        LocalTensor<TY> yUb = yQue.AllocTensor<TY>();
        TransposeBase16M16(yUb, yTranspose, block_.dihiwiAlign, params_.singleCoreNc);
        yQue.EnQue(yUb);
        CopyOut(yGm);
    }
}

__aicore__ inline void SelectGrad(LocalTensor<TGrad>& dstLocal, LocalTensor<uint8_t>& maskUb, LocalTensor<TGrad>& src0Local, uint64_t woCntIndex) {
    uint64_t mask = core_.ncShape;
    int32_t repeat = params_.maxKdhwLen;
    AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 0, 8 };
    if constexpr (is_same<TGrad, float>::value) {
        Select(dstLocal, maskUb, src0Local[params_.singleCoreNc * woCntIndex], static_cast<TGrad>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, repeat, repeatParams);
    } else {
        AscendC::CopyRepeatParams copyRepeatParams = { 1, 1, 4, 0};  // 4 represents dstStride, because datatype is 2bytes and params_.singleCoreNc is 64.
        if constexpr (is_same<TGrad, bfloat16_t>::value) {
            LocalTensor<half> dstLocalFp16 = dstLocal.template ReinterpretCast<half>();
            LocalTensor<half> src0LocalFp16 = src0Local.template ReinterpretCast<half>();
            if constexpr (IsOverlap) {
                Copy(dstLocalFp16[params_.singleCoreNc * repeat], src0LocalFp16[params_.singleCoreNc * woCntIndex], mask, repeat, copyRepeatParams);
                AscendC::PipeBarrier<PIPE_V>();
                Select(dstLocalFp16[params_.singleCoreNc * repeat], maskUb, dstLocalFp16[params_.singleCoreNc * repeat], 
                       static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, repeat * params_.singleCoreNc);
            } else {
                Copy(dstLocalFp16, src0LocalFp16[params_.singleCoreNc * woCntIndex], mask, repeat, copyRepeatParams);
                AscendC::PipeBarrier<PIPE_V>();
                Select(dstLocalFp16, maskUb, dstLocalFp16, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, repeat * params_.singleCoreNc);
            }
        } else {
            if constexpr (IsOverlap) {
                Copy(dstLocal[params_.singleCoreNc * repeat], src0Local[params_.singleCoreNc * woCntIndex], mask, repeat, copyRepeatParams);
                AscendC::PipeBarrier<PIPE_V>();
                Select(dstLocal[params_.singleCoreNc * repeat], maskUb, dstLocal[params_.singleCoreNc * repeat], 
                       static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, repeat * params_.singleCoreNc);
            } else {
                Copy(dstLocal, src0Local[params_.singleCoreNc * woCntIndex], mask, repeat, copyRepeatParams);
                AscendC::PipeBarrier<PIPE_V>();
                Select(dstLocal, maskUb, dstLocal, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, repeat * params_.singleCoreNc);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& outGm) {
    LocalTensor<T> yUb = yQue.DeQue<T>();
    uint64_t kWAlignDtype = AlignUp(block_.deltaW, BLOCK_SIZE / sizeof(T));
    uint64_t maxKwAlign = CeilDiv(block_.deltaW, kWAlignDtype) * kWAlignDtype;
    uint64_t hiwiLen = params_.hiDim * params_.wiDim;
    if constexpr (IsOverlap) {
        SetAtomicAdd<T>();
    } 
    DataCopyExtParams copyParamsY;
    copyParamsY.blockCount = block_.deltaH;
    copyParamsY.blockLen = block_.deltaW * sizeof(T);
    copyParamsY.srcStride = 0;
    copyParamsY.dstStride = (params_.wiDim - block_.deltaW) * sizeof(T);
    for (uint64_t ncIdx = 0; ncIdx < core_.ncShape; ncIdx++) {
        for (uint64_t dIdx = 0; dIdx < block_.deltaD; dIdx++) {
            DataCopyPad(outGm[block_.offsetY + ncIdx * params_.diHiWiLen + dIdx * hiwiLen],
                        yUb[ncIdx * block_.dihiwiAlign + dIdx * block_.deltaH * maxKwAlign], copyParamsY);
        }
    }
    if constexpr (IsOverlap) {
        SetAtomicNone();
    }
    yQue.FreeTensor(yUb);
}

__aicore__ inline void InitCastUbBuffer() {
    pipe->Reset();
    uint64_t maxCalcNum = params_.ubSize / (sizeof(half) + sizeof(float));
    pipe->InitBuffer(wsQue, 1, maxCalcNum * sizeof(float));
    pipe->InitBuffer(yQue, 1, maxCalcNum * sizeof(half));
}

__aicore__ inline void ProcessCast() {
    uint64_t maxCalcNum = params_.ubSize / (sizeof(half) + sizeof(float));
    uint64_t totalLoops = CeilDiv(params_.ncDim * params_.diHiWiLen, maxCalcNum);
    uint64_t calcTail = params_.ncDim * params_.diHiWiLen - (totalLoops - 1) * maxCalcNum;
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
        Cast(b16Ub, fp32Ub, RoundMode::CAST_NONE, calcNum);
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
    BlockParams block_;
    BlockParams core_;
    TilingParams params_;
    TPipe *pipe = nullptr; 
 
    GlobalTensor<TGrad> gradGm;
    GlobalTensor<TArgmax> argmaxGm;
    GlobalTensor<TY> yGm;
    GlobalTensor<float> workspaceGm;
 
    TQue<QuePosition::VECIN, 1> gradQue;
    TQue<QuePosition::VECIN, 1> indicesQue;
    TQue<QuePosition::VECIN, 1> wsQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> indicesTransposeBuf;
    TBuf<TPosition::VECCALC> indicesFloatBuf;
    TBuf<TPosition::VECCALC> indicesDBuf;
    TBuf<TPosition::VECCALC> indicesHBuf;
    TBuf<TPosition::VECCALC> indicesWBuf;
    TBuf<TPosition::VECCALC> tempBuf;
    TBuf<TPosition::VECCALC> tempGradBuf;
 
    TBuf<TPosition::VECCALC> kernelIdxBuf;
    TBuf<TPosition::VECCALC> gradTransposeBuf;
    TBuf<TPosition::VECCALC> yTransposeBuf;
};
}  // namespace AdaptiveMaxPool3DGrad
#endif  // ADAPTIVE_MAX_POOL3D_GRAD_NORMAL_H