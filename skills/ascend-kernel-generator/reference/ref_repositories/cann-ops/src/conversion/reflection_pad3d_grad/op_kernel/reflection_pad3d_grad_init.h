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
 * @file reflection_pad3d_grad_init.h
 */
#ifndef REFLECTION_PAD3D_GRAD_INIT_H
#define REFLECTION_PAD3D_GRAD_INIT_H
#include "reflection_pad3d_grad_utils.h"

using namespace AscendC;

template <typename T>
class ReflectionPad3dGrad {
public:
    const static int32_t BUFFER_NUM = 2;
    const static uint32_t BLOCK_BYTES = 32;
    const static uint32_t MAX_LINE = 16;
    const static uint32_t MAX_COPY = 256;
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> float32Buf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;  
    GlobalTensor<T> workspaceGm;
    uint32_t batch = 0;
    uint32_t channel =0;
    uint32_t depth = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignDepth = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outDepth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t dPad1 = 0;
    uint32_t dPad2 = 0;
    uint32_t hPad1 = 0;
    uint32_t hPad2 = 0;
    uint32_t wPad1 = 0;
    uint32_t wPad2 = 0;
    uint32_t ncPerCore = 0;
    uint32_t tailNC = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t blockIdx = 0;
    uint32_t perBlockCount = 0;
    int64_t gmWorkSpaceOffset1 = 0;
    int64_t gmWorkSpaceOffset2 = 0;
    uint32_t workSpacePart = 32;
    uint32_t loopNC = 0;
    int64_t ncOffset = 0; 
    uint32_t curDepth;
    uint32_t curOutDepth;

public:
    __aicore__ inline ReflectionPad3dGrad() {}

    __aicore__ inline void Init(const ReflectionPad3dGradTilingData &__restrict tilingData, GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR userWS)
    {
        batch = tilingData.batch;
        channel = tilingData.channel;
        depth = tilingData.depth;
        height = tilingData.height;
        width = tilingData.width;
        alignDepth = tilingData.alignDepth;
        alignHeight = tilingData.alignHeight;
        alignWidth = tilingData.alignWidth;
        outDepth = tilingData.outDepth;
        outHeight = tilingData.outHeight;
        outWidth = tilingData.outWidth;
        dPad1 = tilingData.dPad1;
        dPad2 = tilingData.dPad2;
        hPad1 = tilingData.hPad1;
        hPad2 = tilingData.hPad2;
        wPad1 = tilingData.wPad1;
        wPad2 = tilingData.wPad2;
        ncPerCore = tilingData.ncPerCore;
        tailNC = tilingData.tailNC;
        blockNum = tilingData.blockNum;
        ubFactorElement = tilingData.ubFactorElement;
        blockIdx = GetBlockIdx();
        perBlockCount = BLOCK_BYTES / sizeof(T);

        gmWorkSpaceOffset1 = 0;
        gmWorkSpaceOffset2 = Mymax(alignHeight, alignWidth) * MAX_LINE;
        if (blockIdx < tailNC) {
            loopNC = ncPerCore + 1;
            ncOffset = blockIdx * loopNC;
        } else {
            loopNC = ncPerCore;
            ncOffset = blockIdx * ncPerCore + tailNC;
        }
        curDepth = depth;
        curOutDepth = outDepth;
        if (dPad1 == 0 && dPad2 == 0) {
            curDepth = 1;
            curOutDepth = 1;
        }
        InitBuff(x, y, userWS);
    }

    __aicore__ inline void ClearOutput(GM_ADDR y) {
        int64_t totaldata = batch * channel * outDepth * outHeight * outWidth;
        int64_t preLen = totaldata / blockNum;
        int64_t tailLen = totaldata % blockNum;
        int64_t curLen = preLen;
        int64_t curOffset = blockIdx * preLen;
        if (blockIdx < tailLen) {
            curLen = preLen + 1;
            curOffset = blockIdx * curLen;
        } else {
            curLen = preLen;
            curOffset = blockIdx * preLen + tailLen;
        }
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + curOffset);
        InitGlobalMemory<T>(yGm, curLen, 0);
        SyncAll();
    }

    __aicore__ inline void InitBuff(GM_ADDR x, GM_ADDR y, GM_ADDR userWS) {
        ClearOutput(y); 
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + ncOffset * curDepth * height * width );
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + ncOffset * curOutDepth * outHeight * outWidth);
        workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(userWS) + Mymax(alignHeight, alignWidth) * workSpacePart * blockIdx);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, (ubFactorElement * sizeof(T)));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, (ubFactorElement * sizeof(T)));
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            pipe.InitBuffer(transposeBuf, (ubFactorElement * sizeof(float)) );
            pipe.InitBuffer(float32Buf, (ubFactorElement * sizeof(float)) );
        } else {
            pipe.InitBuffer(transposeBuf, (ubFactorElement * sizeof(T)) );
        } 
    }

    __aicore__ inline int GetCurD(size_t i) {
        size_t curDim = i;
        if(i <= dPad1) {   
            curDim = dPad1 - i;  
        } else if (i > dPad1 && i <depth - dPad2) { 
            curDim = i - dPad1;  
        } else if (i >= depth - dPad2) {     
            curDim = (depth - dPad2 - 1) - (i - (depth - dPad2) + 1) - dPad1;   
        }
        return curDim;
    }

    __aicore__ inline void MidProcess();

    __aicore__ inline void FlatProcess();

    __aicore__ inline void SmallProcess();

    __aicore__ inline void BigProcess();

private:
    __aicore__ inline void CopyInSmall(const int64_t offset, const int64_t calH, const int64_t calW, const int64_t srcStride);

    __aicore__ inline void ComputeSmall(size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask, size_t wPad2Mask, const int32_t calH, const int32_t calW);
    template<typename T1>
    __aicore__ inline void ComputeSmallBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal, size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask, size_t wPad2Mask, const int32_t calH, const int32_t calW);

    __aicore__ inline void CopyOutSmall(const int64_t offset, const int64_t srcOffset, const bool isAtomicAdd, const int32_t calH, const int32_t calW, const int32_t alignTransCalW, const int32_t dstStride);
    template<typename T1>
    __aicore__ inline void TransoseSmall(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal, const int32_t calH, const int32_t calW);
   
    __aicore__ inline void MidProcessTopBottom(size_t i,  size_t loop, uint32_t curDim, bool isAtomicAdd);

    __aicore__ inline void MidProcessLeftRight(size_t i,  size_t loop, uint32_t curDim, bool isAtomicAdd);

    __aicore__ inline void MidProcessMid(size_t i,  size_t loop, uint32_t curDim, bool isAtomicAdd);

    __aicore__ inline void CopyIn(GlobalTensor<T>& srcGm, const int64_t srcOffset, const int64_t calH, const int64_t calW);

    __aicore__ inline void CopyInBasic(LocalTensor<T>& dstLocal, GlobalTensor<T>& srcGm, CopyInParam param);

    __aicore__ inline void CopyOut(GlobalTensor<T>& dstGm, CopyOutParam param);

    __aicore__ inline void CopyOutBasic(GlobalTensor<T>& dstGm, LocalTensor<T>& srcLocal, CopyOutParam param);

    __aicore__ inline void ComputeTopGrad(const int32_t calW);

    __aicore__ inline void ComputeBottomGrad(const int32_t calW);

    __aicore__ inline void ComputeCopy(const int32_t totalData);

    //K * 16 -> 16 * K
    template<typename T1>
    __aicore__ inline void Transose1(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal, const int32_t calH);
    //16 *K -> K * 16
    template<typename T1>
    __aicore__ inline void Transose2(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal, const int32_t calW);

    template<typename T1>
    __aicore__ inline void ComputeLeftGradBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal, const int32_t calH);
            
    __aicore__ inline void ComputeLeftGrad(const int32_t calH);

    template<typename T1>
    __aicore__ inline void ComputeRightGradBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal, const int32_t calH);

    __aicore__ inline void ComputeRightGrad(const int32_t calH);

    __aicore__ inline void WidthFlatProcess();

    __aicore__ inline void HeightFlatProcess();

    __aicore__ inline void BigProcessTop(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd);

    __aicore__ inline void BigProcessBottom(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd);

    __aicore__ inline void BigProcessLeft(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd);

    __aicore__ inline void BigProcessRight(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd);

    __aicore__ inline void BigProcessMid(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd);
};

#endif  // REFLECTION_PAD3D_GRAD_INIT_H