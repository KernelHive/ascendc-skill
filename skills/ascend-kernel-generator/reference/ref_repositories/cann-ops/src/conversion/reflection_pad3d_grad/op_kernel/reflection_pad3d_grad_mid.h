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
 * @file reflection_pad3d_grad_mid.h
 */
#ifndef REFLECTION_PAD3D_GRAD_MID_H
#define REFLECTION_PAD3D_GRAD_MID_H
#include "reflection_pad3d_grad_init.h"

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::MidProcess() {
    for (size_t loop = 0; loop < loopNC; loop++) {
        for (size_t i = 0; i < curDepth; i++) {
            size_t curDim = GetCurD(i); 
            bool isAtomicAdd = true;
            MidProcessTopBottom(i, loop, curDim, isAtomicAdd);
            MidProcessLeftRight(i, loop, curDim, isAtomicAdd);
            MidProcessMid(i, loop, curDim, isAtomicAdd);
        }
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::MidProcessTopBottom(size_t i,  size_t loop, uint32_t curDim, bool isAtomicAdd) {
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    //top
    int64_t gmXOffset = (loop * curDepth * height * width
                        + i * height * width);
    int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                        + curDim * outHeight * outWidth
                        + MAX_LINE - wPad1);
    //xGm -> ub
    CopyIn(xGm, gmXOffset, MAX_LINE, width);
    // ub计算
    if (hPad1 > 0) {
        ComputeTopGrad(width);
    } else {
        ComputeCopy((MAX_LINE) * CeilAlign(width, perBlockCount));
    }
    //ub -> yGm、wGm
    LocalTensor<T> yTopLocal = outQueueY.DeQue<T>();
    CopyOutBasic(yGm, yTopLocal, CopyOutParam(gmYOffset, hPad1 * (CeilAlign(width, perBlockCount)) + MAX_LINE, MAX_LINE - hPad1, width - MAX_LINE -MAX_LINE, outWidth, isAtomicAdd, MAX_LINE + MAX_LINE));
    CopyOutBasic(workspaceGm, yTopLocal, CopyOutParam(gmWorkSpaceOffset1, 0, MAX_LINE, width, width, false));
    outQueueY.FreeTensor(yTopLocal);
    //同步
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    
    //bottom
    gmXOffset = (loop * curDepth * height * width
                + i * height * width
                + (height - MAX_LINE) * width);
    gmYOffset = (loop *  curOutDepth * outHeight * outWidth 
                + curDim *  outHeight * outWidth
                + (outHeight - (MAX_LINE - hPad2)) * outWidth
                +  MAX_LINE - wPad1);
    //xGm -> ub
    CopyIn(xGm, gmXOffset, MAX_LINE, width);
    //ub 计算
    if (hPad2 > 0) {
        ComputeBottomGrad(width);
    } else {
        ComputeCopy((MAX_LINE) * CeilAlign(width, perBlockCount));
    }
    //ub -> yGm、wGm
    LocalTensor<T> yBottomLocal = outQueueY.DeQue<T>();
    CopyOutBasic(yGm, yBottomLocal, CopyOutParam(gmYOffset, MAX_LINE, MAX_LINE - hPad2, width - MAX_LINE -MAX_LINE, outWidth, isAtomicAdd, MAX_LINE + MAX_LINE));
    CopyOutBasic(workspaceGm, yBottomLocal, CopyOutParam(gmWorkSpaceOffset2, 0, MAX_LINE, width, width, false));
    outQueueY.FreeTensor(yBottomLocal);
    //同步
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::MidProcessLeftRight(size_t i,  size_t loop, uint32_t curDim, bool isAtomicAdd) {
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    //left
    int64_t gmXOffset = (loop * curDepth * height * width
                        + i * height * width 
                        + MAX_LINE * width);
    int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                        + curDim * outHeight * outWidth );
    int64_t gmWorkSpaceOffsetTop =  gmWorkSpaceOffset1;
    int64_t gmWorkSpaceOffsetBottom =  gmWorkSpaceOffset2;
    //xGm、wGm -> ub
    LocalTensor<T> xLeftLocal = inQueueX.AllocTensor<T>();
    CopyInBasic(xLeftLocal, workspaceGm, CopyInParam(0, gmWorkSpaceOffsetTop, MAX_LINE, MAX_LINE));
    CopyInBasic(xLeftLocal, xGm, CopyInParam(MAX_LINE * MAX_LINE, gmXOffset, height - MAX_LINE - MAX_LINE, MAX_LINE));
    CopyInBasic(xLeftLocal, workspaceGm, CopyInParam((height - MAX_LINE) *  MAX_LINE, gmWorkSpaceOffsetBottom, alignHeight - (height - MAX_LINE), MAX_LINE));
    inQueueX.EnQue(xLeftLocal);
    //ub 计算
    if (wPad1 > 0) {
        ComputeLeftGrad(alignHeight);
    } else {
        ComputeCopy(MAX_LINE * alignHeight);
    }
    //ub -> yGm
    LocalTensor<T> yLeftLocal = outQueueY.DeQue<T>();
    CopyOutBasic(yGm, yLeftLocal, CopyOutParam(gmYOffset, hPad1 * MAX_LINE, outHeight, MAX_LINE - wPad1, outWidth, isAtomicAdd));
    outQueueY.FreeTensor(yLeftLocal);
    //同步
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

    //right
    gmXOffset = (loop * curDepth * height * width
                + i * height * width 
                + MAX_LINE * width
                + width - MAX_LINE);
    gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                + curDim * outHeight * outWidth 
                + outWidth  - (MAX_LINE - wPad2) );
    gmWorkSpaceOffsetTop =  gmWorkSpaceOffset1 + width - MAX_LINE;
    gmWorkSpaceOffsetBottom =  gmWorkSpaceOffset2 + width - MAX_LINE;
    //xGm、wGm -> ub
    LocalTensor<T> xRightLocal = inQueueX.AllocTensor<T>();
    CopyInBasic(xRightLocal, workspaceGm, CopyInParam(0, gmWorkSpaceOffsetTop, MAX_LINE, MAX_LINE));
    CopyInBasic(xRightLocal , xGm, CopyInParam(MAX_LINE * MAX_LINE, gmXOffset, height - MAX_LINE - MAX_LINE, MAX_LINE));
    CopyInBasic(xRightLocal, workspaceGm, CopyInParam((height - MAX_LINE) *  MAX_LINE, gmWorkSpaceOffsetBottom, alignHeight - (height - MAX_LINE), MAX_LINE));
    inQueueX.EnQue(xRightLocal);
    //ub 计算
    if (wPad2 > 0) {
        ComputeRightGrad(alignHeight);
    } else {
        ComputeCopy(MAX_LINE * alignHeight);
    }
    //ub -> yGm
    LocalTensor<T> yRightLocal = outQueueY.DeQue<T>();
    CopyOutBasic(yGm, yRightLocal, CopyOutParam(gmYOffset, hPad1 * MAX_LINE, outHeight, MAX_LINE - wPad2, outWidth, isAtomicAdd));
    outQueueY.FreeTensor(yRightLocal);
    //同步
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::MidProcessMid(size_t i,  size_t loop, uint32_t curDim, bool isAtomicAdd) {
    int64_t blockHeight = FloorDiv(ubFactorElement, CeilAlign(width - MAX_LINE -MAX_LINE, 16));
    int64_t gmLoop = CeilDiv(height - MAX_LINE -MAX_LINE, blockHeight);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    //输入GM搬运
    if (gmLoop > 0) {
        for (size_t j = 0; j < gmLoop; j++) {
            int64_t gmXOffset = (loop * curDepth * height * width
                                + i * height * width
                                + MAX_LINE * width
                                + MAX_LINE
                                + j * blockHeight * width);
            int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                                + curDim * outHeight * outWidth 
                                + (MAX_LINE - hPad1) * outWidth 
                                + MAX_LINE - wPad1
                                + j * blockHeight * outWidth);
            int64_t calHeight = blockHeight;
            if (j == gmLoop - 1) {
                calHeight = height - MAX_LINE -MAX_LINE - j * blockHeight;
            }
            //xGm -> ub
            if (width - MAX_LINE - MAX_LINE > 0 && calHeight > 0) {
                CopyIn(xGm, gmXOffset, calHeight, width - MAX_LINE - MAX_LINE);
                //ub计算
                ComputeCopy((calHeight) * CeilAlign(width - MAX_LINE -MAX_LINE, perBlockCount));
                //ub -> yGm
                CopyOut(yGm, CopyOutParam(gmYOffset, 0, calHeight, width - MAX_LINE -MAX_LINE, outWidth, isAtomicAdd));
            }

            //同步
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyIn(GlobalTensor<T>& srcGm, const int64_t srcOffset, const int64_t calH, const int64_t calW) {
    LocalTensor<T> dstLocal = inQueueX.AllocTensor<T>();
    CopyInBasic(dstLocal, srcGm, CopyInParam(0, srcOffset, calH, calW));
    inQueueX.EnQue(dstLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyInBasic(LocalTensor<T>& dstLocal, GlobalTensor<T>& srcGm, CopyInParam param) {
    int32_t alignCalW = CeilAlign(param.calW, perBlockCount);
    if (param.calH <= 0 || param.calW <= 0) {
        return;
    }
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, 0, 0};
    copyParams.blockCount = param.calH;
    copyParams.blockLen = param.calW * sizeof(T);
    copyParams.srcStride = (width - param.calW)  * sizeof(T);
    copyParams.dstStride = 0;
    padParams.isPad = true;
    padParams.rightPadding = alignCalW - param.calW;
    DataCopyPad(dstLocal[param.dstOffset], srcGm[param.srcOffset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyOut(GlobalTensor<T>& dstGm, CopyOutParam param) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    CopyOutBasic(dstGm, yLocal, param);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::CopyOutBasic(GlobalTensor<T>& dstGm, LocalTensor<T>& srcLocal, CopyOutParam param) {
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    if (param.calH <= 0 || param.calW <=0) {
        return;
    }
    copyParams.blockCount = param.calH ;
    copyParams.blockLen = param.calW * sizeof(T);
    //copyOut 分ub->wGm, yGm 
    copyParams.srcStride = ((param.srcStride) * sizeof(T)) / 32;   
    copyParams.dstStride = (param.offsetWidth - param.calW) * sizeof(T);
    if (param.isAtomicAdd == true) {
        SetAtomicAdd<T>();
        DataCopyPad(dstGm[param.dstOffset], srcLocal[param.srcOffset], copyParams);
        SetAtomicNone();
    } else {
        DataCopyPad(dstGm[param.dstOffset], srcLocal[param.srcOffset], copyParams);
    }  
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeTopGrad(const int32_t calW) {
    if (hPad1 > 0) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        int32_t alignCalW = CeilAlign(calW, perBlockCount);  // calCount % 16 可以不为0
        int32_t totalData = alignCalW * MAX_LINE; 
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            LocalTensor<float> float32Tensor = float32Buf.Get<float>();
            Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
            for(uint32_t i = 0; i < hPad1; i++) {
            auto srcLocal1 = float32Tensor[i * alignCalW];
            auto srcLocal2 = float32Tensor[(2 * hPad1 - i)  * alignCalW];
            Add(srcLocal2, srcLocal2, srcLocal1, alignCalW);
            }
            Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, totalData);
        } else {
            for(uint32_t i = 0; i < hPad1; i++) {
                auto srcLocal1 = xLocal[i * alignCalW];
                auto srcLocal2 = xLocal[(2 * hPad1 - i)  * alignCalW];
                Add(srcLocal2, srcLocal2, srcLocal1, alignCalW);
            }
            Copy(yLocal, xLocal, MAX_COPY / sizeof(T), CeilDiv(totalData, MAX_COPY / sizeof(T)), {1, 1, 8, 8}); 
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeBottomGrad(const int32_t calW) {
    if (hPad2 > 0) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        int32_t alignCalW = CeilAlign(calW, perBlockCount);  // calCount % 16 可以不为0
        int32_t totalData = alignCalW * MAX_LINE; 
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            LocalTensor<float> float32Tensor = float32Buf.Get<float>();
            Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
            // ComputeBottomGradBasic<float>(float32Tensor, calW);
            for(uint32_t i = 0; i < hPad2; i++) {
                auto srcLocal1 = float32Tensor[(MAX_LINE - 1 - i) * alignCalW];
                auto srcLocal2 = float32Tensor[(MAX_LINE - 1 - (2 * hPad2 - i))  * alignCalW];
                Add(srcLocal2, srcLocal2, srcLocal1, alignCalW);
            }
            Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, totalData);
        } else {
            for(uint32_t i = 0; i < hPad2; i++) {
                auto srcLocal1 = xLocal[(MAX_LINE - 1 - i) * alignCalW];
                auto srcLocal2 = xLocal[(MAX_LINE - 1 - (2 * hPad2 - i))  * alignCalW];
                Add(srcLocal2, srcLocal2, srcLocal1, alignCalW);
            }
            Copy(yLocal, xLocal, MAX_COPY / sizeof(T), CeilDiv(totalData, MAX_COPY / sizeof(T)), {1, 1, 8, 8}); 
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeCopy(const int32_t totalData) {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    Copy(yLocal, xLocal, MAX_COPY / sizeof(T), CeilDiv(totalData, MAX_COPY / sizeof(T)), {1, 1, 8, 8}); 
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

//K * 16 -> 16 * K
template <typename T> template<typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::Transose1(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal, const int32_t calH) {
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = calH / 16;
    transDataParams.dstRepStride = (16 * sizeof(T1)) / 32;
    transDataParams.srcRepStride = (16 * 16 * sizeof(T1)) / 32; 
    if (transDataParams.repeatTimes == 1) {
        transDataParams.dstRepStride = 0;
        transDataParams.srcRepStride = 0;
    }
    // 入参类型是LocalTensor地址值的调用方式，推荐使用
    uint64_t srcLocalList[16];
    for (int i = 0; i < 16; i++) {
        srcLocalList[i] = (uint64_t)(srcLocal[16 * i].GetPhyAddr());
    }
    uint64_t dstLocalList[16];
    if constexpr (std::is_same<T1, float>::value) {
        for (int i = 0; i < 8; i++) {
            dstLocalList[2 * i] = (uint64_t)(dstLocal[calH * i].GetPhyAddr());
            dstLocalList[2 * i + 1] = (uint64_t)(dstLocal[calH * i + 8].GetPhyAddr());
        }
        TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);

        for (int i = 0; i < 16; i++) {
            srcLocalList[i] = (uint64_t)(srcLocal[16 * i + 8].GetPhyAddr());
        }
        for (int i = 0; i < 8; i++) {
            dstLocalList[2 * i] = (uint64_t)(dstLocal[calH * 8  + calH * i].GetPhyAddr());
            dstLocalList[2 * i + 1] = (uint64_t)(dstLocal[calH * 8  + calH * i + 8].GetPhyAddr());
        }
        TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
    } else {
        for (int i = 0; i < 16; i++) {
            dstLocalList[i] = (uint64_t)(dstLocal[calH * i].GetPhyAddr());
        }
        TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
    }
}
//16 *K -> K * 16
template <typename T> template<typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::Transose2(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal, const int32_t calW){
    int32_t t = 16;
    if constexpr (std::is_same<T1, float>::value) {    
        t = 8;
    } 
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = calW / t;
    transDataParams.dstRepStride = (t * 16 * sizeof(T1)) / 32 ; 
    transDataParams.srcRepStride = (t * sizeof(T1)) / 32 ; 
    if (transDataParams.repeatTimes == 1) {
        transDataParams.dstRepStride = 0;
        transDataParams.srcRepStride = 0;
    }
    // 入参类型是LocalTensor地址值的调用方式，推荐使用
    uint64_t srcLocalList[16];
    for (int i = 0; i < 16; i++) {
        srcLocalList[i] = (uint64_t)(srcLocal[calW * i].GetPhyAddr());
    }
    uint64_t dstLocalList[16];
    for (int i = 0; i < 16; i++) {
        dstLocalList[i] = (uint64_t)(dstLocal[t * i].GetPhyAddr());
    }
    TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
}

template <typename T> template<typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeLeftGradBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal, const int32_t calH) {
    if (wPad1 > 0) {
        Transose1<T1>(tLocal, xLocal, calH);
        for(uint32_t i = 0; i < wPad1; i++) {
            auto srcLocal1 = tLocal[i * calH];
            auto srcLocal2 = tLocal[(2 * wPad1 - i) * calH];
            Add(srcLocal2, srcLocal2, srcLocal1, calH); 
        }
        for (uint32_t i = 0; i < MAX_LINE - wPad1; i++) {
            auto srcLocal1 = tLocal[i * calH];
            auto srcLocal2 = tLocal[(i + wPad1) * calH];
            Muls(srcLocal1, srcLocal2, (T1)1.0, calH);
        }
    }
}

template <typename T>        
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeLeftGrad(const int32_t calH) {
    if (wPad1 > 0) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        int32_t totalData = calH * MAX_LINE; 
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            LocalTensor<float> tLocal = transposeBuf.Get<float>();
            LocalTensor<float> float32Tensor = float32Buf.Get<float>();
            Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
            ComputeLeftGradBasic<float>(tLocal, float32Tensor, calH);
            Transose2<float>(float32Tensor, tLocal, calH);
            Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, totalData);
        } else {
            LocalTensor<T> tLocal = transposeBuf.Get<T>();
            ComputeLeftGradBasic<T>(tLocal, xLocal, calH);
            Transose2<T>(yLocal, tLocal, calH);
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename T> template<typename T1>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeRightGradBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal, const int32_t calH) {
    if (wPad2 > 0) {
        Transose1<T1>(tLocal, xLocal, calH);
        for(uint32_t i = 0; i < wPad2; i++) {
            auto srcLocal1 = tLocal[(MAX_LINE - 1 - i) * calH];
            auto srcLocal2 = tLocal[(MAX_LINE - 1 - (2 * wPad2 - i)) * calH];
            Add(srcLocal2, srcLocal2, srcLocal1, calH); 
        }
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::ComputeRightGrad(const int32_t calH) {
    if (wPad2 > 0) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        int32_t totalData = calH * MAX_LINE; 
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            LocalTensor<float> tLocal = transposeBuf.Get<float>();
            LocalTensor<float> float32Tensor = float32Buf.Get<float>();
            Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
            ComputeRightGradBasic<float>(tLocal, float32Tensor, calH);
            Transose2<float>(float32Tensor, tLocal, calH);
            Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, totalData);
        } else {
            LocalTensor<T> tLocal = transposeBuf.Get<T>();
            ComputeRightGradBasic<T>(tLocal, xLocal, calH);
            Transose2<T>(yLocal, tLocal, calH);
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

#endif  // REFLECTION_PAD3D_GRAD_MID_H