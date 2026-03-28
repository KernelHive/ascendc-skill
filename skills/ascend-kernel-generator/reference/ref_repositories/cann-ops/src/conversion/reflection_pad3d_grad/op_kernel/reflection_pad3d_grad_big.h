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
 * @file reflection_pad3d_grad_big.h
 */
#ifndef REFLECTION_PAD3D_GRAD_BIG_H
#define REFLECTION_PAD3D_GRAD_BIG_H
#include "reflection_pad3d_grad_init.h"


template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::BigProcess() {
    for ( size_t loop = 0; loop < loopNC; loop++ ) {
        for ( size_t i = 0; i < curDepth; i++ ) {
            size_t cur_D = GetCurD(i); 
            bool isAtomicAdd = true;
            BigProcessTop(loop, i, cur_D, isAtomicAdd);
            BigProcessBottom(loop, i, cur_D, isAtomicAdd);
            BigProcessLeft(loop, i, cur_D, isAtomicAdd);
            BigProcessRight(loop, i, cur_D, isAtomicAdd);
            BigProcessMid(loop, i, cur_D, isAtomicAdd);
        }
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::BigProcessTop(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd) {
    int32_t calH = MAX_LINE;
    int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(calH, MAX_LINE)), MAX_LINE);
    int32_t gmLoop_Width = CeilDiv(width - MAX_LINE -MAX_LINE, blockWidth);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    // first
    int64_t gmXOffset = (loop * curDepth * height * width
                        + i * height * width);
    int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                        + cur_D * outHeight * outWidth);

    CopyInSmall(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
    ComputeSmall(0, 1, 0, 1, calH, MAX_LINE);
    CopyOutSmall(gmYOffset, hPad1 * MAX_LINE, isAtomicAdd, calH - hPad1, MAX_LINE - wPad1, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad1)) * sizeof(T));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

    //mid
    for (size_t j = 0; j < gmLoop_Width; j++) {
        gmXOffset = (loop * curDepth * height * width
                    + i * height * width
                    + MAX_LINE
                    + j * blockWidth);
        gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                    + cur_D * outHeight * outWidth
                    + (MAX_LINE - wPad1)
                    + j * blockWidth);
        int64_t calWidth = blockWidth;
        if (j == gmLoop_Width - 1) {
            calWidth = width - MAX_LINE -MAX_LINE - j * blockWidth;
        }
        CopyInSmall(gmXOffset, calH, calWidth, (width - calWidth) * sizeof(T));
        ComputeSmall(0, 1, 1, 1, calH, calWidth);
        CopyOutSmall(gmYOffset, hPad1 * CeilAlign(calWidth, MAX_LINE), isAtomicAdd, calH - hPad1, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
    //  last
    gmXOffset = (loop * curDepth * height * width
                + i * height * width
                + width - MAX_LINE);
    gmYOffset = (loop *  curOutDepth * outHeight * outWidth 
                + cur_D *  outHeight * outWidth
                + outWidth - (MAX_LINE - wPad2));
    
    CopyInSmall(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
    ComputeSmall(0, 1, 1, 0, calH, MAX_LINE);
    CopyOutSmall(gmYOffset, hPad1 * MAX_LINE, isAtomicAdd, calH - hPad1, MAX_LINE - wPad2, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad2)) * sizeof(T));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::BigProcessBottom(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd) {
    int32_t calH = MAX_LINE;
    int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(calH, MAX_LINE)), MAX_LINE);
    int32_t gmLoop_Width = CeilDiv(width - MAX_LINE -MAX_LINE, blockWidth);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    // first
    int64_t gmXOffset = (loop * curDepth * height * width
                        + i * height * width
                        + (height - MAX_LINE) * width);
    int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                        + cur_D * outHeight * outWidth
                        + (outHeight - (MAX_LINE - hPad2)) * outWidth);

    CopyInSmall(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
    ComputeSmall(1, 0, 0, 1, calH, MAX_LINE);
    CopyOutSmall(gmYOffset, 0, isAtomicAdd, calH - hPad2, MAX_LINE - wPad1, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad1)) * sizeof(T));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

    //mid
    for (size_t j = 0; j < gmLoop_Width; j++) {
        gmXOffset = (loop * curDepth * height * width
                    + i * height * width
                    + (height - MAX_LINE) * width
                    + MAX_LINE
                    + j * blockWidth);
        gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                    + cur_D * outHeight * outWidth
                    + (outHeight - (MAX_LINE - hPad2)) * outWidth
                    + (MAX_LINE - wPad1)
                    + j * blockWidth);
        int64_t calWidth = blockWidth;
        if (j == gmLoop_Width - 1) {
            calWidth = width - MAX_LINE -MAX_LINE - j * blockWidth;
        }
        CopyInSmall(gmXOffset, calH, calWidth, (width - calWidth) * sizeof(T));
        ComputeSmall(1, 0, 1, 1, calH, calWidth);
        CopyOutSmall(gmYOffset, 0, isAtomicAdd, calH - hPad2, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
    //  last
    
    gmXOffset = (loop * curDepth * height * width
                + i * height * width
                + (height - MAX_LINE) * width
                + width - MAX_LINE);
    gmYOffset = (loop *  curOutDepth * outHeight * outWidth 
                + cur_D *  outHeight * outWidth
                + (outHeight - (MAX_LINE - hPad2)) * outWidth
                + outWidth - (MAX_LINE - wPad2));
    
    CopyInSmall(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
    ComputeSmall(1, 0, 1, 0, calH, MAX_LINE);
    CopyOutSmall(gmYOffset, 0, isAtomicAdd, calH - hPad2, MAX_LINE - wPad2, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad2)) * sizeof(T));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::BigProcessLeft(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd) {
    int32_t calW = MAX_LINE;
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    int32_t blockHeight = FloorDiv(ubFactorElement, CeilAlign(calW, MAX_LINE));
    int32_t gmLoop_Height = CeilDiv(height - MAX_LINE -MAX_LINE, blockHeight);

    for (size_t j = 0; j < gmLoop_Height; j++) {
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width
                            + MAX_LINE * width
                            + j * blockHeight * width);
        int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                            + cur_D * outHeight * outWidth
                            + ( MAX_LINE - hPad1) * outWidth
                            + j * blockHeight * outWidth);
        int64_t calHeight = blockHeight;
        if (j == gmLoop_Height - 1) {
            calHeight = height - MAX_LINE -MAX_LINE - j * blockHeight;
        }
        CopyInSmall(gmXOffset, calHeight, calW, (width - calW) * sizeof(T));
        ComputeSmall(1, 1, 0, 1, calHeight, calW);
        CopyOutSmall(gmYOffset, 0, isAtomicAdd, calHeight, calW - wPad1, CeilAlign(calW, MAX_LINE), (outWidth - (calW -wPad1)) * sizeof(T));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::BigProcessRight(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd) {
    int32_t calW = MAX_LINE;
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    int32_t blockHeight = FloorDiv(ubFactorElement, CeilAlign(calW, MAX_LINE));
    int32_t gmLoop_Height = CeilDiv(height - MAX_LINE -MAX_LINE, blockHeight);

    for (size_t j = 0; j < gmLoop_Height; j++) {
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width
                            + MAX_LINE * width
                            + width - MAX_LINE
                            + j * blockHeight * width);
        int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                            + cur_D * outHeight * outWidth
                            + ( MAX_LINE - hPad1) * outWidth
                            + outWidth  - (MAX_LINE - wPad2)
                            + j * blockHeight * outWidth);
        int64_t calHeight = blockHeight;
        if (j == gmLoop_Height - 1) {
            calHeight = height - MAX_LINE -MAX_LINE - j * blockHeight;
        }
        CopyInSmall(gmXOffset, calHeight, calW, (width - calW) * sizeof(T));
        ComputeSmall(1, 1, 1, 0, calHeight, calW);
        CopyOutSmall(gmYOffset, 0, isAtomicAdd, calHeight, calW - wPad2, CeilAlign(calW, MAX_LINE), (outWidth - (calW -wPad2)) * sizeof(T));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::BigProcessMid(size_t loop,  size_t i, uint32_t cur_D, bool isAtomicAdd) {
    int32_t blockHeight = MAX_LINE;
    int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(blockHeight, MAX_LINE)), MAX_LINE);
    int32_t gmLoop_Width = CeilDiv(width - MAX_LINE -MAX_LINE, blockWidth);
    int32_t gmLoop_Height = CeilDiv(height - MAX_LINE -MAX_LINE, blockHeight);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (size_t j = 0; j < gmLoop_Height; j ++ ) {
        int64_t calHeight = blockHeight;
        if (j == gmLoop_Height - 1) {
            calHeight = height - MAX_LINE -MAX_LINE - j * blockHeight;
        }          
        for (size_t k = 0; k < gmLoop_Width; k ++) {
            int64_t gmXOffset = (loop * curDepth * height * width
                                + i * height * width
                                + MAX_LINE * width
                                + MAX_LINE
                                + j * blockHeight * width
                                + k * blockWidth);
            int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                                + cur_D * outHeight * outWidth 
                                + (MAX_LINE - hPad1) * outWidth 
                                + MAX_LINE - wPad1
                                + j * blockHeight * outWidth
                                + k * blockWidth);
            int64_t calWidth = blockWidth;
            if (k == gmLoop_Width - 1) {
                calWidth = width - MAX_LINE -MAX_LINE - k * blockWidth;
            }

            CopyInSmall(gmXOffset, calHeight, calWidth, (width - calWidth) * sizeof(T));
            ComputeCopy((calHeight) * CeilAlign(calWidth, MAX_LINE));
            CopyOutSmall(gmYOffset, 0, isAtomicAdd, calHeight, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
        }
    } 
}

#endif  // REFLECTION_PAD3D_GRAD_BIG_H