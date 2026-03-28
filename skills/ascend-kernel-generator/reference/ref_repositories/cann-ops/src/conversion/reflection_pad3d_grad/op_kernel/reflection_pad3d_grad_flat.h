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
 * @file reflection_pad3d_grad_flat.h
 */
#ifndef REFLECTION_PAD3D_GRAD_FLAT_H
#define REFLECTION_PAD3D_GRAD_FLAT_H
#include "reflection_pad3d_grad_init.h"

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::FlatProcess() {
        if (width < MAX_LINE + MAX_LINE) {
            WidthFlatProcess();
        } else {
            HeightFlatProcess();
        }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::WidthFlatProcess() {
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    int32_t blockHeight = FloorDiv(ubFactorElement, CeilAlign(width, 16));
    int32_t gmLoop_Height = CeilDiv(height - MAX_LINE -MAX_LINE, blockHeight);
    for ( size_t loop = 0; loop < loopNC; loop++ ) {
        for ( size_t i = 0; i < curDepth; i++ ) {
            size_t cur_D = GetCurD(i); 
            bool isAtomicAdd = true;
            // first
            int64_t gmXOffset = (loop * curDepth * height * width
                                + i * height * width);
            int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                                + cur_D * outHeight * outWidth);
            CopyInSmall(gmXOffset, MAX_LINE, width, 0);
            ComputeSmall(0, 1, 0, 0, MAX_LINE, width);
            CopyOutSmall(gmYOffset, hPad1 * CeilAlign(width, 16), isAtomicAdd, MAX_LINE - hPad1, outWidth, CeilAlign(width, 16), 0);
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            //  mid
            for (size_t j = 0; j < gmLoop_Height; j++) {
                gmXOffset = (loop * curDepth * height * width
                            + i * height * width
                            + MAX_LINE * width
                            + j * blockHeight * width);
                gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                            + cur_D * outHeight * outWidth
                            + ( MAX_LINE - hPad1) * outWidth
                            + j * blockHeight * outWidth);
                int64_t calHeight = blockHeight;
                if (j == gmLoop_Height - 1) {
                    calHeight = height - MAX_LINE -MAX_LINE - j * blockHeight;
                }
                CopyInSmall(gmXOffset, calHeight, width, 0);
                ComputeSmall(1, 1, 0, 0, calHeight, width);
                CopyOutSmall(gmYOffset, 0, isAtomicAdd, calHeight, outWidth, CeilAlign(width, 16), 0);
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            }
            //  last
            gmXOffset = (loop * curDepth * height * width
                        + i * height * width
                        + (height - MAX_LINE) * width);
            gmYOffset = (loop *  curOutDepth * outHeight * outWidth 
                        + cur_D *  outHeight * outWidth
                        + (outHeight - (MAX_LINE - hPad2)) * outWidth);
            
            CopyInSmall(gmXOffset, MAX_LINE, width, 0);
            ComputeSmall(1, 0, 0, 0, MAX_LINE, width);
            CopyOutSmall(gmYOffset, 0, isAtomicAdd, MAX_LINE - hPad2, outWidth, CeilAlign(width, 16), 0);
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }
}

template <typename T>
__aicore__ inline void ReflectionPad3dGrad<T>::HeightFlatProcess() {
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(height, 16)), 16);
    int32_t gmLoop_Width = CeilDiv(width - MAX_LINE -MAX_LINE, blockWidth);
    for ( size_t loop = 0; loop < loopNC; loop++ ) {
        for ( size_t i = 0; i < curDepth; i++ ) {
            size_t cur_D = GetCurD(i); 
            bool isAtomicAdd = true;
            // first
            int64_t gmXOffset = (loop * curDepth * height * width
                                + i * height * width);
            int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                                + cur_D * outHeight * outWidth);

            CopyInSmall(gmXOffset, height, MAX_LINE, (width - MAX_LINE) * sizeof(T));
            ComputeSmall(0, 0, 0, 1, height, MAX_LINE);
            CopyOutSmall(gmYOffset, hPad1 * MAX_LINE, isAtomicAdd, outHeight, MAX_LINE - wPad1, CeilAlign(MAX_LINE, 16), (outWidth - (MAX_LINE -wPad1)) * sizeof(T));
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
                CopyInSmall(gmXOffset, height, calWidth, (width - calWidth) * sizeof(T));
                ComputeSmall(0, 0, 1, 1, height, calWidth);
                CopyOutSmall(gmYOffset, hPad1 * CeilAlign(calWidth, 16), isAtomicAdd, outHeight, calWidth, CeilAlign(calWidth, 16), (outWidth - calWidth) * sizeof(T));
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
            
            CopyInSmall(gmXOffset, height, MAX_LINE, (width - MAX_LINE) * sizeof(T));
            ComputeSmall(0, 0, 1, 0, height, MAX_LINE);
            CopyOutSmall(gmYOffset, hPad1 * MAX_LINE, isAtomicAdd, outHeight, MAX_LINE - wPad2, CeilAlign(MAX_LINE, 16), (outWidth - (MAX_LINE -wPad2)) * sizeof(T));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }
}
#endif  // REFLECTION_PAD3D_GRAD_FLAT_H