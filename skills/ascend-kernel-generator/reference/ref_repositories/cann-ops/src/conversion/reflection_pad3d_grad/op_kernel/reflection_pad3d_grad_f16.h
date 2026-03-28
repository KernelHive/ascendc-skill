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
 * @file reflection_pad3d_grad_f16.h
 */
#ifndef REFLECTION_PAD3D_GRAD_F16_H
#define REFLECTION_PAD3D_GRAD_F16_H
#include "reflection_pad3d_grad_utils.h"
using namespace AscendC;


template <typename T>
class ReflectionPad3dGradF16 {
public:
    const static int32_t BUFFER_NUM = 2;
    const static uint32_t BLOCK_BYTES = 32;
    const static uint32_t MAX_LINE = 16;
    const static uint32_t MAX_COPY = 256;
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, 1> float32Buf;
    TBuf<TPosition::VECCALC> transposeBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;  
    GlobalTensor<float> workspaceGm;
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
    uint32_t WORK_SPACE_PART = 32;
    uint32_t loopNC = 0;
    int64_t ncOffset = 0; 
    uint32_t curDepth;
    uint32_t curOutDepth;

public:
    __aicore__ inline ReflectionPad3dGradF16() {}

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
        workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWS) + (alignHeight * alignWidth) * blockIdx);
        InitGlobalMemory<float>(workspaceGm, alignHeight * alignWidth, (float)0.0);
        SyncAll();
        pipe.InitBuffer(inQueueX, BUFFER_NUM, (ubFactorElement * sizeof(T)));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, (ubFactorElement * sizeof(T)));
        pipe.InitBuffer(float32Buf, 1, (ubFactorElement * sizeof(float)) );
        pipe.InitBuffer(transposeBuf, (ubFactorElement * sizeof(float)) );
    }

    __aicore__ inline void SmallProcess() {
        int64_t gmXOffset = 0; 
        int64_t gmYOffset = 0; 
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        for ( size_t loop = 0; loop < loopNC; loop++ ) {
            for (size_t i = dPad1; i < curDepth - dPad2; i++) {
                int mirrorIndex1 = GetMirror1(i);
                int mirrorIndex2 = GetMirror2(i);
                if(mirrorIndex1 != -1) {
                    SmallBasicToWgm(loop, mirrorIndex1, false);
                }
                if (mirrorIndex2 != -1) {
                    SmallBasicToWgm(loop, mirrorIndex2, mirrorIndex1 == -1 ? false : true);
                }
                if (mirrorIndex1 == -1 && mirrorIndex2 == -1) {
                     SmallBasic(loop, i, false);
                } else {
                    SmallBasicToWgm(loop, i, true); 
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                    CopyWgmToYgm(loop, i);
                }
            }
        }
    }

    __aicore__ inline void FlatProcess() {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        if (height <= MAX_LINE + MAX_LINE) {
            for ( size_t loop = 0; loop < loopNC; loop++ ) {
                for (size_t i = dPad1; i < curDepth - dPad2; i++) {
                    int mirrorIndex1 = GetMirror1(i);
                    int mirrorIndex2 = GetMirror2(i);
                    if(mirrorIndex1 != -1) {
                         ProcessTopWidthToWgm(loop, mirrorIndex1, 0, height, false);
                    }
                    if (mirrorIndex2 != -1) {
                        ProcessTopWidthToWgm(loop, mirrorIndex2, 0, height, mirrorIndex1 == -1 ? false : true);
                    }
                    if (mirrorIndex1 == -1 && mirrorIndex2 == -1) {
                        ProcessTopWidth(loop, i, 0, height, false);
                    } else {
                        ProcessTopWidthToWgm(loop, i, 0, height, true);
                        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                        CopyWgmToYgmFlat(loop, i);
                    }
                }
            }
        } else {
            for ( size_t loop = 0; loop < loopNC; loop++ ) {
                for (size_t i = dPad1; i < curDepth - dPad2; i++) {
                    int mirrorIndex1 = GetMirror1(i);
                    int mirrorIndex2 = GetMirror2(i);
                    if(mirrorIndex1 != -1) {
                         FlatProcessHeighToWgm(loop, mirrorIndex1, 0, width, false);
                    }
                    if (mirrorIndex2 != -1) {
                        FlatProcessHeighToWgm(loop, mirrorIndex2, 0, width, mirrorIndex1 == -1 ? false : true);
                    }
                    if (mirrorIndex1 == -1 && mirrorIndex2 == -1) {
                         FlatProcessHeight(loop, i, 0, width, false); 
                    } else {
                        FlatProcessHeighToWgm(loop, i, 0, width, true);
                        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                        CopyWgmToYgmFlat(loop, i);
                    }
                }
            }
        }
    }

    __aicore__ inline void BigProcess() {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        for ( size_t loop = 0; loop < loopNC; loop++ ) {
            for (size_t i = dPad1; i < curDepth - dPad2; i++) {
                int mirrorIndex1 = GetMirror1(i);
                int mirrorIndex2 = GetMirror2(i);
                if(mirrorIndex1 != -1) {
                    BigProcessToWgmBasic(loop, mirrorIndex1, false);
                }
                if (mirrorIndex2 != -1) {
                    BigProcessToWgmBasic(loop, mirrorIndex2,  mirrorIndex1 == -1 ? false : true);
                }
                if (mirrorIndex1 == -1 && mirrorIndex2 == -1) {
                    BigProcessBasic(loop, i, false);
                } else {
                    BigProcessToWgmBasic(loop, i, true);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                    CopyWgmToYgmBig(loop, i);
                }
            }
        }
    }

private:

    __aicore__ inline int GetMirror1(size_t i) { //dPad1 == dPad2 == 0 => return -1
        int dis = i - dPad1;
        int mirrorI = dPad1 - dis;
        if (mirrorI < 0 || mirrorI == i) {
            return -1;
        } else {
            return mirrorI;
        }  
    }

    __aicore__ inline int GetMirror2(size_t i) { //dPad1 == dPad2 == 0 => return -1
        int dis = (depth - dPad2 - 1) - i;
        int mirrorI = (depth - dPad2 - 1) + dis;
         if (mirrorI > depth - 1 || mirrorI == i) {
            return -1;
        } else {
            return mirrorI;
        } 
    }

    __aicore__ inline int GetCurD (size_t i) {
        size_t cur_D = i;
        if (i <= dPad1) {   
            cur_D = dPad1 - i;  
        } else if ( i > dPad1 && i <depth - dPad2){ 
            cur_D = i - dPad1;  
        } else if (i >= depth - dPad2){     
            cur_D = (depth - dPad2 - 1) - (i - (depth - dPad2) + 1) - dPad1;   
        }
        return cur_D;
    }

    __aicore__ inline void BigProcessBasic (size_t loop,  size_t i, bool isAtomicAdd) {
        ProcessTopWidth(loop, i, 1, MAX_LINE, isAtomicAdd);
        ProcessBottomWidth(loop, i, MAX_LINE, isAtomicAdd);
        ProcessLeftHeightMid(loop, i, 1, MAX_LINE, isAtomicAdd);
        ProcessRightHeightMid(loop, i, MAX_LINE, isAtomicAdd);
        ProcessMid(loop, i, isAtomicAdd);
    }

    __aicore__ inline void BigProcessToWgmBasic (size_t loop,  size_t i, bool isAtomicAdd) {
        ProcessTopWidthToWgm(loop, i, 1, MAX_LINE, isAtomicAdd);
        ProcessBottomWidthToWgm(loop, i, MAX_LINE, isAtomicAdd);
        ProcessLeftHeightMidToWgm(loop, i, 1, MAX_LINE, isAtomicAdd);
        ProcessRightHeightMidToWgm(loop, i, MAX_LINE, isAtomicAdd);
        ProcessMidToWgm(loop, i, isAtomicAdd);
    }

    // FLAT AND BIG 
    __aicore__ inline void  ProcessTopWidth(size_t loop,  size_t i, int hPad2Mask, int32_t calH, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(calH, MAX_LINE)), MAX_LINE);
        int32_t gmLoop_Width = CeilDiv(width - MAX_LINE -MAX_LINE, blockWidth);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        // first
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width);
        int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                            + cur_D * outHeight * outWidth);

        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeToy(0, hPad2Mask, 0, 1, calH, MAX_LINE);
        CopyOutYgm(gmYOffset, hPad1 * MAX_LINE, isAtomicAdd, calH - hPad1 - hPad2 * (1 - hPad2Mask), MAX_LINE - wPad1, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad1)) * sizeof(T));
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
            CopyInXgm(gmXOffset, calH, calWidth, (width - calWidth) * sizeof(T));
            ComputeToy(0, hPad2Mask, 1, 1, calH, calWidth);
            CopyOutYgm(gmYOffset, hPad1 * CeilAlign(calWidth, MAX_LINE), isAtomicAdd, calH - hPad1 - hPad2 * (1 - hPad2Mask), calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
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
        
        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeToy(0, hPad2Mask, 1, 0, calH, MAX_LINE);
        CopyOutYgm(gmYOffset, hPad1 * MAX_LINE, isAtomicAdd, calH - hPad1 - hPad2 * (1 - hPad2Mask), MAX_LINE - wPad2, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad2)) * sizeof(T));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void  ProcessTopWidthToWgm(size_t loop,  size_t i, int hPad2Mask, int32_t calH, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(calH, MAX_LINE)), MAX_LINE);
        int32_t gmLoop_Width = CeilDiv(width - MAX_LINE -MAX_LINE, blockWidth);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        // first
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width);
        int64_t gmWOffset = 0;

        int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                                    + cur_D * outHeight * outWidth);

        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeTow(0, hPad2Mask, 0, 1, calH, MAX_LINE);
        CopyOutWgm(gmWOffset, hPad1 * MAX_LINE, isAtomicAdd, calH - hPad1 - hPad2 * (1 - hPad2Mask), MAX_LINE - wPad1, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad1)) * sizeof(float));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        
        //mid
        for (size_t j = 0; j < gmLoop_Width; j++) {
            gmXOffset = (loop * curDepth * height * width
                        + i * height * width
                        + MAX_LINE
                        + j * blockWidth);
            gmWOffset = ((MAX_LINE - wPad1)
                        + j * blockWidth);
            int64_t calWidth = blockWidth;
            if (j == gmLoop_Width - 1) {
                calWidth = width - MAX_LINE -MAX_LINE - j * blockWidth;
            }
            CopyInXgm(gmXOffset, calH, calWidth, (width - calWidth) * sizeof(T));
            ComputeTow(0, hPad2Mask, 1, 1, calH, calWidth);
            CopyOutWgm(gmWOffset, hPad1 * CeilAlign(calWidth, MAX_LINE), isAtomicAdd, calH - hPad1 - hPad2 * (1 - hPad2Mask), calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(float));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
        //  last
        gmXOffset = (loop * curDepth * height * width
                    + i * height * width
                    + width - MAX_LINE);
        gmWOffset = (outWidth - (MAX_LINE - wPad2));
        
        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeTow(0, hPad2Mask, 1, 0, calH, MAX_LINE);
        CopyOutWgm(gmWOffset, hPad1 * MAX_LINE, isAtomicAdd, calH - hPad1 - hPad2 * (1 - hPad2Mask), MAX_LINE - wPad2, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad2)) * sizeof(float));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void ProcessBottomWidth(size_t loop,  size_t i, int32_t calH, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
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

        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeToy(1, 0, 0, 1, calH, MAX_LINE);
        CopyOutYgm(gmYOffset, 0, isAtomicAdd, calH - hPad2, MAX_LINE - wPad1, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad1)) * sizeof(T));
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
            CopyInXgm(gmXOffset, calH, calWidth, (width - calWidth) * sizeof(T));
            ComputeToy(1, 0, 1, 1, calH, calWidth);
            CopyOutYgm(gmYOffset, 0, isAtomicAdd, calH - hPad2, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
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
        
        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeToy(1, 0, 1, 0, calH, MAX_LINE);
        CopyOutYgm(gmYOffset, 0, isAtomicAdd, calH - hPad2, MAX_LINE - wPad2, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad2)) * sizeof(T));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void ProcessBottomWidthToWgm(size_t loop,  size_t i, int32_t calH, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(calH, MAX_LINE)), MAX_LINE);
        int32_t gmLoop_Width = CeilDiv(width - MAX_LINE -MAX_LINE, blockWidth);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        // first
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width
                            + (height - MAX_LINE) * width);
        int64_t gmWOffset = ((outHeight - (MAX_LINE - hPad2)) * outWidth);

        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeTow(1, 0, 0, 1, calH, MAX_LINE);
        CopyOutWgm(gmWOffset, 0, isAtomicAdd, calH - hPad2, MAX_LINE - wPad1, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad1)) * sizeof(float));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        //mid
        for (size_t j = 0; j < gmLoop_Width; j++) {
            gmXOffset = (loop * curDepth * height * width
                        + i * height * width
                        + (height - MAX_LINE) * width
                        + MAX_LINE
                        + j * blockWidth);
            gmWOffset = ((outHeight - (MAX_LINE - hPad2)) * outWidth
                        + (MAX_LINE - wPad1)
                        + j * blockWidth);
            int64_t calWidth = blockWidth;
            if (j == gmLoop_Width - 1) {
                calWidth = width - MAX_LINE -MAX_LINE - j * blockWidth;
            }
            CopyInXgm(gmXOffset, calH, calWidth, (width - calWidth) * sizeof(T));
            ComputeTow(1, 0, 1, 1, calH, calWidth);
            CopyOutWgm(gmWOffset, 0, isAtomicAdd, calH - hPad2, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(float));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
        //  last
        
        gmXOffset = (loop * curDepth * height * width
                    + i * height * width
                    + (height - MAX_LINE) * width
                    + width - MAX_LINE);
        gmWOffset = ((outHeight - (MAX_LINE - hPad2)) * outWidth
                    + outWidth - (MAX_LINE - wPad2));
        
        CopyInXgm(gmXOffset, calH, MAX_LINE, (width - MAX_LINE) * sizeof(T));
        ComputeTow(1, 0, 1, 0, calH, MAX_LINE);
        CopyOutWgm(gmWOffset, 0, isAtomicAdd, calH - hPad2, MAX_LINE - wPad2, CeilAlign(MAX_LINE, MAX_LINE), (outWidth - (MAX_LINE -wPad2)) * sizeof(float));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void  FlatProcessHeight(size_t loop,  size_t i, int wPad2Mask, int32_t calW, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        // first
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width);
        int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                            + cur_D * outHeight * outWidth);
        CopyInXgm(gmXOffset, MAX_LINE, width, 0);
        ComputeToy(0, 1, 0, 0, MAX_LINE, width);
        CopyOutYgm(gmYOffset, hPad1 * CeilAlign(width, MAX_LINE), isAtomicAdd, MAX_LINE - hPad1, outWidth, CeilAlign(width, MAX_LINE), 0);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        //mid
        ProcessLeftHeightMid(loop, i, 0, width, isAtomicAdd);    //起始地址默认是16

        //  last
        gmXOffset = (loop * curDepth * height * width
                    + i * height * width
                    + (height - MAX_LINE) * width);
        gmYOffset = (loop *  curOutDepth * outHeight * outWidth 
                    + cur_D *  outHeight * outWidth
                    + (outHeight - (MAX_LINE - hPad2)) * outWidth);
        
        CopyInXgm(gmXOffset, MAX_LINE, width, 0);
        ComputeToy(1, 0, 0, 0, MAX_LINE, width);
        CopyOutYgm(gmYOffset, 0, isAtomicAdd, MAX_LINE - hPad2, outWidth, CeilAlign(width, MAX_LINE), 0);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void  FlatProcessHeighToWgm(size_t loop,  size_t i, int wPad2Mask, int32_t calW, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        // first
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width);
        int64_t gmWOffset = 0;
        CopyInXgm(gmXOffset, MAX_LINE, width, 0);
        ComputeTow(0, 1, 0, 0, MAX_LINE, width);
        CopyOutWgm(gmWOffset, hPad1 * CeilAlign(width, MAX_LINE), isAtomicAdd, MAX_LINE - hPad1, outWidth, CeilAlign(width, MAX_LINE), 0);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        //mid
        ProcessLeftHeightMidToWgm(loop, i, 0, width, isAtomicAdd);    //起始地址默认是16

        //  last
        gmXOffset = (loop * curDepth * height * width
                    + i * height * width
                    + (height - MAX_LINE) * width);
        gmWOffset = ((outHeight - (MAX_LINE - hPad2)) * outWidth);
        
        CopyInXgm(gmXOffset, MAX_LINE, width, 0);
        ComputeTow(1, 0, 0, 0, MAX_LINE, width);
        CopyOutWgm(gmWOffset, 0, isAtomicAdd, MAX_LINE - hPad2, outWidth, CeilAlign(width, MAX_LINE), 0);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    // FLAT AND BIG 
    __aicore__ inline void  ProcessLeftHeightMid(size_t loop,  size_t i, int wPad2Mask, int32_t calW, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
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
            CopyInXgm(gmXOffset, calHeight, calW, (width - calW) * sizeof(T));
            ComputeToy(1, 1, 0, wPad2Mask, calHeight, calW);
            CopyOutYgm(gmYOffset, 0, isAtomicAdd, calHeight, calW - wPad1 - wPad2 * (1 - wPad2Mask), CeilAlign(calW, MAX_LINE), (outWidth - (calW -wPad1 - wPad2 * (1 - wPad2Mask))) * sizeof(T));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }
    
    __aicore__ inline void  ProcessLeftHeightMidToWgm(size_t loop,  size_t i, int wPad2Mask, int32_t calW, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        int32_t blockHeight = FloorDiv(ubFactorElement, CeilAlign(calW, MAX_LINE));
        int32_t gmLoop_Height = CeilDiv(height - MAX_LINE -MAX_LINE, blockHeight);

        for (size_t j = 0; j < gmLoop_Height; j++) {
            int64_t gmXOffset = (loop * curDepth * height * width
                                + i * height * width
                                + MAX_LINE * width
                                + j * blockHeight * width);
            int64_t gmWOffset = (( MAX_LINE - hPad1) * outWidth
                                + j * blockHeight * outWidth);
            int64_t calHeight = blockHeight;
            if (j == gmLoop_Height - 1) {
                calHeight = height - MAX_LINE -MAX_LINE - j * blockHeight;
            }
            CopyInXgm(gmXOffset, calHeight, calW, (width - calW) * sizeof(T));
            ComputeTow(1, 1, 0, wPad2Mask, calHeight, calW);
            CopyOutWgm(gmWOffset, 0, isAtomicAdd, calHeight, calW - wPad1 - wPad2 * (1 - wPad2Mask), CeilAlign(calW, MAX_LINE), (outWidth - (calW -wPad1 - wPad2 * (1 - wPad2Mask))) * sizeof(float));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }

    __aicore__ inline void ProcessRightHeightMid(size_t loop,  size_t i, int32_t calW, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
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
            CopyInXgm(gmXOffset, calHeight, calW, (width - calW) * sizeof(T));
            ComputeToy(1, 1, 1, 0, calHeight, calW);
            CopyOutYgm(gmYOffset, 0, isAtomicAdd, calHeight, calW - wPad2, CeilAlign(calW, MAX_LINE), (outWidth - (calW -wPad2)) * sizeof(T));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }

    __aicore__ inline void ProcessRightHeightMidToWgm(size_t loop,  size_t i, int32_t calW, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        int32_t blockHeight = FloorDiv(ubFactorElement, CeilAlign(calW, MAX_LINE));
        int32_t gmLoop_Height = CeilDiv(height - MAX_LINE -MAX_LINE, blockHeight);

        for (size_t j = 0; j < gmLoop_Height; j++) {
            int64_t gmXOffset = (loop * curDepth * height * width
                                + i * height * width
                                + MAX_LINE * width
                                + width - MAX_LINE
                                + j * blockHeight * width);
            int64_t gmWOffset = (( MAX_LINE - hPad1) * outWidth
                                + outWidth  - (MAX_LINE - wPad2)
                                + j * blockHeight * outWidth);
            int64_t calHeight = blockHeight;
            if (j == gmLoop_Height - 1) {
                calHeight = height - MAX_LINE -MAX_LINE - j * blockHeight;
            }
            CopyInXgm(gmXOffset, calHeight, calW, (width - calW) * sizeof(T));
            ComputeTow(1, 1, 1, 0, calHeight, calW);
            CopyOutWgm(gmWOffset, 0, isAtomicAdd, calHeight, calW - wPad2, CeilAlign(calW, MAX_LINE), (outWidth - (calW -wPad2)) * sizeof(float));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }
    }

    __aicore__ inline void ProcessMid(size_t loop,  size_t i, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
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

                CopyInXgm(gmXOffset, calHeight, calWidth, (width - calWidth) * sizeof(T));
                ComputeX2Y((calHeight) * CeilAlign(calWidth, MAX_LINE));
                CopyOutYgm(gmYOffset, 0, isAtomicAdd, calHeight, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
            }
        } 
    }

    __aicore__ inline void ProcessMidToWgm(size_t loop,  size_t i, bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
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
                int64_t gmWOffset = ((MAX_LINE - hPad1) * outWidth 
                                    + MAX_LINE - wPad1
                                    + j * blockHeight * outWidth
                                    + k * blockWidth);
                int64_t calWidth = blockWidth;
                if (k == gmLoop_Width - 1) {
                    calWidth = width - MAX_LINE -MAX_LINE - k * blockWidth;
                }

                CopyInXgm(gmXOffset, calHeight, calWidth, (width - calWidth) * sizeof(T));
                ComputeX2Float((calHeight) * CeilAlign(calWidth, MAX_LINE));
                CopyOutWgm(gmWOffset, 0, isAtomicAdd, calHeight, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(float));
            }
        } 
    }

    __aicore__ inline void ComputeX2Y(const int32_t totalData) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        Copy(yLocal, xLocal, MAX_COPY / sizeof(T), CeilDiv(totalData, MAX_COPY / sizeof(T)), {1, 1, 8, 8}); 
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeX2Float(const int32_t totalData) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> float32Tensor = float32Buf.AllocTensor<float>();
        Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
        float32Buf.EnQue(float32Tensor);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void SmallBasic(size_t loop, size_t i,  bool isAtomicAdd) {
        size_t cur_D = GetCurD(i); 
        int64_t gmXOffset = (loop * curDepth * height * width
                    + i * height * width);
        int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                    + cur_D * outHeight * outWidth);
        CopyInXgm(gmXOffset, height, width, 0);
        ComputeToy(0, 0, 0, 0, height, width);
        CopyOutYgm(gmYOffset, hPad1 * alignWidth, isAtomicAdd, outHeight, outWidth, alignWidth, 0);
        pipe_barrier(PIPE_MTE3);
    }

    __aicore__ inline void SmallBasicToWgm(size_t loop, size_t i, const bool isAtomicAdd) {
        int64_t gmXOffset = (loop * curDepth * height * width
                            + i * height * width);
        CopyInXgm(gmXOffset, height, width, 0);
        ComputeTow(0, 0, 0, 0, height, width);
        CopyOutWgm(0, hPad1 * alignWidth, isAtomicAdd, outHeight, outWidth, alignWidth, 0);
        pipe_barrier(PIPE_MTE3);
    }

    __aicore__ inline void CopyInXgm(const int64_t offset, const int64_t calH, const int64_t calW, const int64_t srcStride) {
        LocalTensor<T> dstLocal = inQueueX.AllocTensor<T>();
        int64_t alignCalW = CeilAlign(calW, perBlockCount);
        int64_t alignTransCalW = CeilAlign(calW, 16);
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {true, 0, 0, 0};
        copyParams.blockCount = calH;
        copyParams.blockLen = calW * sizeof(T);
        copyParams.srcStride = srcStride;
        copyParams.dstStride = ((alignTransCalW - alignCalW)) *sizeof(T) / 32; //((alignWidth - alignCalW)) *sizeof(T) / 32;
        padParams.isPad = true;
        padParams.rightPadding =  alignCalW - calW;
        DataCopyPad(dstLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(dstLocal);
    }

    __aicore__ inline void CopyInWgm(const int64_t offset, const int64_t calH, const int64_t calW, const int64_t srcStride) {
        LocalTensor<float> dstLocal = float32Buf.AllocTensor<float>();
        int perBlockCountFloat = perBlockCount = BLOCK_BYTES / sizeof(float);
        int64_t alignCalW = CeilAlign(calW, perBlockCountFloat);
        int64_t alignTransCalW = CeilAlign(calW, 16);
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        DataCopyPadExtParams<float> padParams = {true, 0, 0, 0};
        copyParams.blockCount = calH;
        copyParams.blockLen = calW * sizeof(float);
        copyParams.srcStride = srcStride;
        copyParams.dstStride = ((alignTransCalW - alignCalW)) *sizeof(float) / 32; //((alignWidth - alignCalW)) *sizeof(T) / 32;
        padParams.isPad = true;
        padParams.rightPadding =  alignCalW - calW;
        DataCopyPad(dstLocal, workspaceGm[offset], copyParams, padParams);
        float32Buf.EnQue(dstLocal);
    }

    __aicore__ inline void CopyWgmToYgm(size_t loop, size_t i) {
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        size_t cur_D = GetCurD(i); 
        int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                    + cur_D * outHeight * outWidth);
        CopyInWgm(0, outHeight, outWidth, 0);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        LocalTensor<float> float32Tensor = float32Buf.DeQue<float>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, alignHeight * alignWidth);
        outQueueY.EnQue(yLocal);
        float32Buf.FreeTensor(float32Tensor);
        CopyOutYgm(gmYOffset, 0, true, outHeight, outWidth, CeilAlign(outWidth, MAX_LINE), 0);
    }

   __aicore__ inline void CopyWgmToYgmFlat(size_t loop, size_t i) {
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        size_t cur_D = GetCurD(i); 
        int32_t blockHeight = outHeight;
        int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(blockHeight, MAX_LINE)), MAX_LINE);
        if (width <= MAX_LINE + MAX_LINE ) {
            blockWidth = outWidth;
            blockHeight = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(blockWidth, MAX_LINE)), MAX_LINE);
        }
        int32_t gmLoop_Width = CeilDiv(outWidth, blockWidth);
        int32_t gmLoop_Height = CeilDiv(outHeight, blockHeight);

        for (size_t j = 0; j < gmLoop_Height; j ++ ) {
            int64_t calHeight = blockHeight;
            if (j == gmLoop_Height - 1) {
                calHeight = outHeight - j * blockHeight;
            }          
            for (size_t k = 0; k < gmLoop_Width; k ++) {
                int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                                    + cur_D * outHeight * outWidth 
                                    + j * blockHeight * outWidth
                                    + k * blockWidth);
                int64_t gmWOffset = (j * blockHeight * outWidth
                                    + k * blockWidth);
                int64_t calWidth = blockWidth;
                if (k == gmLoop_Width - 1) {
                    calWidth = outWidth - k * blockWidth;
                }

                CopyInWgm(gmWOffset, calHeight, calWidth, (outWidth - calWidth) * sizeof(float));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                LocalTensor<float> float32Tensor = float32Buf.DeQue<float>();
                LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
                Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, calHeight * CeilAlign(calWidth, MAX_LINE));
                outQueueY.EnQue(yLocal);
                float32Buf.FreeTensor(float32Tensor);
                CopyOutYgm(gmYOffset, 0, false, calHeight, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            }
        } 
    }

    __aicore__ inline void CopyWgmToYgmBig(size_t loop, size_t i) {
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        size_t cur_D = GetCurD(i); 
        int32_t blockHeight = 32;
        int32_t blockWidth = FloorAlign(FloorDiv(ubFactorElement, CeilAlign(blockHeight, MAX_LINE)), MAX_LINE);
        int32_t gmLoop_Width = CeilDiv(outWidth, blockWidth);
        int32_t gmLoop_Height = CeilDiv(outHeight, blockHeight);

        for (size_t j = 0; j < gmLoop_Height; j ++ ) {
            int64_t calHeight = blockHeight;
            if (j == gmLoop_Height - 1) {
                calHeight = outHeight - j * blockHeight;
            }          
            for (size_t k = 0; k < gmLoop_Width; k ++) {
                int64_t gmYOffset = (loop * curOutDepth * outHeight * outWidth 
                                    + cur_D * outHeight * outWidth 
                                    + j * blockHeight * outWidth
                                    + k * blockWidth);
                int64_t gmWOffset = (j * blockHeight * outWidth
                                    + k * blockWidth);
                int64_t calWidth = blockWidth;
                if (k == gmLoop_Width - 1) {
                    calWidth = outWidth - k * blockWidth;
                }

                CopyInWgm(gmWOffset, calHeight, calWidth, (outWidth - calWidth) * sizeof(float));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                LocalTensor<float> float32Tensor = float32Buf.DeQue<float>();
                LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
                Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, calHeight * CeilAlign(calWidth, MAX_LINE));
                outQueueY.EnQue(yLocal);
                float32Buf.FreeTensor(float32Tensor);
                CopyOutYgm(gmYOffset, 0, false, calHeight, calWidth, CeilAlign(calWidth, MAX_LINE), (outWidth - calWidth) * sizeof(T));
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            }
        } 
    }

    __aicore__ inline void CopyOutYgm(const int64_t offset, const int64_t srcOffset, const bool isAtomicAdd, const int32_t calH, const int32_t calW, const int32_t alignTransCalW, const int32_t dstStride) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        copyParams.blockCount = calH ;
        copyParams.blockLen = calW * sizeof(T);
        copyParams.srcStride = (alignTransCalW - calW) * sizeof(T) / 32;   //srcStride 为啥不用设置？ ub里面存储的是 alignCalW = 208， calW = 200， （208 - 200）* sizeof(half) / 32 = 0
        copyParams.dstStride = dstStride;
        if (isAtomicAdd == true) {
            SetAtomicAdd<T>();
            DataCopyPad(yGm[offset], yLocal[srcOffset], copyParams);
            SetAtomicNone();
        }else {
            DataCopyPad(yGm[offset], yLocal[srcOffset], copyParams);
        }  
        outQueueY.FreeTensor(yLocal);
    }
    
    __aicore__ inline void CopyOutWgm(const int64_t offset, const int64_t srcOffset, const bool isAtomicAdd, const int32_t calH, const int32_t calW, const int32_t alignTransCalW, const int32_t dstStride) {
        LocalTensor<float> wLocal = float32Buf.DeQue<float>();
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        copyParams.blockCount = calH ;
        copyParams.blockLen = calW * sizeof(float);
        copyParams.srcStride = (alignTransCalW - calW) * sizeof(float) / 32;   //srcStride 为啥不用设置？ ub里面存储的是 alignCalW = 208， calW = 200， （208 - 200）* sizeof(half) / 32 = 0
        copyParams.dstStride = dstStride;
        if (isAtomicAdd == true) {
            SetAtomicAdd<float>();
            DataCopyPad(workspaceGm[offset], wLocal[srcOffset], copyParams);
            SetAtomicNone();
        }else {
            DataCopyPad(workspaceGm[offset], wLocal[srcOffset], copyParams);
        }  
        float32Buf.FreeTensor(wLocal);
    }

    __aicore__ inline void ComputeToy(size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask, size_t wPad2Mask, const int32_t calH, const int32_t calW) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> float32Tensor = float32Buf.AllocTensor<float>();
        int32_t alignHeight = CeilAlign(calH, 16);
        int32_t alignWidth = CeilAlign(calW, 16);
        LocalTensor<float> tLocal = transposeBuf.Get<float>();
        int32_t totalData = alignHeight * alignWidth; 
        Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
        ComputeBasic<float>(tLocal, float32Tensor, hPad1Mask, hPad2Mask, wPad1Mask, wPad2Mask, calH, calW);
        Transose<float>(float32Tensor, tLocal, alignWidth, alignHeight);
        Cast(yLocal, float32Tensor, RoundMode::CAST_RINT, totalData);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        float32Buf.FreeTensor(float32Tensor);
    }

    __aicore__ inline void ComputeTow(size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask, size_t wPad2Mask, const int32_t calH, const int32_t calW) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> float32Tensor = float32Buf.AllocTensor<float>();
        int32_t alignHeight = CeilAlign(calH, 16);
        int32_t alignWidth = CeilAlign(calW, 16);
        LocalTensor<float> tLocal = transposeBuf.Get<float>();
        int32_t totalData = alignHeight * alignWidth; 
        Cast(float32Tensor, xLocal, RoundMode::CAST_NONE, totalData);
        ComputeBasic<float>(tLocal, float32Tensor, hPad1Mask, hPad2Mask, wPad1Mask, wPad2Mask, calH, calW);
        Transose<float>(float32Tensor, tLocal, alignWidth, alignHeight);
        float32Buf.EnQue(float32Tensor);
        inQueueX.FreeTensor(xLocal);
    }
    template <typename T1>
    __aicore__ inline void ComputeBasic(LocalTensor<T1>& tLocal, LocalTensor<T1>& xLocal, size_t hPad1Mask, size_t hPad2Mask, size_t wPad1Mask, size_t wPad2Mask, const int32_t calH, const int32_t calW) {
        int64_t alignTransCalW = CeilAlign(calW, 16);
        int64_t alignTransCalH = CeilAlign(calH, 16);
        if (hPad1Mask == 0 && hPad1 > 0) {
            for(uint32_t i = 0; i < hPad1; i++) {
                auto srcLocal_1 = xLocal[i * alignTransCalW];
                auto srcLocal_2 = xLocal[(2 * hPad1 - i)  * alignTransCalW];
                Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalW);
            }
        }

        if (hPad2Mask == 0 && hPad2 > 0) {
            for(uint32_t i = 0; i < hPad2; i++) {
                auto srcLocal_1 = xLocal[(calH - 1 - i) * alignTransCalW];
                auto srcLocal_2 = xLocal[(calH - 2*hPad2 - 1 + i)  * alignTransCalW];
                Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalW);
            }
        }
        Transose<T1>(tLocal, xLocal, alignTransCalH, alignTransCalW);
        if (wPad1Mask == 0 &&wPad1 > 0) {
            for(uint32_t i = 0; i < wPad1; i++) {
                auto srcLocal_1 = tLocal[i * alignTransCalH];
                auto srcLocal_2 = tLocal[(2 * wPad1 - i) * alignTransCalH];
                Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalH); 
            } 
        }

        if (wPad2Mask == 0 && wPad2 > 0) {
            for(uint32_t i = 0; i < wPad2; i++) {
                auto srcLocal_1 = tLocal[(calW - 1 - i) * alignTransCalH];
                auto srcLocal_2 = tLocal[(calW - 2 * wPad2 - 1 + i) * alignTransCalH];
                Add(srcLocal_2, srcLocal_2, srcLocal_1, alignTransCalH); 
            } 
        }

        // 平移
        if (wPad1Mask == 0 && wPad1 > 0) {
            for (uint32_t i = 0; i < calW - wPad1; i++){
                auto srcLocal_1 = tLocal[i * alignTransCalH];
                auto srcLocal_2 = tLocal[(i + wPad1) * alignTransCalH];
                Muls(srcLocal_1, srcLocal_2, (T1)1.0, alignTransCalH);
            }
        }
    }
    template <typename T1>
    __aicore__ inline void Transose(LocalTensor<T1>& dstLocal, LocalTensor<T1>& srcLocal, const int32_t calH, const int32_t calW) {
        TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = calH / 16;
        transDataParams.dstRepStride = (16 * sizeof(T1)) / 32;
        transDataParams.srcRepStride = (16 * calW * sizeof(T1)) / 32; 
        if (transDataParams.repeatTimes == 1) {
            transDataParams.dstRepStride = 0;
            transDataParams.srcRepStride = 0;
        }
        // 入参类型是LocalTensor地址值的调用方式，推荐使用
        uint64_t srcLocalList[16];
        uint64_t dstLocalList[16];
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        if constexpr (std::is_same<T1, float>::value) {
            for (int i = 0; i < calW / 8; i ++) {
                for (int j = 0; j < 16; j ++) {
                    srcLocalList[j] = (uint64_t)(srcLocal[srcOffset + calW * j].GetPhyAddr());
                }
                for (int j = 0; j < 8; j++ ) {
                    dstLocalList[2 * j] = (uint64_t)(dstLocal[dstOffset + calH * j].GetPhyAddr());
                    dstLocalList[2 * j + 1] = (uint64_t)(dstLocal[dstOffset + calH * j + 8].GetPhyAddr());
                }
                TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
                srcOffset += 8;
                dstOffset += 8 * calH;
            }
        } else {
            for (int i = 0; i < calW / 16; i ++) {
                for (int j = 0; j < 16; j ++) {
                    srcLocalList[j] = (uint64_t)(srcLocal[srcOffset + calW * j].GetPhyAddr());
                }
                for (int j = 0; j < 16; j++ ) {
                    dstLocalList[j] = (uint64_t)(dstLocal[dstOffset + calH * j].GetPhyAddr());
                }
                TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
                srcOffset += 16;
                dstOffset += 16 * calH;
            }
        }
    }
};

#endif  // REFLECTION_PAD3D_GRAD_F16_H