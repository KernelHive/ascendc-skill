/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad_nc_large_cast.h
 * \brief
 */
#ifndef ADAPTIVE_AVG_POOL3D_GRAD_NC_LARGE_CAST_H
#define ADAPTIVE_AVG_POOL3D_GRAD_NC_LARGE_CAST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "adaptive_avg_pool3d_grad_common.h"
using namespace AscendC;

template <typename T>
class KernelAdaptiveAvgPool3DGradNcLargeCast {
public:
    __aicore__ inline KernelAdaptiveAvgPool3DGradNcLargeCast() {}
    __aicore__ inline void Init(GM_ADDR input_grad, GM_ADDR output_grad, GM_ADDR workspace,
        const AdaptiveAvgPool3dGradTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(T);
        clearCoreNum = GetBlockNum();

        taskNumPerCore = tiling_data->taskNumPerCore;
        taskNumLastCore = tiling_data->taskNumLastCore;
        yNumPerCalc = tiling_data->yNumPerCalc;
        isAtomicAdd = tiling_data->isAtomicAdd;

        ncAlign = tiling_data->ncNum;
        inDepth = tiling_data->dIn;
        inHeight = tiling_data->hIn;
        inWidth = tiling_data->wIn;
        outDepth = tiling_data->dOut;
        outHeight = tiling_data->hOut;
        outWidth = tiling_data->wOut;
        coreNum = tiling_data->taskCoreUsed;

        ncSliceNum = tiling_data->ncSliceNum;
        ncAlignSliceLength = tiling_data->ncAlignSliceLength;
        ncAlignSliceTail = tiling_data->ncAlignSliceTail;

        startOffset = curBlockIdx * taskNumPerCore;
        if (curBlockIdx < coreNum - 1) {
            endOffset = (curBlockIdx + 1) * taskNumPerCore;
        } else if (curBlockIdx == coreNum - 1) {
            endOffset = startOffset + taskNumLastCore;
        } else { 
            startOffset = 0;
            endOffset = 0;
        }

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());

        inputGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input_grad), 
                                                ncAlign * outDepth * outHeight * outWidth);
        outputGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output_grad), 
                                                ncAlign * inDepth * inHeight * inWidth);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace, ncAlign * inDepth * inHeight * inWidth);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(inputUb, yNumPerCalc * ncAlignSliceLength * sizeof(T));
        pipe->InitBuffer(outputUb, yNumPerCalc * ncAlignSliceLength * sizeof(T));

        pipe->InitBuffer(inputFloatUb, yNumPerCalc * ncAlignSliceLength * sizeof(float));
        pipe->InitBuffer(outputFloatUb, yNumPerCalc * ncAlignSliceLength * sizeof(float));
    }

    __aicore__ inline void GetLocalTensor()
    {
        inputLocal = inputUb.Get<T>();
        outputLocal = outputUb.Get<T>();
        
        inputLocalFloat = inputFloatUb.Get<float>();
        outputLocalFloat = outputFloatUb.Get<float>();
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    }

    __aicore__ inline void ClearOutput()
    {
        if (isAtomicAdd == 0) {
            return;
        }
        uint64_t taskNum = inDepth * inHeight * inWidth;
        uint64_t clearTaskNumPerCore = DivCeil(taskNum, clearCoreNum);
        uint64_t clesrStartOffset = curBlockIdx * clearTaskNumPerCore;
        uint64_t clearEndOffset = (curBlockIdx + 1) * clearTaskNumPerCore;
        if (clearEndOffset > taskNum) {
            clearEndOffset = taskNum;
        }
        Duplicate(inputLocalFloat, (float)0, yNumPerCalc * ncAlignSliceLength);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (uint64_t taskIdx = clesrStartOffset; taskIdx < clearEndOffset; taskIdx++)
        {
            ncMoveNum = ncAlignSliceLength;
            for (uint64_t ncIdx = 0; ncIdx < ncSliceNum; ncIdx++) {
                if (ncIdx == ncSliceNum - 1) {
                    ncMoveNum = ncAlignSliceTail;
                }
                DataCopyPad(workspaceGm[taskIdx * ncAlign + ncIdx * ncAlignSliceLength], inputLocalFloat, {1, (uint32_t)(ncMoveNum * sizeof(float)), 0, 0, 0});
            }
        }
        if ASCEND_IS_AIV {
            SyncAll();
        }
    }
    
    __aicore__ inline void Process()
    {
        if (isAtomicAdd == 1) {
            SetAtomicAdd<float>();
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        for (uint64_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            n = taskIdx % ncSliceNum;
            w = taskIdx / ncSliceNum % outWidth;
            h = taskIdx / ncSliceNum / outWidth % outHeight;
            d = taskIdx / ncSliceNum / outWidth / outHeight % outDepth;

            istartD = start_index(d, outDepth, inDepth);
            iendD = end_index(d, outDepth, inDepth);
            kD = iendD - istartD;

            istartH = start_index(h, outHeight, inHeight);
            iendH = end_index(h, outHeight, inHeight);
            kH = iendH - istartH;

            istartW = start_index(w, outWidth, inWidth);
            iendW = end_index(w, outWidth, inWidth);
            kW = iendW - istartW;

            offsetInputGm = (d * outHeight * outWidth + h * outWidth + w) * ncAlign + n * ncAlignSliceLength;
            if (n == ncSliceNum -1) {
                ncMoveNum = ncAlignSliceTail;
            } else {
                ncMoveNum = ncAlignSliceLength;
            }
            copyInParamsV1 = {1, (uint32_t)(ncMoveNum * sizeof(T)), 0, 0, 0};

            Compute((float)1.0 / (kD * kH * kW));
        }
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        if (isAtomicAdd == 1) {
            SetAtomicNone();          
            if ASCEND_IS_AIV {
                SyncAll();
            }
            workSpaceCopyOut();
        }
        SetAtomicNone();
    }

private:
    __aicore__ inline void workSpaceCopyOut()
    {
        uint64_t taskNum = inDepth * inHeight * inWidth;
        uint64_t clearTaskNumPerCore = DivCeil(taskNum, clearCoreNum);
        uint64_t clesrStartOffset = curBlockIdx * clearTaskNumPerCore;
        uint64_t clearEndOffset = (curBlockIdx + 1) * clearTaskNumPerCore;
        if (clearEndOffset > taskNum) {
            clearEndOffset = taskNum;
        }

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        for (uint64_t taskIdx = clesrStartOffset; taskIdx < clearEndOffset; taskIdx++) {
            ncMoveNum = ncAlignSliceLength;
            for (uint64_t ncIdx = 0; ncIdx < ncSliceNum; ncIdx++) {
                moveOffset = taskIdx * ncAlign + ncIdx * ncAlignSliceLength;
                if (ncIdx == ncSliceNum - 1) {
                    ncMoveNum = ncAlignSliceTail;
                }
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                DataCopyPad(outputLocalFloat, workspaceGm[moveOffset], {1, (uint32_t)(ncMoveNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                Cast(outputLocal, outputLocalFloat, RoundMode::CAST_RINT, ncMoveNum);
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);

                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                DataCopyPad(outputGradGm[moveOffset], outputLocal, {(uint16_t)1, (uint32_t)(ncMoveNum * sizeof(T)), 0, 0, 0});//kw约束[1, 4095]
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    
    __aicore__ inline void DataCopyOut(uint64_t kd, uint64_t kh, uint64_t kw)
    {
        if (isAtomicAdd == 1) {
            DataCopyPad(workspaceGm[(kd * inHeight * inWidth + kh * inWidth + kw) * ncAlign + n * ncAlignSliceLength],
                outputLocalFloat, {(uint16_t)1, (uint32_t)(ncMoveNum * sizeof(float)), 0, 0, 0});
        } else {
            DataCopyPad(outputGradGm[(kd * inHeight * inWidth + kh * inWidth + kw) * ncAlign + n * ncAlignSliceLength],
                outputLocal, {(uint16_t)1, (uint32_t)(ncMoveNum * sizeof(T)), 0, 0, 0});
        }
    }

    __aicore__ inline void Compute(float divideFactor)
    {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        DataCopyPad(inputLocal, inputGradGm[offsetInputGm],  copyInParamsV1, padParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(inputLocalFloat, inputLocal, RoundMode::CAST_NONE, ncMoveNum);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(outputLocalFloat, inputLocalFloat, divideFactor, ncMoveNum);
        if (isAtomicAdd == 0) {
            Cast(outputLocal, outputLocalFloat, RoundMode::CAST_RINT, ncMoveNum);
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (uint64_t kd = istartD; kd < iendD; kd++) {
            for (uint64_t kh = istartH; kh < iendH; kh++) {
                for (uint64_t kw = istartW; kw < iendW; kw++) {
                    DataCopyOut(kd, kh, kw);
                }
            }
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGradGm, outputGradGm;
    GlobalTensor<float> workspaceGm;
    TBuf<TPosition::VECCALC> inputUb, outputUb, inputFloatUb, outputFloatUb;

    uint32_t coreNum, clearCoreNum;
    uint32_t curBlockIdx;
    uint32_t dataAlign, blockBytes = 32;
    uint64_t taskNumPerCore, taskNumLastCore, yNumPerCalc, loopNum, taskNumLastLoop;
    uint64_t ncSliceNum, ncAlignSliceLength, ncAlignSliceTail, ncMoveNum;
    uint64_t ncAlign, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth;
    uint64_t moveOffset;
    uint64_t startD, endD, startH, endH, startW, endW;
    uint64_t istartD, iendD, kD, istartH, iendH, kH, istartW, iendW, kW;
    uint64_t startOffset, endOffset, offsetInputGm;
    uint64_t n, w, h, d, isAtomicAdd;
    LocalTensor<T> inputLocal, outputLocal;
    LocalTensor<float> inputLocalFloat, outputLocalFloat;
    DataCopyExtParams copyInParamsV1, copyInParamsV2, copyInParamsV3, copyOutParams;
    DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
};

#endif //ADAPTIVE_AVG_POOL3D_GRAD_NC_LARGE_CAST_H