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
 * \file adaptive_avg_pool3d_grad_nc_large_float.h
 * \brief
 */
 #ifndef ADAPTIVE_AVG_POOL3D_GRAD_NC_LARGE_FLOAT_H 
 #define ADAPTIVE_AVG_POOL3D_GRAD_NC_LARGE_FLOAT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "adaptive_avg_pool3d_grad_common.h"
using namespace AscendC;

template <typename T>
class KernelAdaptiveAvgPool3DGradNcLargeFloat {
public:
    __aicore__ inline KernelAdaptiveAvgPool3DGradNcLargeFloat() {}
    __aicore__ inline void Init(GM_ADDR input_grad, GM_ADDR output_grad, GM_ADDR workspace,
        const AdaptiveAvgPool3dGradTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ncAlign = tiling_data->ncNum;
        outDepth = tiling_data->dOut;
        outHeight = tiling_data->hOut;
        outWidth = tiling_data->wOut;
        inDepth = tiling_data->dIn;
        inHeight = tiling_data->hIn;
        inWidth = tiling_data->wIn;
        coreNum = tiling_data->taskCoreUsed;
        isAtomicAdd = tiling_data->isAtomicAdd;

        ncSliceNum = tiling_data->ncSliceNum;
        ncAlignSliceLength = tiling_data->ncAlignSliceLength;
        ncAlignSliceTail = tiling_data->ncAlignSliceTail;

        taskNumPerCore = tiling_data->taskNumPerCore;
        taskNumLastCore = tiling_data->taskNumLastCore;
        yNumPerCalc = tiling_data->yNumPerCalc;

        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(T);
        clearCoreNum = GetBlockNum();

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
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(inputUb, yNumPerCalc * ncAlignSliceLength * sizeof(T));
        pipe->InitBuffer(outputUb, yNumPerCalc * ncAlignSliceLength * sizeof(T));
    }

    __aicore__ inline void GetLocalTensor()
    {
        inputLocal = inputUb.Get<T>();
        outputLocal = outputUb.Get<T>();
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
        Duplicate(inputLocal, (T)0, ncAlignSliceLength);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (uint64_t taskIdx = clesrStartOffset; taskIdx < clearEndOffset; taskIdx++)
        {
            ncMoveNum = ncAlignSliceLength;
            for (uint64_t ncIdx = 0; ncIdx < ncSliceNum; ncIdx++) {
                if (ncIdx == ncSliceNum - 1) {
                    ncMoveNum = ncAlignSliceTail;
                }
                DataCopyPad(outputGradGm[taskIdx * ncAlign + ncIdx * ncAlignSliceLength], inputLocal, {1, (uint32_t)(ncMoveNum * sizeof(T)), 0, 0, 0});
            }
        }

        if ASCEND_IS_AIV {
            SyncAll();
        }
    } 
        __aicore__ inline void Process()
    {
        if (isAtomicAdd == 1) {
            SetAtomicAdd<T>();
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
        }
    }

private:
    __aicore__ inline void Compute(T divideFactor)
    {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        DataCopyPad(inputLocal, inputGradGm[offsetInputGm],  copyInParamsV1, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(outputLocal, inputLocal, divideFactor, ncMoveNum);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (uint64_t kd = istartD; kd < iendD; kd++) {
            for (uint64_t kh = istartH; kh < iendH; kh++) {
                for (uint64_t kw = istartW; kw < iendW; kw++) {
                DataCopyPad(outputGradGm[(kd * inHeight * inWidth + kh * inWidth + kw) * ncAlign + n * ncAlignSliceLength],
                            outputLocal, {(uint16_t)1, (uint32_t)(ncMoveNum * sizeof(T)), 0, 0, 0});
                }
            }
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGradGm, outputGradGm;
    TBuf<TPosition::VECCALC> inputUb, outputUb, offsetUb;

    uint32_t coreNum, clearCoreNum;
    uint32_t curBlockIdx;
    uint32_t dataAlign, blockBytes = 32;
    uint64_t taskNumPerCore, taskNumLastCore, taskNumThisCore, yNumPerCalc, loopNum, taskNumLastLoop;
    uint64_t ncAlign, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth;
    uint64_t ncSliceNum, ncAlignSliceLength, ncAlignSliceTail, ncMoveNum;
    uint64_t startD, endD, startH, endH, startW, endW;
    uint64_t istartD, iendD, kD, istartH, iendH, kH, istartW, iendW, kW;
    uint64_t startOffset, endOffset, offsetInputGm;
    uint64_t n, w, h, d, isAtomicAdd;
    LocalTensor<T> inputLocal, outputLocal;
    DataCopyExtParams copyInParamsV1, copyInParamsV2, copyInParamsV3, copyOutParams;
    DataCopyPadExtParams<T> padParamsFloat {false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
};

#endif //ADAPTIVE_AVG_POOL3D_GRAD_NC_LARGE_FLOAT_H