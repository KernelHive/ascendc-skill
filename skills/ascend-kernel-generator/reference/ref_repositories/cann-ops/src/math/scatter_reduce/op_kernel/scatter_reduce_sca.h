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
 * @file scatter_reduce_sca.cpp
 */
#ifndef SCATTER_REDUCE_SCA_H
#define SCATTER_REDUCE_SCA_H

#include "kernel_operator.h"
using namespace AscendC;


class ScatterDeduceSca {
   public:
    __aicore__ inline ScatterDeduceSca() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR src, GM_ADDR y, int32_t batchSize,
                                int32_t dimSizeX, int32_t dimSizeSrc, int32_t strideSize,
                                int32_t reduction, bool includeSelf, TPipe* pipeIn) {
        this->coreNum = GetBlockNum();
        this->coreIndex = GetBlockIdx();
        this->pipe = pipeIn;

        this->reduction = reduction;
        this->includeSelf = includeSelf;
        this->totalSizeX = batchSize * dimSizeX * strideSize;
        this->totalSizeSrc = batchSize * dimSizeSrc * strideSize;

        this->batchSize = batchSize;
        this->dimSizeX = dimSizeX;
        this->dimSizeSrc = dimSizeSrc;
        this->strideSize = strideSize;

        xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)x, totalSizeX * sizeof(DTYPE_Y));
        indexGm.SetGlobalBuffer((__gm__ DTYPE_INDEX*)index, totalSizeSrc * sizeof(DTYPE_INDEX));
        srcGm.SetGlobalBuffer((__gm__ DTYPE_Y*)src, totalSizeSrc * sizeof(DTYPE_Y));
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalSizeX * sizeof(DTYPE_Y));

        pipe->InitBuffer(TmpBuf, dimSizeX * sizeof(DTYPE_Y));
        pipe->InitBuffer(CntBuf, dimSizeX * sizeof(int32_t));

        pipe->InitBuffer(ABuf, 128 * sizeof(DTYPE_Y));
        pipe->InitBuffer(BBuf, 128);
    }

    __aicore__ inline float Reduce(float a, float b) {
        switch (reduction) {
            case 0:
                return a + b;
            case 1:
                return a * b;
            case 2:
                return a + b;
            case 3:
                return a > b ? a : b;
            case 4:
                return a < b ? a : b;
            default:
                return a;
        }
    }

    __aicore__ inline half ReduceHalf(half a, half b) {
        auto aBuf = TmpBuf.Get<half>();
        auto bBuf = CntBuf.Get<half>();
        switch (reduction) {
            case 0:
                aBuf.SetValue(0, a);
                bBuf.SetValue(0, b);
                Add(aBuf, aBuf, bBuf, 1);
                return aBuf.GetValue(0);
            case 1:
                aBuf.SetValue(0, a);
                bBuf.SetValue(0, b);
                Mul(aBuf, aBuf, bBuf, 1);
                return aBuf.GetValue(0);
            case 2:
                aBuf.SetValue(0, a);
                bBuf.SetValue(0, b);
                Add(aBuf, aBuf, bBuf, 1);
                return aBuf.GetValue(0);
            case 3:
                return (float)a > (float)b ? a : b;
            case 4:
                return (float)a < (float)b ? a : b;
            default:
                return 0;
        }
    }

    __aicore__ void Process() {
        auto tmpBuf = TmpBuf.Get<DTYPE_Y>();
        auto cntBuf = CntBuf.Get<int32_t>();
        auto aBuf = ABuf.Get<DTYPE_Y>();
        auto bBuf = BBuf.Get<DTYPE_Y>();

        int batchPointerX, batchPointerSrc;
        DTYPE_Y defaultValue = 0;

        // "sum", "prod", "mean", "amax", "amin"
        switch (reduction) {
            case 0:
                defaultValue = 0;
                break;
            case 1:
                defaultValue = 1;
                break;
            case 2:
                defaultValue = 0;
                break;
            case 3:
                defaultValue = -10000;
                break;
            case 4:
                defaultValue = 10000;
                break;
            default:
                break;
        }

        for (int batchP = 0; batchP < batchSize; batchP++) {
            batchPointerX = batchP * dimSizeX * strideSize;
            batchPointerSrc = batchP * dimSizeSrc * strideSize;
            for (int strideP = 0; strideP < strideSize; strideP++) {
                if (includeSelf) {
                    for (int i = 0; i < dimSizeX; i++) {
                        auto xValue = xGm.GetValue(batchPointerX + i * strideSize + strideP);
                        tmpBuf.SetValue(i, xValue);
                        cntBuf.SetValue(i, 1);
                    }
                } else {
                    for (int i = 0; i < dimSizeX; i++) {
                        tmpBuf.SetValue(i, defaultValue);
                        cntBuf.SetValue(i, 0);
                    }
                }

                for (int i = 0; i < dimSizeSrc; i++) {
                    int idx = indexGm.GetValue(batchPointerSrc + i * strideSize + strideP);
                    DTYPE_Y srcValue = srcGm.GetValue(batchPointerSrc + i * strideSize + strideP);

                    DTYPE_Y xValue = tmpBuf.GetValue(idx);

                    DTYPE_Y res = (DTYPE_Y)Reduce((float)xValue, (float)srcValue);
                    int cnt = cntBuf.GetValue(idx) + 1;

                    cntBuf.SetValue(idx, cnt);
                    tmpBuf.SetValue(idx, res);
                }
                for (int i = 0; i < dimSizeX; i++) {
                    int cnt = cntBuf.GetValue(i);
                    if (cnt != 0) {
                        float tmp = tmpBuf.GetValue(i);
                        if (reduction == 2) {
                                tmp = tmp / cnt;
                        }
                        yGm.SetValue(batchPointerX + i * strideSize + strideP, (DTYPE_Y)tmp);
                    } else {
                        auto xValue = xGm.GetValue(batchPointerX + i * strideSize + strideP);
                        yGm.SetValue(batchPointerX + i * strideSize + strideP, xValue);
                    }
                }
            }
        }
    }

   private:
    TPipe* pipe;
    GlobalTensor<DTYPE_Y> xGm;
    GlobalTensor<DTYPE_INDEX> indexGm;
    GlobalTensor<DTYPE_Y> srcGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t batchSize, dimSizeX, dimSizeSrc, strideSize, reduction, totalSizeX, totalSizeSrc;
    uint32_t coreIndex, coreNum;
    uint16_t rows, cols, setNum;
    bool includeSelf;
    TBuf<QuePosition::VECCALC> TmpBuf, CntBuf, ABuf, BBuf;
};

#endif //  SCATTER_REDUCE_SCA_H