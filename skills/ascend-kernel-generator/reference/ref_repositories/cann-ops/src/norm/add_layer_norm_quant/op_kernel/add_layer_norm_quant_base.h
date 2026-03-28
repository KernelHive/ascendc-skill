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
 * \file add_layer_norm_quant_base.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_QUANT_BASE_CLASS_H_
#define ADD_LAYER_NORM_QUANT_BASE_CLASS_H_

#include "add_layer_norm_quant_helper.h"

#define IS_BIAS_ELEWISE ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormQuantBase {
public:
    __aicore__ inline KernelAddLayerNormQuantBase()
    {}

    __aicore__ inline void InitBaseParams(const AddLayerNormQuantTilingData *tiling)
    {
        this->numCore = tiling->numCore;
        this->numLastDim = tiling->numLastDim;
        this->numFirstDim = tiling->numFirstDim;
        this->firstDimPerCore = tiling->firstDimPerCore;
        this->firstDimPerCoreTail = tiling->firstDimPerCoreTail;
        this->firstDimPerTime = tiling->firstDimPerTime;
        this->lastDimPerTime = tiling->lastDimPerTime;
        this->aveNum = tiling->aveFactor;
        this->eps = tiling->eps;
        this->isXOut = (tiling->isXOut == 1);

        if (block_idx != this->numCore - 1) {
            this->rowWork = this->firstDimPerCore;
            this->rowStep = this->firstDimPerTime;
        } else {
            this->rowWork = this->firstDimPerCoreTail;
            this->rowStep = TWO_NUMS_MIN(this->firstDimPerTime, this->rowWork);
        }
        this->rowTail_ = (this->rowWork % this->rowStep == 0) ? this->rowStep : (this->rowWork % this->rowStep);
        this->gmOffset_ = this->firstDimPerCore * this->numLastDim;

        // some params for Div
        this->repsFp32 = this->numLastDim / ELEM_PER_REP_FP32;
        this->offsetsFp32 = this->repsFp32 * ELEM_PER_REP_FP32;
        this->remsFp32 = this->numLastDim - this->offsetsFp32;

        this->numLastDimAligned = this->numLastDim;
        if (ROUND_UP32(this->numLastDim * sizeof(T)) != this->numLastDim * sizeof(T)) {
            lastDimPad = true;
            this->numLastDimAligned = ROUND_UP32(this->numLastDim * sizeof(T)) / sizeof(T);
        }
        this->numLastDimRoundUp32 = ROUND_UP32(this->numLastDim);
    }

    __aicore__ inline void InitInGlobalTensors(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias)
    {
        x1Gm.SetGlobalBuffer((__gm__ T *)(x1) + block_idx * this->gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T *)(x2) + block_idx * this->gmOffset_);
        if constexpr (IS_BIAS_ELEWISE) {
            biasGm.SetGlobalBuffer((__gm__ T *)(bias) + block_idx * this->gmOffset_);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias);
        }
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma);
        betaGm.SetGlobalBuffer((__gm__ T *)beta);
    }

    __aicore__ inline void InitOutGlobalTensors(GM_ADDR y1, GM_ADDR y2, GM_ADDR x)
    {
        y1Gm.SetGlobalBuffer((__gm__ int8_t *)(y1) + block_idx * this->gmOffset_);
        y2Gm.SetGlobalBuffer((__gm__ int8_t *)(y2) + block_idx * this->gmOffset_);
        xGm.SetGlobalBuffer((__gm__ T *)(x) + block_idx * this->gmOffset_);
    }

    __aicore__ inline void InitWorkSpaceGlobalTensors(GM_ADDR workspace)
    {}

protected:
    GlobalTensor<T> x1Gm, x2Gm, gammaGm, betaGm, biasGm, xGm;
    GlobalTensor<int8_t> y1Gm, y2Gm;

    uint32_t numCore;
    uint32_t numLastDim;
    uint32_t numFirstDim;
    uint32_t firstDimPerCore;
    uint32_t firstDimPerCoreTail;
    uint32_t firstDimPerTime;
    uint32_t lastDimPerTime;
    float eps;
    float aveNum;
    bool isXOut;

    uint32_t gmOffset_;
    uint32_t rowTail_;
    uint32_t rowStep;
    uint32_t rowWork;

    uint64_t repsFp32;
    uint64_t offsetsFp32;
    uint64_t remsFp32;

    bool lastDimPad = false;
    size_t numLastDimAligned;
    size_t numLastDimRoundUp32;
};

#endif  // ADD_LAYER_NORM_QUANT_BASE_CLASS_H_
