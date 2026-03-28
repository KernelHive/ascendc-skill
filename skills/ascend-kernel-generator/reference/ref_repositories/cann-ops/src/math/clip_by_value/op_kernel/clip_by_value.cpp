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
 * @file clip_by_value.cpp
 */
#include <type_traits>
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

template <typename TYPE_X, typename TYPE_CLIP_VALUE_MIN, typename TYPE_CLIP_VALUE_MAX, typename TYPE_Y>
class KernelClipByValue 
{
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR clip_value_min,
                                GM_ADDR clip_value_max, GM_ADDR y,
                                uint32_t totalLength, uint32_t ALIGN_NUM,
                                uint32_t block_size, uint32_t core_size,
                                uint32_t core_remain) 
    {
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        Gm_clip_value_min.SetGlobalBuffer((__gm__ TYPE_CLIP_VALUE_MIN*)clip_value_min, 1);
        Gm_clip_value_max.SetGlobalBuffer((__gm__ TYPE_CLIP_VALUE_MAX*)clip_value_max, 1);
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        this->clip_value_min = Gm_clip_value_min.GetValue(0);
        this->clip_value_max = Gm_clip_value_max.GetValue(0);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
        DataCopy(x, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(x);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        Mins(x, x, this->clip_value_max, length);
        Maxs(y, x, this->clip_value_min, length);
        Q_x.FreeTensor(x);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_CLIP_VALUE_MIN> Gm_clip_value_min;
    GlobalTensor<TYPE_CLIP_VALUE_MAX> Gm_clip_value_max;
    GlobalTensor<TYPE_Y> Gm_y;
    TYPE_CLIP_VALUE_MIN clip_value_min;
    TYPE_CLIP_VALUE_MAX clip_value_max;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    KernelClipByValue<DTYPE_X, DTYPE_CLIP_VALUE_MIN, DTYPE_CLIP_VALUE_MAX, DTYPE_Y> op;
    op.Init(x, clip_value_min, clip_value_max, y, tiling_data.totalLength,
            tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}