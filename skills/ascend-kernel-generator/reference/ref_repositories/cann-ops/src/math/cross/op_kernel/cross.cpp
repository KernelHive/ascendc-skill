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
 * @file cross.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T> struct map {
    using type = float;
};
template<> struct map<int32_t> {
    using type = int32_t;
};

template<typename T> class KernelCross {
public:
    __aicore__ inline KernelCross() {}
    __aicore__ inline int64_t get_index(int64_t i, int64_t j) {
        int64_t index = 0, step = 1, offset = 1;
        for (int64_t k = numshapes - 1; k >= 0; --k) {
            const int64_t idx = i / step % outshape[k];
            step *= outshape[k];
            index += idx % shape[j][k] * offset;
            offset *= shape[j][k];
        }
        return index;
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const int64_t ss[], int64_t numshapes, int64_t dim) {
        Gm_x1.SetGlobalBuffer((__gm__ T*)x1, totalSize[0]);
        Gm_x2.SetGlobalBuffer((__gm__ T*)x2, totalSize[1]);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, maxtotalSize);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 64; ++j) {
                this->shape[i][j] = ss[i * 64 + j];
            }
        }
        this->numshapes = numshapes;
        this->dim = dim;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < numshapes; ++j) {
                if (shape[i][j] > outshape[j]) {
                    outshape[j] = shape[i][j];
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < numshapes; ++j) {
                totalSize[i] *= shape[i][j];
                if (j < dim) {
                    batchSize[i] *= shape[i][j];
                }
                if (j > dim) {
                    stepSize[i] *= shape[i][j];
                }
            }
        }
        for (int j = 0; j < numshapes; ++j) {
            maxtotalSize *= outshape[j];
            if (j < dim) {
                maxbatchSize *= outshape[j];
            }
            if (j > dim) {
                maxstepSize *= outshape[j];
            }
        }
    }
    __aicore__ inline void Process() {
        using F = typename map<T>::type;
        for (int64_t i = 0; i < maxbatchSize; ++i) {
            for (int64_t j = 0; j < maxstepSize; ++j) {
                auto index1 = i * 3 * maxstepSize + 0 * maxstepSize + j;
                auto index2 = i * 3 * maxstepSize + 1 * maxstepSize + j;
                auto index3 = i * 3 * maxstepSize + 2 * maxstepSize + j;
                F a1 = Gm_x1.GetValue(get_index(index1, 0));
                F a2 = Gm_x1.GetValue(get_index(index2, 0));
                F a3 = Gm_x1.GetValue(get_index(index3, 0));
                F b1 = Gm_x2.GetValue(get_index(index1, 1));
                F b2 = Gm_x2.GetValue(get_index(index2, 1));
                F b3 = Gm_x2.GetValue(get_index(index3, 1));
                auto result1 = a2 * b3 - a3 * b2;
                auto result2 = a3 * b1 - a1 * b3;
                auto result3 = a1 * b2 - a2 * b1;
                Gm_y.SetValue(index1, (T)result1);
                Gm_y.SetValue(index2, (T)result2);
                Gm_y.SetValue(index3, (T)result3);
            }
        }
    }
private:
    AscendC::GlobalTensor<T> Gm_x1, Gm_x2, Gm_y;
    int64_t shape[2][64];
    int64_t numshapes;
    int64_t dim;
    int64_t outshape[64] = {};
    int64_t maxtotalSize = 1, maxbatchSize = 1, maxstepSize = 1;
    int64_t totalSize[3] = { 1, 1, 1 }, batchSize[3] = { 1, 1, 1 }, stepSize[3] = { 1, 1, 1 };
};
extern "C" __global__ __aicore__ void cross(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCross<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.shape, tiling_data.numshapes, tiling_data.dim);
    op.Process();
}