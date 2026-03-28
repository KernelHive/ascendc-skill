/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define K_MAX_SHAPE_DIM 0
#include <type_traits>
#include "kernel_operator.h"

#define TI this->ti
// Alias
using namespace AscendC;
using std::is_same_v;

template<int tilingKey> struct ReshapeTilingInfo { };

template<> struct ReshapeTilingInfo<1> { 
    uint32_t tileLength, tileNumber, reminder;
    // int32_t axis, numAxes;
};
template<> struct ReshapeTilingInfo<2> { 
    uint32_t tileLength, tileNumber, reminder;
    // int32_t axis, numAxes;
};

template<class DATA, class SHAPE, int tilingKey> class MyReshape {};

template<class DATA, class SHAPE> class MyReshape<DATA, SHAPE, 1> {
    GlobalTensor<DATA> gm_x, gm_y;
    GlobalTensor<SHAPE> gm_shape;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> q_data;
    ReshapeTilingInfo<1> ti;
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        LocalTensor<DATA> data = q_data.AllocTensor<DATA>();
        DataCopy(data, gm_x[offset], length);
        q_data.EnQue<DATA>(data);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        LocalTensor<DATA> data = q_data.DeQue<DATA>();
        DataCopy(gm_y[offset], data, length);
        q_data.FreeTensor(data);
    }
public:
    __aicore__ inline MyReshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, const ReshapeTilingInfo<1> &ti, TPipe *p):ti(ti) {
        gm_x.SetGlobalBuffer((__gm__ DATA *)x);
        gm_shape.SetGlobalBuffer((__gm__ SHAPE *)shape);
        gm_y.SetGlobalBuffer((__gm__ DATA *)y);
        p->InitBuffer(q_data, 2, this->ti.tileLength * sizeof(DATA));
    }
    __aicore__ inline void exec() {
        for(uint32_t i = 0; i < TI.tileNumber; i++) {
            CopyIn(i * TI.tileLength, TI.tileLength);
            CopyOut(i * TI.tileLength, TI.tileLength);
        }
        CopyIn(TI.tileNumber * TI.tileLength, TI.reminder);
        CopyOut(TI.tileNumber * TI.tileLength, TI.reminder);
    }
};
extern "C" __global__ __aicore__ void reshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        ReshapeTilingInfo<1> ti{
            .tileLength = tiling_data.tileLength,
            .tileNumber = tiling_data.tileNumber,
            .reminder = tiling_data.reminder
        };
        TPipe p;
        MyReshape<DTYPE_Y, DTYPE_SHAPE, 1>(x, shape, y, ti, &p).exec();
    }
}