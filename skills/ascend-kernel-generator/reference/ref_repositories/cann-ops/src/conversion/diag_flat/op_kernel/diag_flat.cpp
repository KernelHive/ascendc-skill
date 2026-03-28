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
* \file diag_flat.cpp
* \brief
*/

#include "diag_flat_nd_to_2d.h"
#include "diag_flat_nd_to_2d_with_few.h"
#include "diag_flat_nd_to_2d_b16_more64.h"
#include "diag_flat_nd_to_2d_b16_less.h"
#include "kernel_operator.h"

using namespace DiagFlat;

extern "C" __global__ __aicore__ void diag_flat(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
if (workspace == nullptr) {
        return;
}

GET_TILING_DATA(tilingData, tiling);

// input number less than 64
// dtype is complex32/uint32/int32/float32
if (TILING_KEY_IS(1011)) {
        DiagFlat::DiagFlatNDTo2DWithFew<int32_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
} else if (TILING_KEY_IS(1012)) {
        DiagFlat::DiagFlatNDTo2DWithFew<int64_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
} else if (TILING_KEY_IS(1013)) {
        DiagFlat::DiagFlatNDTo2DWithFew<int16_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
} else if (TILING_KEY_IS(1014)) {
        DiagFlat::DiagFlatNDTo2DWithFew<int8_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
// dtype is complex128, input number is less than 64
} else if (TILING_KEY_IS(102)) {
        DiagFlat::DiagFlatND2To2DB16Less64<int64_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
// input number more than 64
// dtype is complex32/uint32/int32/float32
} else if (TILING_KEY_IS(1031)) {
        DiagFlat::DiagFlatNDTo2D<int32_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
// dtype is complex64/uint64/int64/float64
} else if (TILING_KEY_IS(1032)) {
        DiagFlat::DiagFlatNDTo2D<int64_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
// dtype is int16/uin16/float16
} else if (TILING_KEY_IS(1033)) {
        DiagFlat::DiagFlatNDTo2D<int16_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
// dtype is int8/uin8/float8
} else if (TILING_KEY_IS(1034)) {
        DiagFlat::DiagFlatNDTo2D<int8_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
// input number more than 64, input type is complex128
} else if (TILING_KEY_IS(104)) {
        DiagFlat::DiagFlatND2To2DB16More64<int64_t> op;
        op.Init(input, output, workspace, &tilingData);
        op.Process();
}
}