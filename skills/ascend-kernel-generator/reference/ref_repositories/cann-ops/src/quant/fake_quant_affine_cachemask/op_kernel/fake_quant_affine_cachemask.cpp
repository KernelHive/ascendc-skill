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
 * \file fake_quant_affine_cachemask.cpp
 * \brief
 */

#include "fake_quant_affine_cachemask_fp32.h"
#include "fake_quant_affine_cachemask_fp16.h"

using namespace FakeQuantAffineCachemaskN;

extern "C" __global__ __aicore__ void fake_quant_affine_cachemask(GM_ADDR x, GM_ADDR scale,
    GM_ADDR zero_point, GM_ADDR y, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        FakeQuantAffineCachemaskN::FakeQuantAffineCachemaskFp32<float> op;
        op.Init(x, scale, zero_point, y, mask, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FakeQuantAffineCachemaskN::FakeQuantAffineCachemaskFp16<half> op;
        op.Init(x, scale, zero_point, y, mask, &tilingData);
        op.Process();
    }
}