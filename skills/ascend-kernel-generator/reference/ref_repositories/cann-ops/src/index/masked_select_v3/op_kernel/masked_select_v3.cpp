/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file masked_select_v3.cpp
 * \brief
 */
#include "masked_select_v3.h"

extern "C" __global__ __aicore__ void masked_select_v3(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeout, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); // 获取用户workspace指针。

    if (TILING_KEY_IS(8)) {
        AscendC::KernelMaskedSelectV3<uint64_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    } else if (TILING_KEY_IS(4)) {
        AscendC::KernelMaskedSelectV3<uint32_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    } else if (TILING_KEY_IS(2)) {
        AscendC::KernelMaskedSelectV3<uint16_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    } else if (TILING_KEY_IS(1)) {
        AscendC::KernelMaskedSelectV3<uint8_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    }
}