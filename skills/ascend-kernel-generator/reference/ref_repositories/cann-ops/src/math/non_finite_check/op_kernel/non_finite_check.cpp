/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file non_finite_check.cpp
 * \brief
 */
#include "non_finite_check_n_d.h"

using namespace NonFiniteCheck;

extern "C" __global__ __aicore__ void non_finite_check(GM_ADDR tensor_list, GM_ADDR found_flag, GM_ADDR workspace,
                                                       GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(101)) {
        NonFiniteCheckND<half> op;
        op.Init(tensor_list, found_flag, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(201)) {
        NonFiniteCheckND<bfloat16_t> op;
        op.Init(tensor_list, found_flag, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(301)) {
        NonFiniteCheckND<float> op;
        op.Init(tensor_list, found_flag, &tilingData);
        op.Process();
    }
}