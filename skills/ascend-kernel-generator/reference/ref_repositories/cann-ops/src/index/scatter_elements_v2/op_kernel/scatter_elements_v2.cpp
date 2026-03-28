/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file scatter_elements_v2.cpp
 * \brief
 */
#include "scatter_elements_v2.h"

#define CALL_OP_IMPL(...)                                                \
  do {                                                                   \
    KernelScatterElementsV2<__VA_ARGS__> op;                             \
    op.Init(tilingDevice, &pipe, var, indices, updates);                 \
    if (tilingDevice->modeFlag == 1) {                                   \
        op.ProcessSmall();                                               \
    } else {                                                             \
        op.ProcessScatter();                                             \
    }                                                                    \
  } while (0)

extern "C" __global__ __aicore__ void scatter_elements_v2(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterElementsV2TilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(111)) {
        CALL_OP_IMPL(float, int, 1);
    } else if (TILING_KEY_IS(112)) {
        CALL_OP_IMPL(float, int, 2);
    } else if (TILING_KEY_IS(121)) {
        CALL_OP_IMPL(float, long, 1);
    } else if (TILING_KEY_IS(122)) {
        CALL_OP_IMPL(float, long, 2);
    } else if (TILING_KEY_IS(211)) {
        CALL_OP_IMPL(half, int, 1);
    } else if (TILING_KEY_IS(212)) {
        CALL_OP_IMPL(half, int, 2);
    } else if (TILING_KEY_IS(221)) {
        CALL_OP_IMPL(half, long, 1);
    } else if (TILING_KEY_IS(222)) {
        CALL_OP_IMPL(half, long, 2);
    } else if (TILING_KEY_IS(311)) {
        CALL_OP_IMPL(int, int, 1);
    } else if (TILING_KEY_IS(312)) {
        CALL_OP_IMPL(int, int, 2);
    } else if (TILING_KEY_IS(321)) {
        CALL_OP_IMPL(int, long, 1);
    } else if (TILING_KEY_IS(322)) {
        CALL_OP_IMPL(int, long, 2);
    } else if (TILING_KEY_IS(411)) {
        CALL_OP_IMPL(uint8_t, int, 1);
    } else if (TILING_KEY_IS(412)) {
        CALL_OP_IMPL(uint8_t, int, 2);
    } else if (TILING_KEY_IS(421)) {
        CALL_OP_IMPL(uint8_t, long, 1);
    } else if (TILING_KEY_IS(422)) {
        CALL_OP_IMPL(uint8_t, long, 2);
    } else if (TILING_KEY_IS(511)) {
        CALL_OP_IMPL(int8_t, int, 1);
    } else if (TILING_KEY_IS(512)) {
        CALL_OP_IMPL(int8_t, int, 2);
    } else if (TILING_KEY_IS(521)) {
        CALL_OP_IMPL(int8_t, long, 1);
    } else if (TILING_KEY_IS(522)) {
        CALL_OP_IMPL(int8_t, long, 2);
    } else if (TILING_KEY_IS(611)) {
        CALL_OP_IMPL(bfloat16_t, int, 1);
    } else if (TILING_KEY_IS(612)) {
        CALL_OP_IMPL(bfloat16_t, int, 2);
    } else if (TILING_KEY_IS(621)) {
        CALL_OP_IMPL(bfloat16_t, long, 1);
    } else if (TILING_KEY_IS(622)) {
        CALL_OP_IMPL(bfloat16_t, long, 2);
    }
}
