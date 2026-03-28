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
 * \file diag_v2.cpp
 * \brief
 */
#include "diag_v2_b8.h"
#include "diag_v2_b16.h"
#include "diag_v2_b32.h"
#include "diag_v2_b64.h"
#include "diag_v2_b128.h"
#include "diag_v2_t4.h"

using namespace DiagV2;

extern "C" __global__ __aicore__ void diag_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(2101)) {
        DiagV2::DiagV2B8<int8_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2102)) {
        DiagV2::DiagV2B16<int16_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2103)) {
        DiagV2::DiagV2B32<int32_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2104)) {
        DiagV2::DiagV2B64<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2105)) {
        DiagV2::DiagV2B128<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2401)) {
        DiagV2::DiagV2T4<int8_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2402)) {
        DiagV2::DiagV2T4<int16_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2403)) {
        DiagV2::DiagV2T4<int32_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2404)) {
        DiagV2::DiagV2T4<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } 
}
