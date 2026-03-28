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
 * \file ge_glu_grad_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GE_GLU_GRAD_V2_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GE_GLU_GRAD_V2_TILING_H_
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GeGluGradV2TilingData)
    TILING_DATA_FIELD_DEF(int32_t, approximate);
    TILING_DATA_FIELD_DEF(int32_t, activateLeft);
    TILING_DATA_FIELD_DEF(int64_t, maxProcCount);
    TILING_DATA_FIELD_DEF(int64_t, valueN);
    TILING_DATA_FIELD_DEF(int64_t, valueM);
    TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, loopNumPerCore);
    TILING_DATA_FIELD_DEF(int64_t, tailCoreIndex);
    TILING_DATA_FIELD_DEF(int64_t, tailUbLoopNum);
    TILING_DATA_FIELD_DEF(int64_t, groupNum);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GeGluGradV2, GeGluGradV2TilingData)

struct GeGluGradV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
};

enum class GeGluGradV2TilingKey : uint64_t {
    TILING_KEY_TANH_101 = 101,
    TILING_KEY_TANH_102,
    TILING_KEY_TANH_103,
    TILING_KEY_TANH_201 = 201,
    TILING_KEY_TANH_202,
    TILING_KEY_TANH_203,
    TILING_KEY_TANH_301 = 301,
    TILING_KEY_TANH_302,
    TILING_KEY_TANH_303,
    TILING_KEY_ERF_701 = 701,
    TILING_KEY_ERF_702,
    TILING_KEY_ERF_703,
    TILING_KEY_ERF_801 = 801,
    TILING_KEY_ERF_802,
    TILING_KEY_ERF_803,
    TILING_KEY_ERF_901 = 901,
    TILING_KEY_ERF_902,
    TILING_KEY_ERF_903
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GE_GLU_GRAD_V2_TILING_H_
