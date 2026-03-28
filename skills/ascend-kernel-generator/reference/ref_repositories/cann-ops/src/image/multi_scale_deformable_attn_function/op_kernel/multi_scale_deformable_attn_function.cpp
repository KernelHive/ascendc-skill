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
 * \file multi_scale_deformable_attn_function.cpp
 * \brief
 */

#include "ms_deform_attn_generic.h"
#include "ms_deform_attn_high_perf.h"

extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function(GM_ADDR value, GM_ADDR valueSpatialShapes,
    GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1002)) {
        KernelMultiScaleDeformableAttnOpt<2, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1004)) {
        KernelMultiScaleDeformableAttnOpt<4, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1008)) {
        KernelMultiScaleDeformableAttnOpt<8, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2002)) {
        KernelMultiScaleDeformableAttnOpt<2, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2004)) {
        KernelMultiScaleDeformableAttnOpt<4, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2008)) {
        KernelMultiScaleDeformableAttnOpt<8, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelMultiScaleDeformableAttn op;
        op.Init(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    }
}
