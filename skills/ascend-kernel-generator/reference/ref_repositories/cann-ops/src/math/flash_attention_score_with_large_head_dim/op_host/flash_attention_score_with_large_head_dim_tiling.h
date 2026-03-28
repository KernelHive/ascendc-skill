/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InputParams)
TILING_DATA_FIELD_DEF(int64_t, bSize);
TILING_DATA_FIELD_DEF(int64_t, n2Size);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, s1Size);
TILING_DATA_FIELD_DEF(int64_t, s2Size);
TILING_DATA_FIELD_DEF(int64_t, dSize);
TILING_DATA_FIELD_DEF(float, scaleValue);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InputParamsOp, InputParams)

BEGIN_TILING_DATA_DEF(MultiCoreParams)
// BN2GS1.o
TILING_DATA_FIELD_DEF(int64_t, totalSize);
// BN2GS1.o / core_num
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MultiCoreParamsOp, MultiCoreParams)

BEGIN_TILING_DATA_DEF(CoreParams)
TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
TILING_DATA_FIELD_DEF(int32_t, s2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, nRatio);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CoreParamsOp, CoreParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreWithLargeHeadDimTilingData)
TILING_DATA_FIELD_DEF_STRUCT(InputParams, inputParams);
TILING_DATA_FIELD_DEF_STRUCT(MultiCoreParams, multiCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(CoreParams, coreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FlashAttentionScoreWithLargeHeadDim, FlashAttentionScoreWithLargeHeadDimTilingData)
}
