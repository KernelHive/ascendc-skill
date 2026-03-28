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
 * @file reflection_pad3d_grad_tiling.h
 */
#ifndef REFLECTION_PAD3D_GRAD_TILINGDATA_H
#define REFLECTION_PAD3D_GRAD_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ReflectionPad3dGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch); 
    TILING_DATA_FIELD_DEF(uint32_t, channel);
    TILING_DATA_FIELD_DEF(uint32_t, depth);
    TILING_DATA_FIELD_DEF(uint32_t, height);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, alignDepth);
    TILING_DATA_FIELD_DEF(uint32_t, alignHeight);
    TILING_DATA_FIELD_DEF(uint32_t, alignWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outWidth);
    TILING_DATA_FIELD_DEF(uint32_t, alignOutDepth);
    TILING_DATA_FIELD_DEF(uint32_t, alignOutHeight);
    TILING_DATA_FIELD_DEF(uint32_t, alignOutWidth);
    TILING_DATA_FIELD_DEF(int32_t, dPad1);
    TILING_DATA_FIELD_DEF(int32_t, dPad2);
    TILING_DATA_FIELD_DEF(int32_t, hPad1);
    TILING_DATA_FIELD_DEF(int32_t, hPad2);
    TILING_DATA_FIELD_DEF(int32_t, wPad1);
    TILING_DATA_FIELD_DEF(int32_t, wPad2);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, ncPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailNC);
    TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ReflectionPad3dGrad, ReflectionPad3dGradTilingData)

struct Tiling4PadV3GradV2CompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint32_t sysWorkspaceSize;
};

struct InputParamsInfo {
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t depth = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignDepth = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outDepth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutDepth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    int32_t dPad1 = 0;
    int32_t dPad2 = 0;
    int32_t hPad1 = 0;
    int32_t hPad2 = 0;
    int32_t wPad1 = 0;
    int32_t wPad2 = 0;
    uint32_t mode = 0;
    uint64_t tilingKey = 0;
};
}  // namespace optiling
#endif  // REFLECTION_PAD3D_GRAD_TILINGDATA_H
