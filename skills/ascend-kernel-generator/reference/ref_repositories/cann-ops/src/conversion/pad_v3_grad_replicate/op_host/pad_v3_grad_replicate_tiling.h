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
 * @file pad_v3_grad_replicate_tiling.h
 */
#ifndef __PAD_V3_GRAD_REPLICATE_TILINGDATA_H__
#define __PAD_V3_GRAD_REPLICATE_TILINGDATA_H__

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PadV3GradReplicateTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channel);
    TILING_DATA_FIELD_DEF(uint32_t, height);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, alignHeight);
    TILING_DATA_FIELD_DEF(uint32_t, alignWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outWidth);
    TILING_DATA_FIELD_DEF(uint32_t, alignOutHeight);
    TILING_DATA_FIELD_DEF(uint32_t, alignOutWidth);
    TILING_DATA_FIELD_DEF(int32_t, padTop);
    TILING_DATA_FIELD_DEF(int32_t, padBottom);
    TILING_DATA_FIELD_DEF(int32_t, padLeft);
    TILING_DATA_FIELD_DEF(int32_t, padRight);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, ncPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailNC);
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF(uint32_t, wCalCount);
    TILING_DATA_FIELD_DEF(uint64_t, workspacePerCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PadV3GradReplicate, PadV3GradReplicateTilingData)

struct Tiling4PadV3GradReplicateCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint32_t sysWorkspaceSize;
};

struct InputParamsInfo {
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    int32_t padTop = 0;
    int32_t padBottom = 0;
    int32_t padLeft = 0;
    int32_t padRight = 0;
    uint32_t mode = 1;
    uint32_t dtype = 1;  // 1: float32; 2: float16; 3: bfloat16
};
}  // namespace optiling
#endif  // __PAD_V3_GRAD_REPLICATE_TILINGDATA_H__