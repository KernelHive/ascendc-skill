/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pad_v4_grad_tilingdata.h
 * \brief
 */
#ifndef __PAD_V4_GRAD_TILINGDATA_H__
#define __PAD_V4_GRAD_TILINGDATA_H__

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PadV4GradTilingData)
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
    TILING_DATA_FIELD_DEF(int32_t, hPad1);
    TILING_DATA_FIELD_DEF(int32_t, hPad2);
    TILING_DATA_FIELD_DEF(int32_t, wPad1);
    TILING_DATA_FIELD_DEF(int32_t, wPad2);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, ncPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailNC);
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF(uint32_t, wPadCopyCount);
    TILING_DATA_FIELD_DEF(uint64_t, workspacePerCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PadV4Grad, PadV4GradTilingData)

struct Tiling4PadV4GradCompileInfo {
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
    int32_t hPad1 = 0;
    int32_t hPad2 = 0;
    int32_t wPad1 = 0;
    int32_t wPad2 = 0;
    uint32_t mode = 0;
    uint32_t dtype = 1;  // 1:float32; 2:float16; 3:bfloat16
};
}  // namespace optiling
#endif  // __PAD_V4_GRAD_TILINGDATA_H__