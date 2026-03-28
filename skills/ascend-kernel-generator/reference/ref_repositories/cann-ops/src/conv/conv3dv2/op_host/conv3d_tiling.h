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
 * \file conv3d_tiling.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV3D_TILING_H
#define ASCENDC_TIKCFW_TILING_CONV3D_TILING_H

#include "conv3d_tiling_base.h"

namespace conv3d_tiling {
class Conv3dTiling : public Conv3dTilingBase {
public:
    Conv3dTiling() {};
    explicit Conv3dTiling(const PlatformInfo& platform) : Conv3dTilingBase(platform) {};
    ~Conv3dTiling() override = default;
    int64_t GetTiling(optiling::TConv3DTiling &tiling) override;
protected:
    int64_t Compute() override;
};
} // namespace conv3d_tiling

#endif

