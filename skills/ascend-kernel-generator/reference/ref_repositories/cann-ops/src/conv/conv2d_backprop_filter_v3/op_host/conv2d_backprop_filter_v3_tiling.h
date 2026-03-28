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
 * \file conv2d_backprop_filter_v3_tiling.h
 * \brief
 */
#ifndef CONV2D_BACKPROP_FILTER_V3_TILING_H
#define CONV2D_BACKPROP_FILTER_V3_TILING_H
#include "conv3d_backprop_filter_v2_tiling.h"
#include "conv3d_dw_v2_basic_block_tiling.h"
#include "conv2dbp_adapt_to_conv3dbp.h"
namespace optiling {  

REGISTER_TILING_DATA_CLASS(Conv2DBackpropFilterV3, Conv3DBackpropFilterV2TilingData)

class Conv2DBackpropFilterV3Tiling : public Conv3DBackpropFilterV2Tiling {
public:
    explicit Conv2DBackpropFilterV3Tiling(gert::TilingContext *context) : Conv3DBackpropFilterV2Tiling(context)
    {
        Reset();
    }
    ~Conv2DBackpropFilterV3Tiling() override = default;
protected:
    ge::graphStatus DoLibApiTiling() override;
};

class Conv2DDWV3BasicBlockTiling : public Conv3DDWV2BasicBlockTiling {
public:
    explicit Conv2DDWV3BasicBlockTiling(gert::TilingContext *context) : Conv3DDWV2BasicBlockTiling(context)
    {
        Reset();
    }
    ~Conv2DDWV3BasicBlockTiling() override = default;
protected:
    bool IsCapable() override;
};
}
#endif  // CONV2D_BACKPROP_FILTER_V3_TILING_H