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
 * \file conv2d_backprop_input_v2_tiling.h
 * \brief
 */

#ifndef CONV2D_BACKPROP_INPUT_V2_TILING_H
#define CONV2D_BACKPROP_INPUT_V2_TILING_H
#include "conv3d_backprop_input_v2_tiling.h"
#include "cube/include/cube_tiling_param.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(Conv2DBackpropInputV2, Conv3DBackpropInputV2TilingData)

class Conv2DBackpropInputV2Tiling : public Conv3DBackpropInputV2Tiling {
public:
    explicit Conv2DBackpropInputV2Tiling(gert::TilingContext *context) : Conv3DBackpropInputV2Tiling(context)
    {
        Reset();
        opType_ = cachetiling::kConv3DBackpropInput;
    }
    ~Conv2DBackpropInputV2Tiling() override = default;
};

ge::graphStatus AdaptTilingToConv3DBp(gert::TilingContext *context, std::string opType);
}
#endif  // CONV2D_BACKPROP_INPUT_V2_TILING_H