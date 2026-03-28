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
 * \file conv3d_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "conv3d_tiling_algorithm.h"
#include "conv3d_tiling_algorithm_pointwise.h"
#include "conv3d_tiling_algorithm_hw_mode.h"
#include "conv3d_tiling_base.h"
#include "conv3d_tiling.h"

namespace conv3d_tiling {
int64_t Conv3dTiling::GetTiling(optiling::TConv3DTiling &tiling)
{
    int64_t ret = Compute();
    if (ret == -1) {
        TILING_ERROR_LOG("can not gen conv3d api tiling");
        return -1;
    }

    SetFinalTiling(tiling);
    PrintTilingData();
    return ret;
}

int64_t Conv3dTiling::Compute()
{
    bool isPointWise = (this->descInfo.fMapType.format == ConvFormat::NCDHW);
    if (!isPointWise && !CheckInputParam()) {
        return -1;
    }

    if (isPointWise && !CheckInputParamPointWise()) {
        return -1;
    }
    // get cube info
    GetCubeInfo();
    // cal output and check valid
    if (!ShapeInitCalc()) {
        return -1;
    }
    if (!CheckParamsOverflow()) {
        return -1;
    }
    if (isPointWise) {
        Conv3dTilingAlgorithmPointWise tilingAlgo(this);
        int64_t ret = tilingAlgo.Process();
        return ret;
    } else if (outputOrder_ == M_Mode) {
        Conv3dTilingAlgorithm tilingAlgo(this);
        int64_t ret = tilingAlgo.Process();
        return ret;
    } else {
        Conv3dTilingAlgorithmHwMode tilingAlgo(this);
        int64_t ret = tilingAlgo.Process();
        return ret;
    }
}
} // namespace conv3d_tiling