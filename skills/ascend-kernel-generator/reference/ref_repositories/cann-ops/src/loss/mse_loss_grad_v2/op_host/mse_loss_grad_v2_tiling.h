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
 * \file mse_loss_grad_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MSE_LOSS_GRAD_V2_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MSE_LOSS_GRAD_V2_TILING_H

#include <iostream>
#include <cstring>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossGradTilingData)
    TILING_DATA_FIELD_DEF(float, cof);
    TILING_DATA_FIELD_DEF(uint64_t, totalLength);//总计算数据量
    TILING_DATA_FIELD_DEF(uint64_t, tileNum);//每个核上总计算数据分块个数
    TILING_DATA_FIELD_DEF(uint64_t, padLength);//尾块的个数
    TILING_DATA_FIELD_DEF(uint64_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, usedDb);   //  8 Bytes align with cof
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossGradV2, MseLossGradTilingData)

void GetTilingKey(const uint32_t dtypeKey, uint32_t &tilingKey);
}

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_MSE_LOSS_GRAD_V2_TILING_H