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
 * \file cross_entropy_loss_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_CROSS_ENTROPY_LOSS_GRAD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_CROSS_ENTROPY_LOSS_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CrossEntropyLossGradTilingData)
  TILING_DATA_FIELD_DEF(int64_t, reduction);
  TILING_DATA_FIELD_DEF(int64_t, ignoreIndex);
  TILING_DATA_FIELD_DEF(float, labelSmoothing);
  TILING_DATA_FIELD_DEF(int64_t, rowVal);           // N
  TILING_DATA_FIELD_DEF(int64_t, colVal);           // C
  TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);     // 前面多算一个的核有几个
  TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);      // 后面的核有几个
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);      // 用了多少核
  TILING_DATA_FIELD_DEF(int64_t, frontRowNum);      // 前面的核要算几行
  TILING_DATA_FIELD_DEF(int64_t, tailRowNum);       // 后面的核要算几行
  TILING_DATA_FIELD_DEF(int64_t, alignColLoopNum);  // 满ub可以处理的最大对齐数据量
  TILING_DATA_FIELD_DEF(int64_t, colLoop);          // 一行里满ub要循环多少次
  TILING_DATA_FIELD_DEF(int64_t, colLoopNumTail);   // 一行中最后一次循环，不满ub要处理的数据量
  TILING_DATA_FIELD_DEF(int64_t, targetSize);
  TILING_DATA_FIELD_DEF(int64_t, targetCastSize);
  TILING_DATA_FIELD_DEF(int64_t, gradLossSize);
  TILING_DATA_FIELD_DEF(int64_t, gradLossFp32Size);
  TILING_DATA_FIELD_DEF(int64_t, ignoreSize);
  TILING_DATA_FIELD_DEF(int64_t, maskSize);
  TILING_DATA_FIELD_DEF(int64_t, targetWeightSize);
  TILING_DATA_FIELD_DEF(int64_t, tBuf2Size);        // tmpbuf2
  TILING_DATA_FIELD_DEF(int64_t, tBuf3Size);        // tmpbuf3
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossEntropyLossGrad, CrossEntropyLossGradTilingData)
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_CROSS_ENTROPY_LOSS_GRAD_TILING_H
