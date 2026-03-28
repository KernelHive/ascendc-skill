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
 * \file deep_norm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeepNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_core);
TILING_DATA_FIELD_DEF(uint32_t, num_last_dim);
TILING_DATA_FIELD_DEF(uint32_t, num_first_dim);
TILING_DATA_FIELD_DEF(uint32_t, nl_firstdim_per_core);
TILING_DATA_FIELD_DEF(uint32_t, l_firstdim_per_core);
TILING_DATA_FIELD_DEF(uint32_t, first_dim_per_times);
TILING_DATA_FIELD_DEF(uint32_t, updated_last_dim);
TILING_DATA_FIELD_DEF(uint32_t, updated_last_times);
TILING_DATA_FIELD_DEF(uint32_t, eps_str);
TILING_DATA_FIELD_DEF(uint32_t, ave_str);
TILING_DATA_FIELD_DEF(uint32_t, alpha_str);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeepNorm, DeepNormTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_H
