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
 * \file rms_norm_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace optiling {

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(RmsNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint32_t, data_type);
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_split_dim);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(uint32_t, core_calc_num);
TILING_DATA_FIELD_DEF(uint32_t, core_calc_tail);
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_num);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_loop);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_num);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_tail);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_loop);
TILING_DATA_FIELD_DEF(uint32_t, fixed_output);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormGrad, RmsNormGradTilingData)
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H
