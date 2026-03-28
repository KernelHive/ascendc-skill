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
 * \file rms_norm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

namespace optiling {

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

BEGIN_TILING_DATA_DEF(RMSNormTilingData)
TILING_DATA_FIELD_DEF(uint64_t, num_row);
TILING_DATA_FIELD_DEF(uint64_t, num_col);
TILING_DATA_FIELD_DEF(uint64_t, num_col_align);
TILING_DATA_FIELD_DEF(uint64_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(uint32_t, reduce_mask);
TILING_DATA_FIELD_DEF(uint32_t, left_num);
TILING_DATA_FIELD_DEF(uint32_t, last_reduce_mask);
TILING_DATA_FIELD_DEF(uint32_t, last_left_num);
TILING_DATA_FIELD_DEF(uint32_t, rstd_size);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNorm, RMSNormTilingData)

struct Tiling4RmsNormCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
};

class RMSNormTilingInfo {
public:
    uint64_t ubSize{0};
    uint64_t numCol{0};
    uint64_t numRow{0};

    bool isSoc910B{false};
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_
