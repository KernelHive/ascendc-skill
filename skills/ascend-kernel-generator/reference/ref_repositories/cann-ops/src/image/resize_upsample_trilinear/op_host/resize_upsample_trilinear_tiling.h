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
 * \file resize_upsample_trilinear_tiling.h
 * \brief
 */
#ifndef RESIZE_UPSAMPLE_TRILINEAR_TILING_DEF_H
#define RESIZE_UPSAMPLE_TRILINEAR_TILING_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint16_t MAX_CORE_COUNT = 48;
BEGIN_TILING_DATA_DEF(UpsampleTrilinearTilingData)
TILING_DATA_FIELD_DEF(float, scale_w);
TILING_DATA_FIELD_DEF(float, scale_h);
TILING_DATA_FIELD_DEF(float, scale_d);
TILING_DATA_FIELD_DEF(uint16_t, total_core_num);
TILING_DATA_FIELD_DEF(uint32_t, real_core_num);
TILING_DATA_FIELD_DEF(uint64_t, ratio_metrix_size);
TILING_DATA_FIELD_DEF(int64_t, output_w);
TILING_DATA_FIELD_DEF(int64_t, output_h);
TILING_DATA_FIELD_DEF(int64_t, output_d);
TILING_DATA_FIELD_DEF(int64_t, input_w);
TILING_DATA_FIELD_DEF(int64_t, input_h);
TILING_DATA_FIELD_DEF(int64_t, input_d);
TILING_DATA_FIELD_DEF(int64_t, batches);
TILING_DATA_FIELD_DEF(uint32_t, align_corners);  // 1 true 0 false

TILING_DATA_FIELD_DEF(int64_t, each_core_slide_num);
TILING_DATA_FIELD_DEF(int64_t, remainder);
TILING_DATA_FIELD_DEF(int64_t, tail_start_slide_num);
TILING_DATA_FIELD_DEF(int64_t, slide_size);
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, tensor_size);

TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_start_inx_w_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_end_inx_w_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_slide_start_inx_w_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_slide_end_inx_w_list);

TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_start_inx_h_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_end_inx_h_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_batch_start_inx_h_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_batch_end_inx_h_list);

TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_start_inx_d_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_end_inx_d_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_batch_start_inx_d_list);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_COUNT, tail_group_batch_end_inx_d_list);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmul_tiling_w);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmul_tiling_h);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmul_tiling_d);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeUpsampleTrilinear, UpsampleTrilinearTilingData)
}  // namespace optiling

#endif  // RESIZE_UPSAMPLE_TRILINEAR_TILING_DEF_H
