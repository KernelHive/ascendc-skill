/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aoe/op_tuning_tiling/conv2d_dw_tuning_tiling.h"

namespace tuningtiling {
DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv2DBackpropFilter, Conv2DDwInputArgs, a_shape_n, a_shape_h, a_shape_w, b_shape_h,
                              b_shape_w, c_shape_n, c_shape_c, c_shape_h, c_shape_w, groups, stride_h, stride_w,
                              dilation_h, dilation_w, pad_u, pad_d, pad_l, pad_r, a_dtype, b_dtype, c_dtype,
                              binary_mode, hf32_flag, reserved_params1, reserved_params2, reserved_params3,
                              reserved_params4, reserved_params5);
REGISTER_TUNING_TILING_CLASS(Conv2DBackpropFilter, Conv2DDwTunnerTiling);
}  // namespace tuningtiling