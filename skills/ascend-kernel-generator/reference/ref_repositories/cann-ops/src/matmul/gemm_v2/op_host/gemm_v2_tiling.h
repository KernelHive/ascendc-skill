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
 * \file gemm_v2_tiling.h
 * \brief
 */
#ifndef __OP_HOST_GEMM_V2_TILING_H__
#define __OP_HOST_GEMM_V2_TILING_H__
#include "../../mat_mul_v3/op_host/mat_mul_v3_tiling.h"
#include "../../mat_mul_v3/op_host/mat_mul_v3_compile_info.h"
#include "../../mat_mul_v3/op_host/mat_mul_v3_base_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(GemmV2, MatmulTilingData)
}
#endif // __OP_HOST_GEMM_V2_TILING_H__
