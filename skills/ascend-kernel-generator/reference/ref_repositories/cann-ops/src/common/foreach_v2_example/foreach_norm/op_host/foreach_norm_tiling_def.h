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
 * \file foreach_tiling_norm_tiling_def.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NORM_TILING_DEF_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NORM_TILING_DEF_H_

#include "register/tilingdata_base.h"
#include "foreach_reduce_tiling_def.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(ForeachNorm, ForeachReduceTilingData)
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_NORM_TILING_DEF_H_