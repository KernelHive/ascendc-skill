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
 * \file conv2dbp_adapt_to_conv3dbp.h
 * \brief
 */
#ifndef CONV2DBP_ADAPT_TO_CONV3DBP
#define CONV2DBP_ADAPT_TO_CONV3DBP
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "register/op_impl_registry.h"
#include "base/context_maker/kernel_run_context_maker.h"
namespace optiling {
ge::graphStatus AdaptTilingToConv3DBackprop(gert::TilingContext *context, std::string opType);
} // namespace optiling
#endif // CONV2DBP_ADAPT_TO_CONV3DBP