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
 * \file foreach_sigmoid.cpp
 * \brief
 */

#include "foreach/foreach_proto_utils.h"
#include "register/op_def_registry.h"
 
namespace ops {
    FOREACH_OPDEF(HOST_CONFIG, UNARY, Sigmoid, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16)
}  // namespace ops
 