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
 * \file foreach_pow_scalar_and_tensor.cpp
 * \brief
 */

#include "foreach/foreach_proto_utils.h"
#include "register/op_def_registry.h"
 
namespace ops {
    FOREACH_OPDEF_BEGIN(PowScalarAndTensor)
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16)
    FOREACH_SCALAR_DTYPE_PREPARE
    FOREACH_OPDEF_PARAM_SCALAR(Input, scalar)
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x)
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)
    FOREACH_OPDEF_END_HOST_CONFIG(PowScalarAndTensor)
    
    OP_ADD(ForeachPowScalarAndTensor);
    }  // namespace ops