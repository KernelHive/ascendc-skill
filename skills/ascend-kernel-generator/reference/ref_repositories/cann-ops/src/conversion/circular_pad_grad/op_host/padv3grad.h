/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_PADV3GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_PADV3GRAD_OP_H_

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor *PadV3Grad(const aclTensor *gradOutput, const aclTensor *paddings, const std::string& mode,
                           const bool paddingsContiguous, const bool padFlag, aclOpExecutor *executor);
}
#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_PADV3GRAD_OP_H_
