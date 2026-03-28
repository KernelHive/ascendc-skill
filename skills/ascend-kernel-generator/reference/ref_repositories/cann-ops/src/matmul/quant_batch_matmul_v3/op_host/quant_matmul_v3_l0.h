/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_QUANT_BATCH_MATMUL_V3_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_QUANT_BATCH_MATMUL_V3_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor* QuantBatchMatmulV3(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                                    const aclTensor* offset, const aclTensor* bias, const aclTensor* pertokenScale,
                                    int64_t dtype, bool transposeX1, bool transposeX2, aclOpExecutor* executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_QUANT_BATCH_MATMUL_V3_H
