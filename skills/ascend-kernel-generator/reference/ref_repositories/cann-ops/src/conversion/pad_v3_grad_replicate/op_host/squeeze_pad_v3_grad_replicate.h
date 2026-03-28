/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file squeeze_pad_v3_grad_replicate.h
 */
#ifndef OP_API_INC_LEVEL0_SQUEEZE_PAD_V3_GRAD_REPLICATE_ND_H
#define OP_API_INC_LEVEL0_SQUEEZE_PAD_V3_GRAD_REPLICATE_ND_H

# include "opdev/op_def.h"

namespace l0op {

const aclTensor *SqueezeNd(const aclTensor *x, const aclIntArray* dim, aclOpExecutor *executor);

const aclTensor *SqueezeNd(const aclTensor *x, int64_t dim, aclOpExecutor *executor);

} // l0op

#endif  // OP_API_INC_LEVEL0_SQUEEZE_PAD_V3_GRAD_REPLICATE_ND_H