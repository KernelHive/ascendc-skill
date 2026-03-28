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
 * \file multi_scale_deformable_attention_grad_l0.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_H_
#define OP_API_INC_LEVEL0_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::tuple<aclTensor*, aclTensor*, aclTensor*> MultiScaleDeformableAttentionGrad(const aclTensor *value,
                                                                                       const aclTensor *spatialShape,
                                                                                       const aclTensor *levelStartIndex,
                                                                                       const aclTensor *location,
                                                                                       const aclTensor *attnWeight,
                                                                                       const aclTensor *gradOutput,
                                                                                       aclOpExecutor *executor);
} // l0op

#endif // OP_API_INC_LEVEL0_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_H_
