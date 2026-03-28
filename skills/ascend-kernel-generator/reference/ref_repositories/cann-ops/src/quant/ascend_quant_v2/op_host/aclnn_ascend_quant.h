/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ASCEND_QUANT_H_
#define OP_API_INC_LEVEL2_ACLNN_ASCEND_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAscendQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 待进行AscendQuant计算的入参。npu device侧的aclTensor，
 * 数据类型支持float16, bfloat16, float32, 数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [in] scale: npu device侧的aclTensor, 数据类型支持float, bf16, float16
 * @param [in] offset: npu device侧的aclTensor，数据类型支持float, bf16, float16
 * @param [in] sqrtMode:  host侧的aclScalar，数据类型bool
 * @param [in] roundMode:  host侧的aclScalar，数据类型string
 * @param [in] dstType:  host侧的aclScalar, 数据类型int
 * @param [in] y: AscendQuant计算的出参。npu device侧的aclTensor，
 * 数据类型支持int8, 数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAscendQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* scale,
                                                       const aclTensor* offset, bool sqrtMode, const char* roundMode,
                                                       int32_t dstType, const aclTensor* y, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnAscendQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAscendAntiQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAscendQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ASCEND_QUANT_H_