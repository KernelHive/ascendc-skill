/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_CIRCULAR_PAD2D_BACKWARD_H_
#define OP_API_INC_CIRCULAR_PAD2D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCircularPad2dBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：循环填充的反向传播。
 * @param [in] gradOutput: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, BFLOAT16, 数据格式支持ND，
 * 维度支持三维或四维且与self和gradInput一致，shape需要与circular_pad2d正向传播的output一致。
 * @param [in] self: npu device侧的aclTensor,
 * 数据类型与gradOutput一致，数据格式支持ND，维度支持三维或四维且与gradOutput和 gradInput一致，shape与gradInput一致。
 * @param [in] padding: npu device侧的aclIntArray数组, 数据类型为INT64，长度为4，数值依次代表左右上下需要填充的值。
 * 前两个数值需小于self最后一维度的数值，后两个数值需小于self倒数第二维度的数值。
 * @param [in] gradInput: npu device侧的aclTensor, 数据类型与gradOutput一致，shape与self一致，数据格式支持ND，
 * 维度支持三维或四维且与gradOutput和self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCircularPad2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                                   const aclIntArray* padding, aclTensor* gradInput,
                                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnCircularPad2dBackward的第二段接口，用于执行计算
 *
 * 算子功能：循环填充的反向传播。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnCircularPad2dBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCircularPad2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CIRCULAR_PAD2D_BACKWARD_H_