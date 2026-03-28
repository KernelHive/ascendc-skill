/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_UNAMPLE_BICUBIC2D_AA_H_
#define OP_API_INC_UNAMPLE_BICUBIC2D_AA_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleBicubic2dAA的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 
 * 算子功能：upsample_bicubic2d的抗锯齿计算。
 *
 * @param [in] x: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持NCHW。
 * @param [in] outputSize: 输入IntArray，size大小为2。表示输输出out在H和W维度上的空间大小。
 * @param [in] alignCorners: bool常量，表示是否使用角对齐计算。
 * @param [in] scalesH: double常量，表示输出out的height维度乘数。
 * @param [in] scalesW: double常量，表示输出out的width维度乘数。
 * @param [out] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，且数据类型与x的数据类型一致。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持NCHW。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleBicubic2dAAGetWorkspaceSize(const aclTensor* x, const aclIntArray* outputSize,
                                                             const bool alignCorners, const double scalesH,
                                                             const double scalesW, aclTensor* out,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUpsampleBicubic2dAA的第二段接口，用于执行计算。
 * 
 * 算子功能：upsample_bicubic2d的抗锯齿计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnUpsampleBicubic2dAAGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleBicubic2dAA(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_BICUBIC2D_AA_H_
