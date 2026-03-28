/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_UNAMPLE_NEAREST_EXACT1D_H_
#define OP_API_INC_UNAMPLE_NEAREST_EXACT1D_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleNearestExact1d的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对由三个输入通道组成的输入信号应用最近邻精确插值算法进行上采样插值
 * 计算公式：
 * out(N, C, l) = self(N, C, min(floor((l + 0.5) * scales),  L- 1))
 * @domain aclnn_ops_train
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。支持非连续Tensor，数据格式支持ND、NCL。
 * @param [in]   outputSize
 * 输出的size大小，数据类型支持INT32、INT64。
 * @param [in]   scales
 * 输出的缩放系数，数据类型支持DOUBLE。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。支持非连续Tensor，数据格式支持ND、NCL。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnUpsampleNearestExact1dGetWorkspaceSize(const aclTensor *self,
    const aclIntArray *outputSize, double scales, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnUpsampleNearestExact1d的第二段接口，用于执行计算。
 *
 * 功能描述：对由三个输入通道组成的输入信号应用最近邻精确插值算法进行上采样插值。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnUpsampleNearestExact1dGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnUpsampleNearestExact1d(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_EXACT1D_H_
