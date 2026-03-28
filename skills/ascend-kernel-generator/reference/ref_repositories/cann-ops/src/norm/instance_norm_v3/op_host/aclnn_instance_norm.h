/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_INSTANCE_NORM_H_
#define OP_API_INC_INSTANCE_NORM_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInstanceNorm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成计算 InstanceNorm 的值。
 *
 * @param [in] x: npu
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。
 * 支持连续和非连续的Tensor，数据格式支持ND。
 * @param [in] gamma: npu
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。
 * 支持连续和非连续的Tensor，数据格式支持ND。
 * @param [in] beta: npu
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。
 * 支持连续和非连续的Tensor，数据格式支持ND。
 * @param [in] dataFormat:
 * 字符串儿。表示输入tensor实际的输入数据排布， 当前仅支持NHWC和NCHW。
 * @param [in] eps:
 * double类型数据，norm计算时需要的epsilon参数。
 * @param [in] y: npu
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。
 * 且数据类型和 x 保持一致，支持连续和非连续的Tensor，数据格式支持ND。
 * @param [in] mean: npu
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。
 * 且数据类型和 x 保持一致，支持连续和非连续的Tensor，数据格式支持ND。
 * @param [in] variance: npu
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。
 * 且数据类型和 x 保持一致，支持连续和非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnInstanceNormGetWorkspaceSize(const aclTensor *x,
    const aclTensor *gamma, const aclTensor *beta, const char *dataFormat, double eps, aclTensor *y, aclTensor *mean,
    aclTensor *variance, uint64_t *workspaceSize, aclOpExecutor **executor);
/**
 * @brief aclnnInstanceNorm的第二段接口，用于执行计算。
 *
 * 算子功能：完成计算输入的k个极值及下标。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInstanceNormGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnInstanceNorm(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_INSTANCE_NORM_H_
