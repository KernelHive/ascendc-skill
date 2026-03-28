/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_MSE_LOSS_BACKWARD_H_
#define OP_API_INC_MSE_LOSS_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMseLossBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：均方误差函数的反向传播。
 *
 * @param [in] gradOutput：npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，数据类型需要与self相同，shape需要与self、target
 * 满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] self：npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，shape需要与gradOutput、target
 * 满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] target：npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，数据类型需要与self相同，shape需要与gradOutput、self
 * 满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] reduction：host侧的int64，指定要应用到输出的缩减，支持 0('none') | 1('mean') | 2('sum')。'none'
 * 表示不应用减少， 'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。
 * @param [in] out：npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，shape需要是target与self、gradOutput
 * broadcast之后的shape。 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMseLossBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                           const aclTensor* target, int64_t reduction, aclTensor* out,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMseLossBackward的第二段接口，用于执行计算。
 *
 * 算子功能：均方误差函数的反向传播。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMseLossBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMseLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MSE_LOSS_BACKWARD_H_
