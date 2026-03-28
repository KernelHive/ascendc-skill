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
#ifndef OP_API_INC_MASKED_SELECT_H_
#define OP_API_INC_MASKED_SELECT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMaskedSelect的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：根据一个布尔掩码张量（mask）中的值选择输入张量（self）中的元素作为输出,形成一个新的一维张量。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL
 * shape需要与mask满足[broadcast关系]()。支持[非连续的Tensor]()，数据格式支持ND.
 * @param [in] mask: npu
 * device侧的aclTensor，仅支持bool或uint8，如果为uint8，其值必须为0或1，shape需要与self满足[broadcast关系]()。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaskedSelectGetWorkspaceSize(const aclTensor* self, const aclTensor* mask, aclTensor* out,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnMaskedSelect的第二段接口，用于执行计算。
 *
 * 算子功能：对输入的 Tensor self， 根据mask进行按位掩码的选择操作，选择mask掩码中非零位置对应的元素，形成一个新的
 * Tensor，并返回。 实现说明： api计算的基本路径：
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMaskedSelect(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MASKED_SELECT_H_
