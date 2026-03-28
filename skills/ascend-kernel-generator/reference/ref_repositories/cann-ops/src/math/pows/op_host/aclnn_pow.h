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
#ifndef OP_API_INC_POW_TENSOR_SCALAR_H_
#define OP_API_INC_POW_TENSOR_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnPowTensorScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、INT16、INT8、
 * UINT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型），
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] exponent: npu
 * device侧的aclScalar，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、INT16、INT8、
 * UINT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型）。
 * @param [in] out:npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、INT16、INT8、
 * UINT8、COMPLEX64、COMPLEX128、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型），
 * 且数据类型、数据格式和shape与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* exponent,
                                                           const aclTensor* out, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);
/**
 * @brief aclnnPowTensorScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnPowTensorScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnPowTensorScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream);

/**
 * @brief aclnnInplacePowTensorScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、INT16、INT8、
 * UINT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型），
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] exponent: npu
 * device侧的aclScalar，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、INT16、INT8、
 * UINT8、COMPLEX64、COMPLEX128、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* exponent,
                                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplacePowTensorScalar的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小,
 * 由第一段接口aclnnInplacePowTensorScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplacePowTensorScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream);

/**
 * @brief aclnnPowScalarTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(const aclScalar* self, const aclTensor* exponent,
                                                           const aclTensor* out, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnPowScalarTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_POW_TENSOR_SCALAR_H_
