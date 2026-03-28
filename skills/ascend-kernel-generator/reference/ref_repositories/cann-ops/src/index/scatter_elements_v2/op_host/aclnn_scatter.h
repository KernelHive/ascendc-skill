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
#ifndef OP_API_INC_SCATTER_H_
#define OP_API_INC_SCATTER_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScatterEle的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] self: npu device侧的aclTensor,
 * 数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、
 * COMPLEX64、COMPLEX128类型。self的维度数量需要与index、src相同。self的数据类型需要与src一致。支持空tensor, 支持
 * 非连续的tensor。数据格式支持ND。
 * @param [in] dim: 用来scatter的维度，数据类型为INT64。范围为[-self的维度数量, self的维度数量-1]。
 * @param [in] index: npu device侧的aclTensor。数据类型支持INT32,
 * INT64。index的维度数量需要与self、src相同。支持空tensor， 支持非连续的tensor。数据格式支持ND。
 * @param [in] src: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、
 * COMPLEX64、COMPLEX128类型。src的维度数量需要与self、index相同。src的数据类型需要与self一致。支持空tensor，
 * 支持非连续的 tensor。数据格式支持ND。
 * @param [in] reduce: 选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。
 * 具体操作含义如下：
 * 0：表示替换操作，将src中的对应位置的值按照index替换到out中的对应位置
 * 1：表示累加操作，将src中的对应位置的值按照index累加到out中的对应位置
 * 2：表示累乘操作，将src中的对应位置的值按照index累乘到out中的对应位置
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、
 * COMPLEX64、COMPLEX128类型。数据格式、数据类型、shape需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnScatterEleGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                   const aclTensor* src, int64_t reduce, aclTensor* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnScatterEle的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterEleGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatterEle(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   const aclrtStream stream);

/**
 * @brief aclnnScatterEleValue的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] self: npu device侧的aclTensor,
 * 数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、
 * COMPLEX64、COMPLEX128类型。self的维度数量需要与index、src相同。self的数据类型需要与src一致。支持空tensor, 支持
 * 非连续的tensor。数据格式支持ND。
 * @param [in] dim: 用来scatter的维度，数据类型为INT64。范围为[-self的维度数量, self的维度数量-1]。
 * @param [in] index: npu device侧的aclTensor。数据类型支持INT32,
 * INT64。index的维度数量需要与self、src相同。支持空tensor， 支持非连续的tensor。数据格式支持ND。
 * @param [in] value: host侧的aclScalar,
 * 数据类型支持UINT8、INT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、
 * COMPLEX128。当value为COMPLEX时，self也必须为COMPLEX tensor。
 * @param [in] reduce: 选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。
 * 具体操作含义如下：
 * 0：表示替换操作，将src中的对应位置的值按照index替换到out中的对应位置
 * 1：表示累加操作，将src中的对应位置的值按照index累加到out中的对应位置
 * 2：表示累乘操作，将src中的对应位置的值按照index累乘到out中的对应位置
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、
 * COMPLEX64、COMPLEX128类型。数据格式、数据类型、shape需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnScatterEleValueGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                        const aclScalar* value, int64_t reduce, aclTensor* out,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnScatterEle的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterEleValueGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatterEleValue(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        const aclrtStream stream);

/**
 * @brief aclnnInplaceScatterEle的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] selfRef:
 * 数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128类型。
 * selfRef的维度数量需要与index、src相同。selfRef的数据类型需要与src一致。支持空tensor，
 * 支持非连续的tensor。数据格式支持ND。
 * @param [in] dim: 用来scatter的维度，数据类型为INT64。范围为[-selfRef的维度数量, selfRef的维度数量-1]。
 * @param [in] index: npu device侧的aclTensor。数据类型支持INT32,
 * INT64。index的维度数量需要与selfRef、src相同。支持空tensor， 支持非连续的tensor。数据格式支持ND。
 * @param [in] src: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、
 * COMPLEX64、COMPLEX128类型。src的维度数量需要与selfRef、index相同。src的数据类型需要与selfRef一致。支持空tensor，
 * 支持非连续的 tensor。数据格式支持ND。
 * @param [in] reduce: 选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。
 * 具体操作含义如下：
 * 0：表示替换操作，将src中的对应位置的值按照index替换到out中的对应位置
 * 1：表示累加操作，将src中的对应位置的值按照index累加到out中的对应位置
 * 2：表示累乘操作，将src中的对应位置的值按照index累乘到out中的对应位置
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceScatterEleGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                          const aclTensor* src, int64_t reduce, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceScatterEle的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceScatterEleGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceScatterEle(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

/**
 * @brief aclnnInplaceScatterEleValue的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] selfRef: npu device侧的aclTensor, 数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、
 * DOUBLE、COMPLEX64、COMPLEX128类型。selfRef的维度数量需要与index、src相同。selfRef的数据类型需要与src一致。支持空tensor,
 * 支持 非连续的tensor。数据格式支持ND。
 * @param [in] dim: 用来scatter的维度，数据类型为INT64。范围为[-selfRef的维度数量, selfRef的维度数量-1]。
 * @param [in] index: npu device侧的aclTensor。数据类型支持INT32,
 * INT64。index的维度数量需要与selfRef、src相同。支持空tensor， 支持非连续的tensor。数据格式支持ND。
 * @param [in] value: host侧的aclScalar,
 * 数据类型支持UINT8、INT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、
 * COMPLEX128。当value为COMPLEX时，selfRef也必须为COMPLEX tensor。
 * @param [in] reduce: 选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。
 * 具体操作含义如下：
 * 0：表示替换操作，将src中的对应位置的值按照index替换到out中的对应位置
 * 1：表示累加操作，将src中的对应位置的值按照index累加到out中的对应位置
 * 2：表示累乘操作，将src中的对应位置的值按照index累乘到out中的对应位置
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、
 * COMPLEX64、COMPLEX128类型。数据格式、数据类型、shape需要与selfRef一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceScatterEleValueGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                               const aclScalar* value, int64_t reduce,
                                                               uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceScatterEleValue的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceScatterEleValueGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceScatterEleValue(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
