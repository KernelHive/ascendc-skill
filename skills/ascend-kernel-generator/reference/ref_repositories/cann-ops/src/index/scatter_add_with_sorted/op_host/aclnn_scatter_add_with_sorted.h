/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SCATTER_H_
#define OP_API_INC_SCATTER_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScatter的第一段接口，根据具体的计算流程，计算workspace大小。
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
ACLNN_API aclnnStatus aclnnScatterGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                   const aclTensor* src, int64_t reduce, aclTensor* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnScatter的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   const aclrtStream stream);

/**
 * @brief aclnnScatterValue的第一段接口，根据具体的计算流程，计算workspace大小。
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
ACLNN_API aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                        const aclScalar* value, int64_t reduce, aclTensor* out,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnScatter的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterValueGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatterValue(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        const aclrtStream stream);

/**
 * @brief aclnnInplaceScatter的第一段接口，根据具体的计算流程，计算workspace大小。
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
ACLNN_API aclnnStatus aclnnInplaceScatterGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                          const aclTensor* src, int64_t reduce, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceScatter的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceScatterGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

/**
 * @brief aclnnInplaceScatterValue的第一段接口，根据具体的计算流程，计算workspace大小。
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
ACLNN_API aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                               const aclScalar* value, int64_t reduce,
                                                               uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceScatterValue的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成scatter操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceScatterValueGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceScatterValue(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

/**
 * @brief aclnnScatterAdd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 将源tensor中的值按指定的轴方向和index tensor中的位置关系逐个填入输出tensor中，
 * 若有多于一个src值被填入到self的同一位置，那么这些值将会在这一位置上进行累加
 * @param [in] self: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8,
 * 支持非连续的Tensor，数据格式支持ND,
 * @param [in] dim: host侧的num, 数据类型支持INT64。
 * @param [in] index: npu device侧的aclTensor，数据类型支持INT32, int64类型，dim反向的维度数量需要与src相同。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] src: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, INT32, INT8,
 * UINT8类型，dim反向的维度数量需要与src相同。 支持非连续的Tensor，数据格式支持ND，且数据类型与self保持一致。
 * @param [in] out: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8,
 * 数据类型,数据格式,tensor shape需要与self保持一致
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                      const aclTensor* src, aclTensor* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief: aclnnScatterAdd的第二段接口，用于执行计算
 *
 * 算子功能: 将源tensor中的值按指定的轴方向和index tensor中的位置关系逐个填入输出tensor中，
 * 若有多于一个src值被填入到self的同一位置，那么这些值将会在这一位置上进行累加
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatterAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
