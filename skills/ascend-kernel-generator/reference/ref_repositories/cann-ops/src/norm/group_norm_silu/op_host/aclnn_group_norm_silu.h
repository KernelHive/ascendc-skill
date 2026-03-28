/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_group_norm_silu.h
 * \brief
 */

#ifndef OP_API_INC_OPEN_GROUP_NORM_SILU_H___
#define OP_API_INC_OPEN_GROUP_NORM_SILU_H___

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupNormSilu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成GroupNorm和Silu融合算子功能。
 * 计算公式是：
 * y=((x-Ex) / (sqrt(Var(x)+ϵ​x))​)∗ γ + β
 * 将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值
 *
 * 附：Silu计算公式为：
 * f(x) = x * sigmoid(x)
 * 当x大于0时,Silu激活函数将会放大x,而当x小于0时,Silu激活函数将会降低x,可以抑制过拟合
 * ```
 *
 * @param [in] self：计算输入，shape大于等于2维
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] gamma：计算输入，shape为1维，且等于c维度。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] beta：shape为1维，且等于c维度。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND，支持非连续的Tensor。
 * @param [in] group：计算属性，host侧的整数，数据类型支持INT64,分组信息
 * @param [in] esp：计算属性，host侧的浮点数，数据类型支持double,默认le-5
 * @param [out] out：y的输出。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] meanOut：均值的输出。
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] rstdOut：1/sqrt(Var(x)+ϵ​x)结果输出
 * npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，
 * 数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupNormSiluGetWorkspaceSize(const aclTensor* self, const aclTensor* gamma,
                                   const aclTensor* beta, int64_t group, double eps,
                                   aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut,
                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupNormSilu的第二段接口，用于执行计算
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupNormSilu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnGroupNormSiluV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupNormSiluV2GetWorkspaceSize(const aclTensor* self, const aclTensor* gamma,
                                     const aclTensor* beta, int64_t group, double eps,
                                     bool activateSilu, aclTensor* out, aclTensor* meanOut,
                                     aclTensor* rstdOut, uint64_t* workspaceSize,
                                     aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnGroupNormSiluV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC__OPEN_GROUP_NORM_SILU_H___
