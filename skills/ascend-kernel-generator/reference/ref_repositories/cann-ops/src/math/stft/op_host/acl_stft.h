/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACL_STFT_H_
#define OP_API_INC_LEVEL2_ACL_STFT_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclStft的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_fft
 * 算子功能：返回输入Tensor做Stft的结果
 * 计算公式：
 * $$ out = Stft(input) $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)]--->B([l0op::Stft])
 *   B--->C([l0op::ViewCopy])
 *   C--->D[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，待计算的输入，要求是一个1D/2D的Tensor，shape为(L)/(B, L)，
 * 其中，L为时序采样序列的长度，B为时序采样序列的个数。数据类型支持FLOAT32、DOUBLE、COMPLEX64、COMPLEX128，
 * 支持非连续的Tensor，数据格式要求为ND。
 * @param [in] windowOptional: npu device侧的aclTensor，可选参数，要求是一个1D的Tensor，shape为(winLength)，
 * winLength为STFT窗函数的长度。 数据类型支持FLOAT32、DOUBLE、COMPLEX64、COMPLEX128，且数据类型与self保持一致，数据格式要求为ND。
 * @param [in] out: npu device侧的aclTensor，self在window内的傅里叶变换结果，要求是一个2D/3D/4D的Tensor，
 * 数据类型支持FLOAT32、DOUBLE、COMPLEX64、COMPLEX128，支持非连续的Tensor，数据格式要求为ND。
 * @param [in] nFft: 必选参数，Host侧的int，FFT的点数（大于0）。
 * @param [in] hopLength: 必选参数，Host侧的int，滑动窗口的间隔（大于0）。
 * @param [in] winLength: 必选参数，Host侧的int，window的大小（大于0）。
 * @param [in] normalized: 必选参数，Host侧的bool，是否对傅里叶变换结果进行标准化。
 * @param [in] onesided: 必选参数，Host侧的bool，是否返回全部的结果或者一半结果。
 * @param [in] returnComplex: 必选参数，Host侧的bool，确认返回值是complex tensor或者是实部、虚部分开的tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包括算子计算流程
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclStftGetWorkspaceSize(const aclTensor* self, const aclTensor* windowOptional, aclTensor* out, int64_t nFft,
                                    int64_t hopLength, int64_t winLength, bool normalized, bool onesided,
                                    bool returnComplex, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclStft的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAbsGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclStft(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACL_STFT_H_