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
 * \file aclnn_ctc_loss_backward.h
 * \brief
 */
#ifndef OP_API_INC_CTCLOSSBACKWARD_H_
#define OP_API_INC_CTCLOSSBACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCtcLossBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * _ctc_loss的反向传播。
 *
 *
 * 计算图：
 * ```mermaid
 * graph LR
 * A[(logProbs)] -->B([l0op::Contiguous])-->D([l0op::CTCLossV2Grad])
 * A1[(targets)] -->B1([l0op::Contiguous])--> D
 * A6[(gradOut)] -->B6([l0op::Contiguous])-->D
 * A7[(negLogLikelihood)] -->B7([l0op::Contiguous])-->D
 * A8[(logAlpha)] -->B8([l0op::Contiguous])-->D
 * A2[(targetLengths)] -->B2([ConvertToTensor])-->D
 * A3[(inputLengths)] -->B3([ConvertToTensor])-->D
 * A4((blank)) -->D
 * A5((zeroInfinity)) -->D
 * D--> F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] gradOut(aclTensor*): 表示梯度更新系数，数据类型支持FLOAT,
 * DOUBLE数据类型(数据类型必须和logProbs一致)，该Tensor必须为1维，支持非连续的Tensor，数据格式支持ND。
 * @param [in] logProbs(aclTensor*): 数据类型支持FLOAT,DOUBLE数据类型，shape为($T,N,C$)，
 * $T$为输入长度，$N$为批处理大小，$C$为类别数，必须大于0，包括空白标识，该Tensor表示输出的对数概率，
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] targets(aclTensor*): 数据类型支持INT64,INT32,BOOL,FLOAT,FLOAT16数据类型，当shape为($N,S$)，
 * $S$为不小于$targetLengths$中的最大值的值；或者shape为(SUM($targetLengths$))，假设$targets$是未填充的而且在1维内级联的；
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] inputLengths(aclIntArray*)：数组长度为$N$，
 * 数组中的每个值必须小于等于$T$。
 * @param [in] targetLengths(aclIntArray*)：数组长度为$N$，
 * 当targets的shape为($N,S$)时，数组中的每个值必须小于等于$S$。
 * @param [in] negLogLikelihood(aclTensor*)：数据类型支持FLOAT,DOUBLE数据类型(数据类型必须和logProbs一致)，
 * 表示相对于每个输入节点可微分的损失值，该Tensor必须为1维，支持非连续的Tensor，数据格式支持ND。
 * @param [in] logAlpha(aclTensor*)：数据类型支持FLOAT,DOUBLE数据类型(数据类型必须和logProbs一致)，
 * 表示输入到目标的可能跟踪的概率，该Tensor必须为3维，支持非连续的Tensor，数据格式支持ND。
 * @param [in] blank(int)：int整型，空白标识，默认为0，数值必须小于$C$大于等于0。
 * @param [in] zeroInfinity(bool)：bool类型，表示是否将无限损耗和相关梯度归零，默认值为$False$。
 * @param [out] out(aclTensor*): 表示CTC的损失梯度，数据类型支持FLOAT,DOUBLE，shape为($T,N,C$)，
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* logProbs,
                                                           const aclTensor* targets, const aclIntArray* inputLengths,
                                                           const aclIntArray* targetLengths,
                                                           const aclTensor* negLogLikelihood, const aclTensor* logAlpha,
                                                           int64_t blank, bool zeroInfinity, aclTensor* out,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnCtcLossBackward 的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnCtcLossBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCtcLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CTCLOSSBACKWARD_H_
