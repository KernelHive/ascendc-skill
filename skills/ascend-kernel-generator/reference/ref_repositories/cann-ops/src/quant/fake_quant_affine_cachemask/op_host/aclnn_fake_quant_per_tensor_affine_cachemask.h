/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file aclnn_fake_quant_per_tensor_affine_cachemask.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS_H_
#define OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFakeQuantPerTensorAffineCachemask的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：
 *   1）fake_quant_enabled >= 1: 对于输入数据self，使用scale和zero_point对输入self进行伪量化处理
 * 并根据quant_min和quant_max对伪量化输出进行值域更新，最终返回结果out及对应位置掩码mask。
 *   2) fake_quant_enabled < 1: 返回结果out为self.clone(）对象，掩码mask为全True。
 *
 * 计算图：如下
 * 场景一：当fake_quant_enabled小于1时，使用ViewCopy计算out，使用Fill计算mask。
 * ```mermaid
 * graph LR
 *   A1[(self)] -->B1(l0op::Contiguous)-->C1(l0op::ViewCopy)-->D1[(out)]
 *   A1[(self)] -->B1(l0op::Contiguous)-->D(l0op::Fill)-->C2(l0op::ViewCopy)-->D2[(mask)]
 * ```
 *
 * 场景二：当fake_quant_enabled大于等于1时，因为scale和zero_point的size大小都为1，故先broadcast后，再使用FakeQuantPerChannelAffineCachemask算子完成计算。
 * ```mermaid
 * graph LR
 *   A1[(self)] -->B1(l0op::Contiguous)-->C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A2[(scale)] -->B2(l0op::Contiguous)-->F1(l0op::BroadcastTo)-->C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A3[(zeroPoint)]-->B3(l0op::Contiguous)-->F2(l0op::BroadcastTo)-->C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A4((quantMin)) --> C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A5((quantMax)) --> C(l0op::FakeQuantPerChannelAffineCachemask)
 *   C(l0op::FakeQuantPerChannelAffineCachemask)-->D1(l0op::ViewCopy)-->E1[(out)]
 *   C(l0op::FakeQuantPerChannelAffineCachemask)-->D2(l0op::ViewCopy)-->E2[(mask)]
 * ```
 *
 * @param [in] self:
 * Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。支持非连续的Tensor，[数据格式](common/数据格式.md)支持ND。
 * @param [in] scale: Device侧的aclTensor，表示输入伪量化的缩放系数。数据类型支持FLOAT16、FLOAT32，size大小为1。
 * @param [in] zeroPoint: Device侧的aclTensor，表示输入伪量化的零基准参数。数据类型支持INT32，size大小为1。
 * @param [in] fake_quant_enabled: Host侧的浮点型，表示是否进行伪量化计算，数据类型支持FLOAT。
 * @param [in] quantMin: Host侧的整型，表示输入数据伪量化后的最小值，数据类型支持INT。
 * @param [in] quantMax: Host侧的整型，表示输入数据伪量化后的最大值，数据类型支持INT。
 * @param [out] out:
 * Device侧的aclTensor，数据类型支持LOAT16、FLOAT32，支持非连续Tensor，[数据格式](common/数据格式.md)支持ND。
 * @param [out] mask: Device侧的aclTensor，数据类型支持BOOL，支持非连续Tensor，[数据格式](common/数据格式.md)支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, float fakeQuantEnbled, int64_t quantMin,
    int64_t quantMax, aclTensor* out, aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFakeQuantPerTensorAffineCachemask的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnFakeQuantPerTensorAffineCachemask获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerTensorAffineCachemask(void* workspace, uint64_t workspaceSize,
                                                             aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS_H_
