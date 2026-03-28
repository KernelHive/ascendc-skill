/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GRID_SAMPLER2D_H_
#define OP_API_INC_GRID_SAMPLER2D_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGridSampler2D的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：提供一个输入tensor以及一个对应的flow-field网格，然后根据grid中每个位置提供的坐标信息，
 * 将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(input)] --> B([l0op::Contiguous]) --> C([l0op::GridSampler2D])
 *     D[(grid)] --> E([l0op::Contiguous]) --> C
 *     F((interpolationMode)) --> C
 *     G((paddingMode)) --> C
 *     H((alignCorners)) --> C
 *     C --> I([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] input: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [in] grid: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [in] interpolationMode：host侧的int64_t， 表示插值模式，0：bilinear（双线性插值），1：nearest（最邻近插值），
 * 2（不支持）：bicubic（双三次插值）。
 * @param [in] paddingMode：host侧的int64_t，表示填充模式，即当（x,y）取值超过输入特征图采样范围时，返回一个特定值，
 * 有0：zeros、1：border、2：reflection三种模式。
 * @param [in] alignCorners：host侧的bool，表示设定特征图坐标与特征值的对应方式，设定为true时，特征值位于像素中心。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGridSampler2DGetWorkspaceSize(const aclTensor *input,
    const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGridSampler2D的第二段接口，用于执行计算。
 *
 * 算子功能：提供一个输入tensor以及一个对应的flow-field网格，然后根据grid中每个位置提供的坐标信息，
 * 将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(input)] --> B([l0op::Contiguous]) --> C([l0op::GridSampler2D])
 *     D[(grid)] --> E([l0op::Contiguous]) --> C
 *     F((interpolationMode)) --> C
 *     G((paddingMode)) --> C
 *     H((alignCorners)) --> C
 *     C --> I([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGridSampler2DGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGridSampler2D(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GRID_SAMPLER2D_H_