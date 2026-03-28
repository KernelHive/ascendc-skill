/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_AVGPOOL3D_H_
#define OP_API_INC_AVGPOOL3D_H_

#include "aclnn/aclnn_base.h"

#define ACLNN_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAvgPool3d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对输入self进行平均池化运算。
 *
 * @param [in] self: npu
 * device侧的aclTensor，支持空tensor场景，数据类型支持FLOAT16、BFLOAT16和FLOAT。支持4维或5维。支持非连续的Tensor。支持数据格式为ND。
 * @param [in] kernelSize: npu
 * device侧的aclIntArray，长度为1(kD=kH=kW)或3(kD,kH,kW)，表示池化窗口大小。数据类型支持INT32和INT64。数值必须大于0。
 * @param [in] stride: npu
 * device侧的aclIntArray，长度为0(默认为kernelSize)或1(sD=sH=sW)或3(sD,sH,sW)，表示池化操作的步长。
 * 数据类型支持INT32和INT64。数值必须大于0。
 * @param [in] padding: npu
 * device侧的aclIntArray，长度为1(padD=padH=padW)或3(padD,padH,padW)，表示在输入的D、H、W方向上padding补0的层数。
 * 数据类型支持INT32和INT64。数值在[0, kernelSize/2]的范围内。
 * @param [in] ceilMode: 数据类型支持BOOL。表示计算输出shape时，向下取整（False），否则向上取整。
 * @param [in] countIncludePad: 数据类型支持BOOL。表示平均计算中包括零填充（True），否则不包括。
 * @param [in] divisorOverride: 数据类型支持INT64。如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效。
 * @param [out] output: npu
 * device侧的aclTensor，输出Tensor，数据类型支持FLOAT16、BFLOAT16和FLOAT。支持4维或5维。支持非连续的Tensor。支持数据格式为ND。
 * 数据类型、数据格式需要与self一致。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAvgPool3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize,
                                                     const aclIntArray* stride, const aclIntArray* padding,
                                                     bool ceilMode, bool countIncludePad,
                                                     int64_t divisorOverride, aclTensor* out,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAvgPool3d的第二段接口，用于执行计算。
 *
 * 算子功能： 对输入self进行平均池化运算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAtan2GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAvgPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AVGPOOL3D_H_
