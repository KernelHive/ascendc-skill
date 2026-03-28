/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ADAPTIVE_MAX_POOL3D_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_ADAPTIVE_MAX_POOL3D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#define ACLNN_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAdaptiveMaxPool3dBackward First segment interface. Calculate the workspace size based on the specific calculation process.
 * Function description: Calculates AdaptiveMaxPool3DGrad from the gradOutput and self, at the result gradInput
 * @domain aclnn_ops_train
 * @param [in] gradOutput: gradient Tensor, which is the same as the positive output shape. It is aclTensor on the NPU device side. The data type can be float32/float16/bfloat16. Data format: ND and discontinuous tensors are supported. Only 4- or 5-dimensional tensors are supported.
     *  If there are four dimensions, the value of N is considered as 1, and the value of each dimension must be greater than 0.
     *  If the value is five dimensions, all dimensions except dimension 0 must be greater than 0. When dimension 0 is 0, gradInput is empty.
 * @param [in] self: aclTensor on the NPU device side, positive operator input. The data type can be float32/float16/bfloat16. . Data format: ND and discontinuous tensors are supported. Only 4- or 5-dimensional tensors are supported.
 * @param [in] indices: aclTensor on the NPU device side, which is the output of the forward operator. The index data with the maximum value on the DHW plane is int64 and ND is supported. Supports discontinuous tensors. Only 4- or 5-dimensional tensors are supported.
 * @param [in] gradInput: It is the same as the positive input shape. It is the aclTensor on the NPU device side. The data type can be float16, float32, or bfloat16. Added the data format ND and discontinuous tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the npu device side.
 * @param [out] executor: Return the op executor, including the operator calculation process.
 * @return aclnnStatus: Return the status code.
 */

ACLNN_API aclnnStatus aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                                     const aclTensor* indices, aclTensor* gradInput,
                                                                     uint64_t* workspaceSize,
                                                                     aclOpExecutor** executor);
/**
 * @brief A second interface of aclnnAdaptiveMaxPool3dBackward, used to perform calculation.
 * @param [in] workspace: start address of the workspace memory allocated on the NPU device.
 * @param [in] workspace_size: size of the workspace applied on the NPU device, which is obtained by calling the first segment interface aclnnMaxPool3dWithIndicesBackwardGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: acl stream.
 * @return aclnnStatus: returned status code
 */

ACLNN_API aclnnStatus aclnnAdaptiveMaxPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, 
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ADAPTIVE_MAX_POOL3D_BACKWARD_H_