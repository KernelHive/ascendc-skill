/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_WITH_ARGMAX_H_
#define OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_WITH_ARGMAX_H_

#include "aclnn/aclnn_base.h"
#define ACLNN_API __attribute__((visibility("default")))
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMaxPool3dWithArgmax First segment interface. Calculate the workspace size based on the specific calculation process.
 * Function description: Calculates the aclnnMaxPool3dWithArgmax from the input tensor, at the output contains 2 tensors: out and indices
 * @domain aclnn_ops_infer
 * @param [in] self: aclTensor on the NPU device. the data type can be float32/float16/bfloat16. The data format supports ND.
 * @param [in] kernelSize: aclIntArray type, indicating the maxpooling window size.
 * @param [in] stride: aclIntArray type, the step size of the window movement.
 * @param [in] padding: aclIntArray type, number of padding layers for each edge. The value is negative infinity.
 * @param [in] dilation: aclIntArray type, controls the stride of elements in the window.
 * @param [in] ceilMode: bool type, when true, the output shape is calculated using round-up method. By default is rounding down.
 * @param [in] out: aclTensor on the NPU device. the data type can be float32/float16/bfloat16. The data format supports ND.
 * @param [in] indices: aclTensor on the NPU device. the data type can be int32. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the npu device side.
 * @param [out] executor: Return the op executor, including the operator calculation process.
 * @return aclnnStatus: Return the status code.
 */

ACLNN_API aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize,
                                                               const aclIntArray* stride, const aclIntArray* padding,
                                                               const aclIntArray* dilation, bool ceilMode, aclTensor* out,
                                                               aclTensor* indices, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);
/**
 * @brief A second interface of aclnnMaxPool3dWithArgmax, used to perform calculation.
 * @param [in] workspace: start address of the workspace memory allocated on the NPU device.
 * @param [in] workspaceSize: size of the workspace applied on the NPU device, which is obtained by calling the first segment interface aclnnMaxPool3dWithArgmaxGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: acl stream.
 * @return aclnnStatus: returned status code
 */

ACLNN_API aclnnStatus aclnnMaxPool3dWithArgmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_WITH_ARGMAX_H_