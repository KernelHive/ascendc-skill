/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef OP_API_INC_LEVEL2_ACL_RFFT1D_H_
#define OP_API_INC_LEVEL2_ACL_RFFT1D_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclRfft1D First segment interface. Calculate the workspace size based on the specific calculation process.
 * Function description: Calculates the RFFT of the input tensor and outputs the result.
 * Calculation formula:
 * $$ out = Rfft(input) $$
 * Calculation chart:
 ```mermaid
 * graph LR
 * A[(self)]--->B([l0op::Rfft1D])
 * B--->C([l0op::ViewCopy])
 * C--->D[(out)]
 * ` ` `
 * @domain aclnn_ops_infer
 * Parameter description:
 * @param [in] Input
 * Input tensor. The type is FLOAT. The data format supports ND, but does not support discontinuous tensors.
 * @param [in] out
 * Output tensor. The type is FLOAT. The data format supports ND, but does not support discontinuous tensors.
 * @param [out] workspace_size: Returns the workspace size that a user needs to apply for on the NPU device.
 * @param [out] executor: Return the op executor, including the operator calculation process.
 * @return aclnnStatus: Return the status code.
 */
 
aclnnStatus aclRfft1DGetWorkspaceSize(const aclTensor* self, int64_t n, int64_t dim, int64_t norm, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief A second interface of aclRfft1D, used to perform calculation.
 * @param [in] workspace: start address of the workspace memory allocated on the NPU device.
 * @param [in] workspace_size: size of the workspace applied on the NPU device, which is obtained by calling the first segment interface aclRfft1DGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: acl stream.
 * @return aclnnStatus: returned status code
 */
                                    
aclnnStatus aclRfft1D(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACL_RFFT1D_H_
