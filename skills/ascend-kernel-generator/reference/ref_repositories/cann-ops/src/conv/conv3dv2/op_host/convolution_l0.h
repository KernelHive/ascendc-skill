/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 #ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H_
 #define OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H_
 
 #include "opdev/op_executor.h"
 
 namespace l0op {
 const aclTensor *Conv3dv26HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, aclOpExecutor *executor);
 
 const aclTensor *Conv3dv26HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, aclOpExecutor *executor);
 
 const aclTensor *Conv3dv26HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, bool useHf32, aclOpExecutor *executor);
 
 const aclTensor *Conv3dv2NCDHWFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, bool useHf32, aclOpExecutor *executor);
 
 const aclTensor *Conv3dv2NCDHWBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, aclOpExecutor *executor);
 
 const aclTensor *Conv3dv2NCDHWFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, aclOpExecutor *executor);
 
 bool IsSupportConv3DToConv3DV2();
 }  // namespace l0op
 
 #endif  // OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H_
 