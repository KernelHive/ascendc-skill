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
 * @file convolutionbackward.h
 */
#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

struct ConvBackpropParams {
  const aclTensor *input;
  const aclTensor *weight;
  const aclTensor *outBackprop;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  int groups;
};

// Conv2dBackpropInput
// 5HD->FZ with Fp16
const aclTensor *Conv2DBackpropInputFp162Fp16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 5HD->FZ with Fp32
const aclTensor *Conv2DBackpropInputFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 5HD->FZ with Hf32
const aclTensor *Conv2DBackpropInputHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, int groups, aclOpExecutor *executor);
// 1971 5HD->FZ with Bf16
const aclTensor *Conv2DBackpropInputBf162Bf16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);

// Conv2dBackpropFilter
// 5HD->FZ with Fp16
const aclTensor *Conv2DBackpropFilterFp162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 5HD->FZ with Fp32
const aclTensor *Conv2DBackpropFilterFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);
// 1971 5HD->FZ with Hf32
const aclTensor *Conv2DBackpropFilterHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                          const aclIntArray *stride, const aclIntArray *padding,
                                          const aclIntArray *dilation, int groups, aclOpExecutor *executor);
// 1971 5HD->FZ with Bf16
const aclTensor *Conv2DBackpropFilterBf162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// Conv2DBackpropFilterV2
// 1971 5HD->FZ_C04 with Fp16
const aclTensor *Conv2DBackpropFilterV2Fp162Fp32(const aclTensor *input, const aclTensor *weight,
                                                 const aclTensor *outBackprop, const aclIntArray *stride,
                                                 const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                                 aclOpExecutor *executor);

// 1971 5HD->FZ_C04 with Bf16
const aclTensor *Conv2DBackpropFilterV2Bf162Fp32(const aclTensor *input, const aclTensor *weight,
                                                 const aclTensor *outBackprop, const aclIntArray *stride,
                                                 const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                                 aclOpExecutor *executor);

// Conv3dBackpropFilter
// 6HD->FZ_3D with Fp16
const aclTensor *Conv3DBackpropFilterFp162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 6HD->FZ_3D with Fp32
const aclTensor *Conv3DBackpropFilterFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 6HD->FZ_3D with Bf16
const aclTensor *Conv3DBackpropFilterBf162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 6HD->FZ_3D with Hf32
const aclTensor *Conv3DBackpropFilterHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                          const aclIntArray *stride, const aclIntArray *padding,
                                          const aclIntArray *dilation, int groups, aclOpExecutor *executor);

// Conv3dBackpropInput
// 6HD->FZ with Fp16
const aclTensor *Conv3DBackpropInputFp162Fp16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 6HD->FZ with Fp32
const aclTensor *Conv3DBackpropInputFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 6HD->FZ with Hf32
const aclTensor *Conv3DBackpropInputHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, int groups, aclOpExecutor *executor);
// 1971 6HD->FZ with Bf16
const aclTensor *Conv3DBackpropInputBf162Bf16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);

bool IsConv3DBackpropInputV2(const ConvBackpropParams &params);

bool IsConv3DBackpropFilterV2(const ConvBackpropParams &params);

bool IsInputTransdataWhiteListCase(const ConvBackpropParams &params);

bool IsConv2DBackpropInputV2(const ConvBackpropParams &params);

bool IsConv2DBackpropFilterV3(const ConvBackpropParams &params);
}  // namespace l0op

#endif  // OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_
