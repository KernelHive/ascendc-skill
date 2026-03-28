/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_SRC_CONVOLUTION_UTIL_H_
#define OP_API_SRC_CONVOLUTION_UTIL_H_

#include <map>
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

struct ConvolutionOpInfo {
    op::DataType inputDtype;
    op::Format inputFormat;
    op::DataType weightDtype;
    op::Format weightFormat;
    op::DataType biasDtype;
    op::Format biasFormat;
    op::DataType outputDtype;
    op::Format outputFormat;
};

class Conv2DSplitWInfo {
public:
    void InitConv2DSplitWInfo(const aclTensor* input, const aclTensor* weight, const aclIntArray* stride,
                            const aclIntArray* padding, const aclIntArray* dilation);
    bool CanSwitchSplitW(const aclTensor* bias, aclTensor* output, int64_t groups, const ConvolutionOpInfo& opInfo);

private:
    bool CheckConv2DTbeOptFlag(const ConvolutionOpInfo& opInfo);
    bool CheckConv2DPad();
    bool CheckConv2DInput();
    bool CheckBasicInfoInSplitW(int64_t groups, const ConvolutionOpInfo& opInfo);
    bool CheckLoad3dIns();
    bool CheckLoadL1InSplitW(const aclTensor* bias, aclTensor* output);

private:
    int64_t hi = 0;
    int64_t wi = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t padU = 0;
    int64_t padD = 0;
    int64_t padL = 0;
    int64_t padR = 0;
    int64_t biasTypeSize = 0;
    int64_t k0 = 0;
};

aclnnStatus ChangeConv2dAttrToConv3d(const aclIntArray* &stride, const aclIntArray* &padding,
                                    const aclIntArray* &dilation, aclOpExecutor* executor);
aclnnStatus ChangeConv2dInputToConv3d(const aclTensor* &input, const aclTensor* &weight, aclOpExecutor* executor);
const aclTensor* View5dAs4dForOutput(const aclTensor* input, aclOpExecutor* executor);

#endif  // OP_API_SRC_CONVOLUTION_UTIL_H_