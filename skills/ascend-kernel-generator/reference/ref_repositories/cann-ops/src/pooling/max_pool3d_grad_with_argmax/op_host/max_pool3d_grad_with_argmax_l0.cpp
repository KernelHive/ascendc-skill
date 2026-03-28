/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "max_pool3d_grad_with_argmax_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(MaxPool3DGradWithArgmax);

static const std::initializer_list<op::DataType> NULL_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<op::DataType> GRAD_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                            op::DataType::DT_FLOAT,
                                                                            op::DataType::DT_BF16};

static const size_t DHW_DIMS = 3;

static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910_93:
    case SocVersion::ASCEND910B: {
      return GRAD_DTYPE_SUPPORT_LIST;
    }
    case SocVersion::ASCEND910: {
      return NULL_DTYPE_SUPPORT_LIST;
    }
    default: {
      return NULL_DTYPE_SUPPORT_LIST;
    }
  }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *gradOutput) {
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    if (!CheckType(gradOutput ->GetDataType(), dtypeSupportList)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput dtype not supported. support list is %s",
                op::ToString(dtypeSupportList).GetString());
        return false;
    }
    return true;
}

const inline aclTensor* MaxPool3DGradWithArgmaxAiCore(const aclTensor *gradOutput, const aclTensor *self,
                                                      const aclTensor *indices, const aclIntArray *kernelSize,
                                                      const aclIntArray *stride, const aclIntArray *padding,
                                                      const aclIntArray *dilation, bool ceilMode,
                                                      aclTensor *gradInput,
                                                      aclOpExecutor *executor) {
    L0_DFX(MaxPool3DGradWithArgmaxAiCore, gradOutput, self, indices, kernelSize, stride, padding, dilation, 
           ceilMode, gradInput);
    ADD_TO_LAUNCHER_LIST_AICORE(MaxPool3DGradWithArgmax, OP_INPUT(self, gradOutput, indices), OP_OUTPUT(gradInput),
  							                             OP_ATTR(kernelSize, stride, padding, dilation, ceilMode));
    return gradInput;
}

const aclTensor* MaxPool3DGradWithArgmax(const aclTensor *gradOutput, const aclTensor *self,
                                         const aclTensor *indices, const aclIntArray *kernelSize,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, bool ceilMode,
                                         aclOpExecutor *executor) {
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kernelD = kernelRef[0];
    const int64_t kernelH = (kernelRef.Size() == 1) ? kernelD : kernelRef[1];
    const int64_t kernelW = (kernelRef.Size() == 1) ? kernelD : kernelRef[2];

    const aclIntArray& strideRef = *stride;
    const int64_t strideD = (strideRef.Size() == 0) ? kernelD : strideRef[0];
    const int64_t strideH = (strideRef.Size() == 0) ? kernelH : ((strideRef.Size() == 1) ? strideD : strideRef[1]);
    const int64_t strideW = (strideRef.Size() == 0) ? kernelW : ((strideRef.Size() == 1) ? strideD : strideRef[2]);

    const aclIntArray& paddingRef = *padding;
    const int64_t paddingD = paddingRef[0];
    const int64_t paddingH = (paddingRef.Size() == 1) ? paddingD : paddingRef[1];
    const int64_t paddingW = (paddingRef.Size() == 1) ? paddingD : paddingRef[2];

    const aclIntArray& dilationRef = *dilation;
    const int64_t dilationD = dilationRef[0];
    const int64_t dilationH = (dilationRef.Size() == 1) ? dilationD : dilationRef[1];
    const int64_t dilationW = (dilationRef.Size() == 1) ? dilationD : dilationRef[2];

    FVector<int64_t> kernelSizeData{kernelD, kernelH, kernelW};
    FVector<int64_t> strideSizeData{strideD, strideH, strideW};
    FVector<int64_t> paddingSizeData{paddingD, paddingH, paddingW};
    FVector<int64_t> dilationSizeData{dilationD, dilationH, dilationW};

    aclIntArray* kernelSize3 = executor->AllocIntArray(kernelSizeData.data(), DHW_DIMS);
    aclIntArray* stride3 = executor->AllocIntArray(strideSizeData.data(), DHW_DIMS);
    aclIntArray* padding3 = executor->AllocIntArray(paddingSizeData.data(), DHW_DIMS);
    aclIntArray* dilation3 = executor->AllocIntArray(dilationSizeData.data(), DHW_DIMS);

    op::DataType outType = self->GetDataType();
    auto gradInput = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_NCDHW);

    if (IsAiCoreSupport(gradOutput)) {
        // 调用算子计算
        return MaxPool3DGradWithArgmaxAiCore(gradOutput, self, indices, kernelSize3, stride3, padding3, dilation3,
                                             ceilMode, gradInput, executor);
    } else {
        // 当前没有匹配的aicpu算子
    	  OP_LOGE(ACLNN_ERR_PARAM_INVALID, "no dtype support on AICPU");
        return nullptr;
    }
}
} // l0op
