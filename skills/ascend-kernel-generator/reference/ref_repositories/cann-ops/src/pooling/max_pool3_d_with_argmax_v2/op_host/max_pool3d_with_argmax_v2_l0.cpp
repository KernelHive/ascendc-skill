/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "max_pool3d_with_argmax_v2_l0.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MaxPool3DWithArgmaxV2);

static const std::initializer_list<DataType> NULL_SUPPORT_LIST = {};
static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16,
                                                                            DataType::DT_FLOAT};

static const uint32_t C = 1;
static const uint32_t NC = 2;
static const int64_t TWO_SIDED = 2;
static const size_t DHW_DIMS = 3;
static const int64_t NCDHW_DIMS = 5;

static const std::initializer_list<DataType> GetDtypeSupportListBySocVersion() {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        return SELF_DTYPE_SUPPORT_LIST;
    } else {
        return NULL_SUPPORT_LIST;
    }
}

static bool IsMaxPool3DWithArgmaxV2NcdhwAiCoreSupported(const aclTensor* self) {
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    if (!CheckType(self->GetDataType(), dtypeSupportList)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype not supported. support list is %s",
                op::ToString(dtypeSupportList).GetString());
        return false;
    }
    op::Shape input_shape = self->GetViewShape();
    int64_t dims = input_shape.GetDimNum();
    const uint32_t batchesDims = dims == NCDHW_DIMS ? NC : C;
    const int64_t oneChannelDataVolume = input_shape[batchesDims] * input_shape[batchesDims + 1] * input_shape[batchesDims + NC];
    if (oneChannelDataVolume > INT_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "product of depth, height and width can't be greater than INT_MAX");
        return false;
    }
    return true;
}

static const std::tuple<aclTensor*, aclTensor*> MaxPool3DWithArgmaxV2NcdhwAiCore(const aclTensor* self, const aclIntArray* kernelSize, 
                                                                                 const aclIntArray* stride, const aclIntArray* padding, 
                                                                                 const aclIntArray* dilation, bool ceilMode, 
                                                                                 std::string dataFormat, aclTensor* out,
                                                                                 aclTensor* indices, aclOpExecutor* executor) {
    L0_DFX(MaxPool3DWithArgmaxV2NcdhwAiCore, self, kernelSize, stride, padding, dilation, ceilMode, dataFormat, out, indices);

    ADD_TO_LAUNCHER_LIST_AICORE(MaxPool3DWithArgmaxV2, OP_INPUT(self), OP_OUTPUT(out, indices),
                                OP_ATTR(kernelSize, stride, padding, dilation, ceilMode, dataFormat));
    return std::tuple<aclTensor*, aclTensor*>(out, indices);
}

static int64_t RoundedDivision(int64_t x, int64_t y) {
    if (y == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "division by zero");
        return 0;
    }
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        q -= 1;
    };
    return q;
}

static int64_t PoolingOutShape(int64_t inputSize, int64_t kernelSize, int64_t stride, 
                               int64_t padding, int64_t dilation, bool ceilMode) {
    int64_t outputSize = RoundedDivision(inputSize + TWO_SIDED * padding - dilation * (kernelSize - 1) - 1 + (ceilMode ? stride - 1 : 0), stride) + 1;
  
    if (ceilMode) {
        if ((outputSize - 1) * stride >= inputSize + padding) {
            outputSize -= 1;
        }
    }
    return outputSize;
}

static op::Shape GetOutputShape(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
                                const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode) {
    op::Shape input_shape = self->GetViewShape();
    int64_t dims = input_shape.GetDimNum();
    const aclIntArray& kernelRef = *kernelSize;
    const aclIntArray& strideRef = *stride;
    const aclIntArray& paddingRef = *padding;
    const aclIntArray& dilationRef = *dilation;
    op::Shape outputShape;
    const uint32_t batchesDims = dims == NCDHW_DIMS ? NC : C;
    int64_t curDim[DHW_DIMS];
    
    curDim[0] = PoolingOutShape(input_shape.GetDim(batchesDims), kernelRef[0], strideRef[0], paddingRef[0], dilationRef[0], ceilMode);
    curDim[1] = PoolingOutShape(input_shape.GetDim(batchesDims + 1), kernelRef[1], strideRef[1], paddingRef[1], dilationRef[1], ceilMode);
    curDim[NC] = PoolingOutShape(input_shape.GetDim(batchesDims + NC), kernelRef[NC], strideRef[NC], paddingRef[NC], dilationRef[NC], ceilMode);
    
    if (dims == NCDHW_DIMS) {
        outputShape = {input_shape.GetDim(0), input_shape.GetDim(1), curDim[0], curDim[1], curDim[NC]};
    } else {
        outputShape = {input_shape.GetDim(0), curDim[0], curDim[1], curDim[NC]};
    }
    return outputShape;
}

const std::tuple<const aclTensor*, const aclTensor*> MaxPool3DWithArgmaxV2Ncdhw(const aclTensor* self, const aclIntArray* kernelSize, 
                                                                                const aclIntArray* stride, const aclIntArray* padding,
                                                                                const aclIntArray* dilation, bool ceilMode, 
                                                                                std::string dataFormat, aclOpExecutor* executor) {
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

    op::Shape outShape = GetOutputShape(self, kernelSize3, stride3, padding3, dilation3, ceilMode);
    op::DataType outType = self->GetDataType();
    auto out = executor->AllocTensor(outShape, outType, self->GetViewFormat());
    auto indices = executor->AllocTensor(outShape, op::DataType::DT_INT32, self->GetViewFormat());

    if (IsMaxPool3DWithArgmaxV2NcdhwAiCoreSupported(self)) {
        return MaxPool3DWithArgmaxV2NcdhwAiCore(self, kernelSize3, stride3, padding3, dilation3, 
                                                ceilMode, dataFormat, out, indices, executor);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "no dtype support on ai cpu");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
}

}  // namespace l0op