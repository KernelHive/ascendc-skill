/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 #include "convolution_l0.h"
 #include "opdev/make_op_executor.h"
 #include "opdev/op_def.h"
 #include "opdev/op_dfx.h"
 #include "opdev/op_executor.h"
 #include "opdev/op_log.h"
 #include "opdev/shape_utils.h"
 
 using namespace op;
 
 namespace l0op {
 OP_TYPE_REGISTER(Conv3DV2);
 
 static const int64_t conv3dDimNum = 5;
 const int64_t PAD_DIM_6 = 6;
 const int64_t DIM_0 = 0;
 const int64_t DIM_1 = 1;
 const int64_t DIM_2 = 2;
 const int64_t C0_DIM_NDC1HWC0_INDEX = 5;
 const int64_t C0_BF16 = 16;
 const int64_t H_DIM_NCDHW_INDEX = 3;
 const int64_t W_DIM_NCDHW_INDEX = 4;
 
 
 static aclnnStatus Conv3dv2WithFlag(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, 
                                     const aclTensor *scale, const aclTensor *offset, const aclIntArray *stride,
                                     const aclIntArray *padding, const aclIntArray *dilation,
                                     int groups, bool useHf32, aclTensor *&output, aclOpExecutor *executor)
 {
     L0_DFX(Conv3dv2WithFlag, input, weight, bias, scale, offset, stride, padding, dilation, groups, useHf32);
 
     aclIntArray *stride5;
     aclIntArray *dilation5;
 
     FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1], (*stride)[2]};
     FVector<int64_t> newDalition{1, 1, (*dilation)[0], (*dilation)[1], (*dilation)[2]};
     stride5 = executor->AllocIntArray(newStrides.data(), conv3dDimNum);
     dilation5 = executor->AllocIntArray(newDalition.data(), conv3dDimNum);
 
     FVector<int64_t> newPad{(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1], (*padding)[2], (*padding)[2]};
 
     auto pad6 = executor->AllocIntArray(newPad.data(), PAD_DIM_6);
 
     ge::AscendString originalFormat = op::ToString(input->GetOriginalFormat());
     const char *dataFormat = originalFormat.GetString();
     if (bias) {
       auto ret = INFER_SHAPE(Conv3DV2, OP_INPUT(input, weight, bias), OP_OUTPUT(output),
                              OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0, useHf32));
       if (ret != ACLNN_SUCCESS) {
         OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
         output = nullptr;
         return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
       }
       if (input->GetDataType() != output->GetDataType()) {
         auto storageShape = input->GetStorageShape();  // storageShape dim of output is same as input
         auto viewShape = output->GetViewShape();
         storageShape[DIM_0] = viewShape[DIM_0];
         storageShape[DIM_1] = viewShape[DIM_2];
         storageShape[DIM_2] = (viewShape[DIM_1] + C0_BF16 - 1) / C0_BF16;
         storageShape[H_DIM_NCDHW_INDEX] = viewShape[H_DIM_NCDHW_INDEX];
         storageShape[W_DIM_NCDHW_INDEX] = viewShape[W_DIM_NCDHW_INDEX];
         storageShape[C0_DIM_NDC1HWC0_INDEX] = C0_BF16;
         output->SetStorageShape(storageShape);
       }
       ADD_TO_LAUNCHER_LIST_AICORE(
         Conv3DV2, OP_INPUT(input, weight, bias, scale, offset, nullptr), OP_OUTPUT(output),
         OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0, useHf32));
     } else {
       auto ret = INFER_SHAPE(Conv3DV2, OP_INPUT(input, weight), OP_OUTPUT(output),
                              OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0, useHf32));
       if (ret != ACLNN_SUCCESS) {
         OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
         output = nullptr;
         return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
       }
       ADD_TO_LAUNCHER_LIST_AICORE(
         Conv3DV2, OP_INPUT(input, weight, nullptr, scale, offset, nullptr), OP_OUTPUT(output),
         OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0, useHf32));
     }
     return ACLNN_SUCCESS;
 }
 
 const aclTensor *Conv3dv26HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                  const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                  int groups, aclOpExecutor *executor)
 {
     L0_DFX(Conv3dv26HdBf16, input, weight, bias, stride, padding, dilation, groups);
     auto output =
         executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), input->GetOriginalFormat());
     Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, false, output, executor);
     return output;
 }
 
 const aclTensor *Conv3dv26HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, bool useHf32, aclOpExecutor *executor)
 {
     L0_DFX(Conv3dv26HdFp32, input, weight, bias, stride, padding, dilation, groups, useHf32);
     auto output =
         executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), input->GetOriginalFormat());
     Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, useHf32, output, executor);
     return output;
 }
 
 const aclTensor *Conv3dv26HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                  const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                  int groups, aclOpExecutor *executor)
 {
     L0_DFX(Conv3dv26HdFp16, input, weight, bias, stride, padding, dilation, groups);
     auto output =
         executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), input->GetOriginalFormat());
     Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, false, output, executor);
     return output;
 }
 
 const aclTensor *Conv3dv2NCDHWFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, bool useHf32, aclOpExecutor *executor)
 {
     L0_DFX(Conv3dv2NCDHWFp32, input, weight, bias, stride, padding, dilation, groups, useHf32);
     auto output =
         executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), input->GetOriginalFormat());
     Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, useHf32, output, executor);
     return output;
 }
 
 const aclTensor *Conv3dv2NCDHWBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, aclOpExecutor *executor)
 {
     L0_DFX(Conv3dv2NCDHWBf16, input, weight, bias, stride, padding, dilation, groups);
     auto output =
         executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), input->GetOriginalFormat());
     Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, false, output, executor);
     return output;
 }
 
 const aclTensor *Conv3dv2NCDHWFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                int groups, aclOpExecutor *executor)
 {
     L0_DFX(Conv3dv2NCDHWFp16, input, weight, bias, stride, padding, dilation, groups);
     auto output =
         executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), input->GetOriginalFormat());
     Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, false, output, executor);
     return output;
 }
 
 }  // namespace l0op
 