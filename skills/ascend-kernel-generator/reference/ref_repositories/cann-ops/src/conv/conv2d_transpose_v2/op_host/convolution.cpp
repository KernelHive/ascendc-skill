/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "convolution.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Conv2D);
OP_TYPE_REGISTER(Conv2DTranspose);
OP_TYPE_REGISTER(Conv2DTransposeV2);
OP_TYPE_REGISTER(DepthwiseConv2D);
OP_TYPE_REGISTER(Conv3DTranspose);
OP_TYPE_REGISTER(Conv3DTransposeV2);
OP_TYPE_REGISTER(Conv3D);
OP_TYPE_REGISTER(Conv3DV2);

static const int64_t conv2dDimNum = 4;
static const int64_t conv3dDimNum = 5;
const int64_t PAD_DIM_2 = 2;
const int64_t PAD_DIM_4 = 4;
const int64_t PAD_DIM_6 = 6;
const int64_t PAD_TOP_INDEX = 0;
const int64_t PAD_BOTTOM_INDEX = 1;
const int64_t PAD_LEFT_INDEX = 2;
const int64_t PAD_RIGHT_INDEX = 3;
const int64_t PAD_H_INDEX = 0;
const int64_t PAD_W_INDEX = 1;
const int64_t DIM_0 = 0;
const int64_t DIM_1 = 1;
const int64_t DIM_2 = 2;
const int64_t C0_DIM_NDC1HWC0_INDEX = 5;
const int64_t C0_BF16 = 16;
const size_t CONV3D_TRANSPOSE_V2_WHITE_LIST_CASE_SIZE = 29;
const size_t CONV3D_V2_WHITE_LIST_CASE_SIZE = 22;
const int64_t D_DIM_NCDHW_INDEX = 2;
const int64_t H_DIM_NCDHW_INDEX = 3;
const int64_t W_DIM_NCDHW_INDEX = 4;

struct Conv3DTransPoseV2Prarams {
  const aclTensor *input;
  const aclTensor *weight;
  const aclTensor *output;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  const aclIntArray *outputPadding;
  int groups;
};

const vector<vector<int64_t>> CONV2D_TRANSPOSE_V2_WHITE_LIST =
{
  {
    DataType::DT_FLOAT, // input data type
    4, 320, 80, 80,     // input shape
    320, 320, 3, 3,     // filter shape
    4, 320, 80, 80,     // outBackprop shape
    1, 1,               // stide
    1, 1,               // padding
    1, 1,               // dilation
    0, 0,               // output padding
    1,                  // groups
  },
};

const vector<vector<int64_t>> CONV3D_TRANSPOSE_V2_WHITE_LIST =
{
  {
    DataType::DT_BF16,     // input data type
    1, 256, 62, 66, 66,    // input shape
    256, 256, 4, 4, 4,     // filter shape
    1, 256, 120, 128, 128, // outBackprop shape
    2, 2, 2,               // stide
    3, 3, 3,               // padding
    1, 1, 1,               // dilation
    0, 0, 0,               // output padding
    1,                     // groups
  },
  {
    DataType::DT_BF16,
    1, 256, 122, 130, 130,
    256, 3, 4, 4, 4,
    1, 3, 240, 256, 256,
    2, 2, 2,
    3, 3, 3,
    1, 1, 1,
    0, 0, 0,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 6, 62, 62,
    256, 256, 4, 4, 4,
    1, 256, 8, 120, 120,
    2, 2, 2,
    3, 3, 3,
    1, 1, 1,
    0, 0, 0,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 10, 122, 122,
    256, 3, 4, 4, 4,
    1, 3, 16, 240, 240,
    2, 2, 2,
    3, 3, 3,
    1, 1, 1,
    0, 0, 0,
    1,
  },
};

static void AddAclIntArrayToCaseInfo(const aclIntArray &seg, vector<int64_t> &caseInfo)
{
  size_t len = seg.Size();
  for (size_t i = 0; i < len; i++) {
    caseInfo.push_back(seg[i]);
  }
}

static void AddTensorShapeToCaseInfo(const aclTensor &seg, vector<int64_t> &caseInfo)
{
  auto segShape = seg.GetViewShape();
  size_t dimNum = segShape.GetDimNum();
  for (size_t i=0; i < dimNum; i++) {
    caseInfo.push_back(segShape.GetDim(i));
  }
}

static void ConstructCaseInfo(const Conv3DTransPoseV2Prarams &params, vector<int64_t> &caseInfo)
{
  caseInfo.reserve(CONV3D_TRANSPOSE_V2_WHITE_LIST_CASE_SIZE);
  auto inputDataType = params.input->GetDataType();
  caseInfo.push_back(static_cast<int64_t>(inputDataType));
  AddTensorShapeToCaseInfo(*(params.input), caseInfo);
  AddTensorShapeToCaseInfo(*(params.weight), caseInfo);
  AddTensorShapeToCaseInfo(*(params.output), caseInfo);
  AddAclIntArrayToCaseInfo(*(params.stride), caseInfo);
  AddAclIntArrayToCaseInfo(*(params.padding), caseInfo);
  AddAclIntArrayToCaseInfo(*(params.dilation), caseInfo);
  AddAclIntArrayToCaseInfo(*(params.outputPadding), caseInfo);
  caseInfo.push_back(params.groups);
}

static bool IsConv3DTransposeV2WhiteListCase(const vector<int64_t> &caseInfo, const vector<vector<int64_t>> &whiteList)
{
  if (caseInfo[0] != DataType::DT_BF16 && caseInfo[0] != DataType::DT_FLOAT16) {
    return false;
  }
  vector<int64_t> caseInfoShape(caseInfo);
  caseInfoShape.erase(caseInfoShape.begin());
  for (auto itCase = whiteList.begin(); itCase != whiteList.end(); ++itCase) {
    vector<int64_t> itShape(*itCase);
    itShape.erase(itShape.begin());
    if (caseInfoShape == itShape) {
      return true;
    }
  }
  return false;
}

static bool IsConv2DTransposeV2WhiteListCase(const vector<int64_t> &caseInfo, const vector<vector<int64_t>> &whiteList)
{
  for (auto it = whiteList.begin(); it != whiteList.end(); ++it) {
    if (*it == caseInfo) {
      return true;
    }
  }
  return false;
}

static bool CheckV2Dilation(const Conv3DTransPoseV2Prarams &params)
{
  const aclIntArray &dilation = *(params.dilation);
  for (size_t i = 0; i < dilation.Size(); i++) {
    if (dilation[i] > 1) {
      return true;
    }
  }
  return false;
}

static bool CheckV2Stride(const Conv3DTransPoseV2Prarams &params)
{
  const aclIntArray &stride = *(params.stride);
  for (size_t i = 0; i < stride.Size(); i++) {
    if (stride[i] > 2) { // 2: stride threshold for v2
      OP_LOGD("Conv3d transpose v2 not support stride > 2");
      return false;
    }
  }
  return true;
}

static bool CheckV2Pad(const Conv3DTransPoseV2Prarams &params)
{
  const aclIntArray &padding = *(params.padding);
  for (size_t i = 0; i < padding.Size(); i++) {
    if (padding[i] > 2) { // 2: padding threshold for v2
      OP_LOGD("Conv3d transpose v2 not support pad > 2");
      return false;
    }
  }
  return true;
}

static bool CheckV2OutputPad(const Conv3DTransPoseV2Prarams &params)
{
  const aclIntArray &outputPadding = *(params.outputPadding);
  for (size_t i = 0; i < outputPadding.Size(); i++) {
    if ((outputPadding[i] > 0) && 
        !(params.input->GetDataType() == DataType::DT_FLOAT16 && 
         params.weight->GetDataType() == DataType::DT_FLOAT16 &&
         params.output->GetDataType() == DataType::DT_FLOAT16) && 
        !(params.input->GetDataType() == DataType::DT_BF16 && 
         params.weight->GetDataType() == DataType::DT_BF16 &&
         params.output->GetDataType() == DataType::DT_BF16)) {
      OP_LOGD("Conv3d transpose v2 support outputPadding > 0 only if datatype is fp16/bf16");
      return false;
    }
  }
  return true;
}

static bool CheckV2Kernel(const Conv3DTransPoseV2Prarams &params)
{
  const aclTensor &kernel = *(params.weight);
  auto kernelShape = kernel.GetViewShape();
  for (size_t i = D_DIM_NCDHW_INDEX; i < kernelShape.GetDimNum(); i++) {
    if (kernelShape.GetDim(i) > 4) { // 4: kernel threshold for v2
      OP_LOGD("Conv3d transpose v2 not support kernel > 4");
      return false;
    }
  }
  return true;
}

static bool CheckV2Functionality(const Conv3DTransPoseV2Prarams &params)
{
  const aclIntArray &stride = *(params.stride);
  const aclIntArray &dilation = *(params.dilation);
  auto kernelShape = params.weight->GetViewShape();
  if (stride[0] > kernelShape[D_DIM_NCDHW_INDEX]) {
    OP_LOGD("Conv3d transpose v2 not support stride_d > kernel_d");
    return false;
  }

  const aclIntArray &padding = *(params.padding);
  for (size_t i = 0; i < padding.Size(); i++) {
    if (((i + D_DIM_NCDHW_INDEX) >= kernelShape.GetDimNum()) || (padding[i] >= kernelShape[i + D_DIM_NCDHW_INDEX])) {
      OP_LOGD("Conv3d transpose v2 not support pad >= kernel");
      return false;
    }
  }

  auto outputShape = params.output->GetViewShape();
  if ((outputShape[D_DIM_NCDHW_INDEX] + padding[0] + padding[0] - ((kernelShape[D_DIM_NCDHW_INDEX] - 1) * dilation[0] + 1))
      % stride[0] > padding[0]) {
    OP_LOGD("Conv3d transpose v2 not support if the condition is not meet:");
    OP_LOGD("mod((d_in + pad_h + pad_t - ((k_d - 1) * dilation_d + 1)), stride_d) <= pad_t");
    return false;
  }
  
  OP_CHECK(
    (outputShape[H_DIM_NCDHW_INDEX] + padding[DIM_1] + padding[DIM_1] - ((kernelShape[H_DIM_NCDHW_INDEX] - 1) * dilation[DIM_1] + 1))
     % stride[DIM_1] <= padding[DIM_1],
    OP_LOGD("Conv3dTransposeV2 not support: mod((h_o + pad_up + pad_down - ((k_h - 1) * dilation_h + 1)), stride_h) > pad_down"),
    return false
  );

  return true;
}

// 判断dout能否分核，能分核返回true,不能分核返回false
static bool CheckV2DoutCoreDim(int64_t dout)
{
  auto coreNum = static_cast<int32_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
  vector<int32_t> coreFactors = {};
  for (int32_t factor = 2; factor <= coreNum; factor++) { // 2: min factor
    if (coreNum % factor == 0) {
      coreFactors.push_back(factor);
    }
  }
  for (auto factor : coreFactors) {
    if (dout % factor == 0) {
      return true;
    }
  }
  return false;
}

static bool IsConv3DTransposeUseV2(const Conv3DTransPoseV2Prarams &params)
{
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();

  if (params.input->GetOriginalFormat() != op::Format::FORMAT_NCDHW) {
    OP_LOGD("Conv3d transpose v2 not support format except FORMAT_NCDHW");
    return false;
  }

  // dilation > 1, 走V2
  if (CheckV2Dilation(params)) {
    return true;
  }

  // 网络泛化case规格，测试结果 V2平均性能好于TBE
  if (!CheckV2Stride(params) || !CheckV2Pad(params) || !CheckV2Kernel(params) || !CheckV2OutputPad(params)) {
    return false;
  }

  // 以下3个检查为V2功能上不支持的规格，走TBE
  if (!CheckV2Functionality(params)) {
    return false;
  }

  // 参考CUBE算子规格梳理
  if (socVersion == SocVersion::ASCEND910B) {
    auto inputShape = params.input->GetViewShape();
    auto kernelShape = params.weight->GetViewShape();
    const aclIntArray &stride = *(params.stride);
    const aclIntArray &dilation = *(params.dilation);
    // fmapH = min((2+(kernel_h-1)*dilation_h), ((Hi-1)*stride_h+1))
    int64_t fmapH = (inputShape[H_DIM_NCDHW_INDEX] - 1) * stride[1] + 1;
    // 2为待装载数据头尾存在跨行的情况
    fmapH = min(fmapH, 2 + (kernelShape[H_DIM_NCDHW_INDEX] - 1) * dilation[DIM_1]);
    int64_t l1SizeLimit = fmapH * inputShape[W_DIM_NCDHW_INDEX] * stride[DIM_2];
    if (l1SizeLimit > 4096) { // 4096: l1SizeLimit
      OP_LOGD("Conv3d transpose v2 not support because L1 size limit is not meet");
      return false;
    }
  }

  // 对网络泛化case中，V1性能劣化的场景进行拦截(dout不能分核，分核集中在M方向，M又比较小，导致用不满核或者拖尾严重)
  // 从劣化的shape中分析出来的条件为: dout不能分核 && (m<5500 || dout>85)
  auto outputShape = params.output->GetViewShape();
  int64_t m = outputShape[H_DIM_NCDHW_INDEX] * outputShape[W_DIM_NCDHW_INDEX];
  const int64_t M_THRESHOLD = 5500;
  const int64_t DOUT_THRESHOLD = 85;
  int64_t dout = outputShape[D_DIM_NCDHW_INDEX];
  if (!CheckV2DoutCoreDim(dout) && (m < M_THRESHOLD || dout > DOUT_THRESHOLD)) {
    OP_LOGD("Conv3d transpose v2 not support because shapes are not suitable");
    return false;
  }
  return true;
}

bool IsSupportConv3DToConv3DV2()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    return socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;
}

aclIntArray* ConstructNewPad(const aclIntArray *padding, aclOpExecutor *executor)
{
    FVector<int64_t> newPad;
    if (padding->Size() == PAD_DIM_4) {
        newPad = {(*padding)[PAD_TOP_INDEX], (*padding)[PAD_BOTTOM_INDEX],
                  (*padding)[PAD_LEFT_INDEX], (*padding)[PAD_RIGHT_INDEX]};
    } else {
        newPad = {(*padding)[PAD_H_INDEX], (*padding)[PAD_H_INDEX],
                  (*padding)[PAD_W_INDEX], (*padding)[PAD_W_INDEX]};
    }
    return executor->AllocIntArray(newPad.data(), conv2dDimNum);
}

static aclnnStatus Conv2dWithFlag(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                  const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                                  int groups, bool useHf32, aclTensor *&output, aclOpExecutor *executor)
{
    L0_DFX(Conv2dWithFlag, input, weight, stride, padding, dilation, groups, useHf32);

    aclIntArray *stride4;
    aclIntArray *dilation4;
    const char *empty_str = "";
    const int64_t hf32 = useHf32 ? 0x40 : 0x20;
    if (input->GetOriginalFormat() == op::Format::FORMAT_NCHW) {
        FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1]};
        FVector<int64_t> newDalition{1, 1, (*dilation)[0], (*dilation)[1]};
        stride4 = executor->AllocIntArray(newStrides.data(), conv2dDimNum);
        dilation4 = executor->AllocIntArray(newDalition.data(), conv2dDimNum);
    } else {
        FVector<int64_t> newStrides{1, (*stride)[0], (*stride)[1], 1};
        FVector<int64_t> newDalition{1, (*dilation)[0], (*dilation)[1], 1};
        stride4 = executor->AllocIntArray(newStrides.data(), conv2dDimNum);
        dilation4 = executor->AllocIntArray(newDalition.data(), conv2dDimNum);
    }
    auto pad4 = ConstructNewPad(padding, executor);
    ge::AscendString originalFormat = op::ToString(input->GetOriginalFormat());
    const char *dataFormat = originalFormat.GetString();
    OP_LOGD("new pad: [%ld] [%ld] [%ld] [%ld]", (*pad4)[PAD_TOP_INDEX], (*pad4)[PAD_BOTTOM_INDEX],
      (*pad4)[PAD_LEFT_INDEX], (*pad4)[PAD_RIGHT_INDEX]);
    auto inferShapeRet = ACLNN_SUCCESS;
    auto launchRet = ACLNN_SUCCESS;
    if (bias) {
      inferShapeRet = INFER_SHAPE(Conv2D, OP_INPUT(input, weight, bias), OP_OUTPUT(output),
        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, 0, empty_str, empty_str, hf32));
      launchRet = ADD_TO_LAUNCHER_LIST_AICORE(
        Conv2D, OP_INPUT(input, weight, bias), OP_OUTPUT(output),
        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, 0, empty_str, empty_str, hf32));
    } else {
      inferShapeRet = INFER_SHAPE(Conv2D, OP_INPUT(input, weight), OP_OUTPUT(output),
        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, 0, empty_str, empty_str, hf32));
      launchRet = ADD_TO_LAUNCHER_LIST_AICORE(
        Conv2D, OP_INPUT(input, weight), OP_OUTPUT(output),
        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, 0, empty_str, empty_str, hf32));
    }
    if (inferShapeRet != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
      output = nullptr;
      return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
    }
    if (launchRet != ACLNN_SUCCESS) {
      OP_LOGE(ACLNN_ERR_INNER_FIND_KERNEL_ERROR, "Conv2D ADD_TO_LAUNCHER_LIST_AICORE failed.");
      output = nullptr;
      return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    return ACLNN_SUCCESS;
}
const aclTensor *Conv2d5HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor)
{
    L0_DFX(Conv2d5HdFp16, input, weight, stride, padding, dilation, groups);
    auto convOut =
        executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), input->GetOriginalFormat());
    Conv2dWithFlag(input, weight, bias, stride, padding, dilation, groups, false, convOut, executor);
    return convOut;
}

// 仅Ascend910芯片支持 fp16进fp32出
const aclTensor *Conv2d5HdFp1625HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                       const aclIntArray *stride, const aclIntArray *padding,
                                       const aclIntArray *dilation, int groups, aclOpExecutor *executor)
{
    L0_DFX(Conv2d5HdFp1625HdFp32, input, weight, stride, padding, dilation, groups);
    auto convOut =
        executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), input->GetOriginalFormat());
    Conv2dWithFlag(input, weight, bias, stride, padding, dilation, groups, false, convOut, executor);
    return convOut;
}

const aclTensor *Conv2d5HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor)
{
    L0_DFX(Conv2d5HdFp32, input, weight, bias, stride, padding, dilation, groups, useHf32);
    auto convOut =
        executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), input->GetOriginalFormat());
    Conv2dWithFlag(input, weight, bias, stride, padding, dilation, groups, useHf32, convOut, executor);
    return convOut;
}

// 仅Ascend910B芯片支持 bf16进bf16
const aclTensor *Conv2d5HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor)
{
    L0_DFX(Conv2d5HdBf16, input, weight, bias, stride, padding, dilation, groups);
    auto convOut =
        executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), input->GetOriginalFormat());
    Conv2dWithFlag(input, weight, bias, stride, padding, dilation, groups, false, convOut, executor);
    return convOut;
}

static aclnnStatus Conv3dWithFlag(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                           const aclIntArray *stride, const aclIntArray *padding,
                                           const aclIntArray *dilation, int groups,
                                           bool useHf32, aclTensor *&output, aclOpExecutor *executor)
{
    L0_DFX(Conv3dWithFlag, input, weight, bias, stride, padding, dilation, groups, useHf32);

    aclIntArray *stride5;
    aclIntArray *dilation5;
    if (input->GetOriginalFormat() == op::Format::FORMAT_NCDHW) {
        FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1], (*stride)[2]};
        FVector<int64_t> newDalition{1, 1, (*dilation)[0], (*dilation)[1], (*dilation)[2]};
        stride5 = executor->AllocIntArray(newStrides.data(), conv3dDimNum);
        dilation5 = executor->AllocIntArray(newDalition.data(), conv3dDimNum);
    } else {
        FVector<int64_t> newStrides{1, (*stride)[0], (*stride)[1], (*stride)[2], 1};
        FVector<int64_t> newDalition{1, (*dilation)[0], (*dilation)[1], (*dilation)[2], 1};
        stride5 = executor->AllocIntArray(newStrides.data(), conv3dDimNum);
        dilation5 = executor->AllocIntArray(newDalition.data(), conv3dDimNum);
    }

    FVector<int64_t> newPad{(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1], (*padding)[2], (*padding)[2]};

    auto pad6 = executor->AllocIntArray(newPad.data(), PAD_DIM_6);

    ge::AscendString originalFormat = op::ToString(input->GetOriginalFormat());
    const char *dataFormat = originalFormat.GetString();

    if (bias) {
      auto ret = INFER_SHAPE(Conv3D, OP_INPUT(input, weight, bias), OP_OUTPUT(output),
                             OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0));
      if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        output = nullptr;
        return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
      }
      ADD_TO_LAUNCHER_LIST_AICORE(
          Conv3D, OP_INPUT(input, weight, bias), OP_OUTPUT(output),
          OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0));
    } else {
      auto ret = INFER_SHAPE(Conv3D, OP_INPUT(input, weight), OP_OUTPUT(output),
                             OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0));
      if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        output = nullptr;
        return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
      }
      ADD_TO_LAUNCHER_LIST_AICORE(
          Conv3D, OP_INPUT(input, weight), OP_OUTPUT(output),
          OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, 0));
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus Conv3dv2WithFlag(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, 
                                    const aclTensor *scale, const aclTensor *offset, const aclIntArray *stride,
                                    const aclIntArray *padding, const aclIntArray *dilation,
                                    int groups, bool useHf32, aclTensor *&output, aclOpExecutor *executor)
{
    L0_DFX(Conv3dv2WithFlag, input, weight, bias, scale, offset, stride, padding, dilation, groups, useHf32);

   if (!IsSupportConv3DToConv3DV2()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3dv2 only support Ascend910B/910_93");
        output = nullptr;
        return ACLNN_ERR_PARAM_INVALID;
    }

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

const aclTensor *Conv3d6HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor)
{
    L0_DFX(Conv3d6HdFp16, input, weight, bias, stride, padding, dilation, groups);
    auto output =
        executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), input->GetOriginalFormat());
    if (!IsSupportConv3DToConv3DV2()) {
        Conv3dWithFlag(input, weight, bias, stride, padding, dilation, groups, false, output, executor);
    } else {
        OP_LOGD("Conv3dFp16 to Conv3dv2Fp16!");
        Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, false, output, executor);
    }
    return output;
}

const aclTensor *Conv3d6HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor)
{
    L0_DFX(Conv3d6HdFp32, input, weight, bias, stride, padding, dilation, groups, useHf32);
    auto output =
        executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), input->GetOriginalFormat());
    if (!IsSupportConv3DToConv3DV2()) {
        Conv3dWithFlag(input, weight, bias, stride, padding, dilation, groups, useHf32, output, executor);
    } else {
        OP_LOGD("Conv3dFp32 to Conv3dv2Fp32!");
        Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, useHf32, output, executor);
    }
    return output;
}

const aclTensor *Conv3d6HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor)
{
    L0_DFX(Conv3d6HdBf16, input, weight, bias, stride, padding, dilation, groups);
    auto output =
        executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), input->GetOriginalFormat());
    if (!IsSupportConv3DToConv3DV2()) {
        Conv3dWithFlag(input, weight, bias, stride, padding, dilation, groups, false, output, executor);
    } else {
        OP_LOGD("Conv3dBf16 to Conv3dv2Bf16!");
        Conv3dv2WithFlag(input, weight, bias, nullptr, nullptr, stride, padding, dilation, groups, false, output, executor);
    }
    return output;
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

const aclTensor *QuantConv3d6HdInt8(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, 
                                    const aclTensor *scale, const aclTensor *offset, const aclIntArray *stride, 
                                    const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                    aclOpExecutor *executor)
{
    L0_DFX(QuantConv3d6HdInt8, input, weight, bias, scale, offset, stride, padding, dilation, groups);
    auto output =
        executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), input->GetOriginalFormat());
    Conv3dv2WithFlag(input, weight, bias, scale, offset, stride, padding, dilation, groups, 0, output, executor);
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

// 根据shape生成对应shape的Input tensor
static aclTensor* InitializeTensor(op::FVector<int64_t, op::MAX_DIM_NUM> shape, aclOpExecutor* executor) {
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
    auto inputTensor = executor->ConvertToTensor(shapeArray, DataType::DT_INT32);
    op::Format inputFormat= shape.size() == conv3dDimNum ? Format::FORMAT_NCDHW : Format::FORMAT_NCHW;
    const_cast<aclTensor *>(inputTensor)->SetStorageFormat(inputFormat);
    const_cast<aclTensor *>(inputTensor)->SetViewFormat(inputFormat);
    const_cast<aclTensor *>(inputTensor)->SetOriginalFormat(inputFormat);
    return const_cast<aclTensor *>(inputTensor);
}

static aclnnStatus ConvTranspose2dWithFlag(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                           const aclIntArray *stride, const aclIntArray *padding,
                                           const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                           bool useHf32, aclTensor *&output, aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose2dWithFlag, input, weight, bias, stride, padding, dilation, groups, outputPadding, useHf32);
    const char *dataFormat = "NCHW"; // 由于transpose中infershape的逻辑只支持dataformat为‘NCHW’,而不是input的GetOriginalForamt

    aclIntArray *stride4;
    aclIntArray *dilation4;
    aclIntArray *outputPad4;
    const int64_t hf32 = useHf32 ? 0x40 : 0x20;
    if (input->GetOriginalFormat() == op::Format::FORMAT_NCHW) {
        FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1]};
        FVector<int64_t> newDalition{1, 1, (*dilation)[0], (*dilation)[1]};
        FVector<int64_t> newOutputPad{0, 0, (*outputPadding)[0], (*outputPadding)[1]};
        stride4 = executor->AllocIntArray(newStrides.data(), conv2dDimNum);
        dilation4 = executor->AllocIntArray(newDalition.data(), conv2dDimNum);
        outputPad4 = executor->AllocIntArray(newOutputPad.data(), conv2dDimNum);
    } else {
        FVector<int64_t> newStrides{1, (*stride)[0], (*stride)[1], 1};
        FVector<int64_t> newDalition{1, (*dilation)[0], (*dilation)[1], 1};
        FVector<int64_t> newOutputPad{0, (*outputPadding)[0], (*outputPadding)[1], 0};
        stride4 = executor->AllocIntArray(newStrides.data(), conv2dDimNum);
        dilation4 = executor->AllocIntArray(newDalition.data(), conv2dDimNum);
        outputPad4 = executor->AllocIntArray(newOutputPad.data(), conv2dDimNum);
    }

    auto pad4 = ConstructNewPad(padding, executor);
    OP_LOGD("new pad: [%ld] [%ld] [%ld] [%ld]", (*pad4)[PAD_TOP_INDEX], (*pad4)[PAD_BOTTOM_INDEX],
      (*pad4)[PAD_LEFT_INDEX], (*pad4)[PAD_RIGHT_INDEX]);
    FVector<int64_t> output_shape;

    auto output_shape1 = executor->AllocIntArray(output_shape.data(), 0);
    auto weightShape = weight->GetViewShape();

    // cann包中需要第一个私有格式输入为全零才可以走Transpose的infershape逻辑，
    // 不然只走Con2D的infershape导致报错
    auto inputShape = op::Shape{0, 0, 0, 0};
    auto inputShapeVector = op::ToShapeVector(inputShape);
    auto inputSize = InitializeTensor(inputShapeVector, executor);

    const char *padding_p = "";
    const char *auto_pad = "NOTSET";

    auto ret = INFER_SHAPE(
        Conv2DTranspose, OP_INPUT(inputSize, input, weight, bias), OP_OUTPUT(output),
        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, outputPad4, 0, padding_p, auto_pad, output_shape1, hf32));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        output = nullptr;
        return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
    }

    // 将inputSize刷新成预期的out的shape
    auto outputShapeVector = op::ToShapeVector(output->GetViewShape());
    inputSize = InitializeTensor(outputShapeVector, executor);
    vector<int64_t> caseInfo2d;
    Conv3DTransPoseV2Prarams params = {input, weight, output, stride, padding, dilation, outputPadding, groups};
    ConstructCaseInfo(params, caseInfo2d);
    if (IsConv2DTransposeV2WhiteListCase(caseInfo2d, CONV2D_TRANSPOSE_V2_WHITE_LIST) && bias == nullptr) {
        ADD_TO_LAUNCHER_LIST_AICORE(Conv2DTransposeV2, OP_INPUT(inputSize, input, weight, nullptr, nullptr),
            OP_OUTPUT(output),
            OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, outputPad4, 0, padding_p, hf32));
    } else {
        if (bias) {
            ADD_TO_LAUNCHER_LIST_AICORE(Conv2DTranspose, OP_INPUT(inputSize, input, weight, bias), OP_OUTPUT(output),
                                        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, outputPad4, 0, padding_p,
                                                auto_pad, output_shape1, hf32));
        } else {
            ADD_TO_LAUNCHER_LIST_AICORE(Conv2DTranspose, OP_INPUT(inputSize, input, weight), OP_OUTPUT(output),
                                        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, outputPad4, 0, padding_p,
                                                auto_pad, output_shape1, hf32));
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ConvTranspose3dWithFlag(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                           const aclIntArray *stride, const aclIntArray *padding,
                                           const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                           bool useHf32, aclTensor *&output, aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose3dWithFlag, input, weight, bias, stride, padding, dilation, groups, outputPadding, useHf32);
    const char *dataFormat = "NCDHW"; // 由于transpose中infershape的逻辑只支持dataformat为‘NCHW’,而不是input的GetOriginalForamt
    const int64_t hf32 = useHf32 ? 0x40 : 0x00;
    aclIntArray *stride5;
    aclIntArray *dilation5;
    aclIntArray *outputPad5;
    if (input->GetOriginalFormat() == op::Format::FORMAT_NCDHW) {
        FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1], (*stride)[2]};
        FVector<int64_t> newDalition{1, 1, (*dilation)[0], (*dilation)[1], (*dilation)[2]};
        FVector<int64_t> newOutputPad{0, 0, (*outputPadding)[0], (*outputPadding)[1], (*outputPadding)[2]};
        stride5 = executor->AllocIntArray(newStrides.data(), conv3dDimNum);
        dilation5 = executor->AllocIntArray(newDalition.data(), conv3dDimNum);
        outputPad5 = executor->AllocIntArray(newOutputPad.data(), conv3dDimNum);
    } else {
        FVector<int64_t> newStrides{1, (*stride)[0], (*stride)[1], (*stride)[2], 1};
        FVector<int64_t> newDalition{1, (*dilation)[0], (*dilation)[1], (*dilation)[2], 1};
        FVector<int64_t> newOutputPad{0, (*outputPadding)[0], (*outputPadding)[1], (*outputPadding)[2], 0};
        stride5 = executor->AllocIntArray(newStrides.data(), conv3dDimNum);
        dilation5 = executor->AllocIntArray(newDalition.data(), conv3dDimNum);
        outputPad5 = executor->AllocIntArray(newOutputPad.data(), conv3dDimNum);
    }

    FVector<int64_t> newPad{(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1], (*padding)[2], (*padding)[2]};

    auto pad5 = executor->AllocIntArray(newPad.data(), PAD_DIM_6);
    auto weightShape = weight->GetViewShape();
    auto inputShape = op::Shape{0, 0, 0, 0, 0};
    auto inputShapeVector = op::ToShapeVector(inputShape);
    auto inputSize = InitializeTensor(inputShapeVector, executor);

    const char *paddingP = "";

    auto ret = INFER_SHAPE(
        Conv3DTranspose, OP_INPUT(inputSize, input, weight, bias), OP_OUTPUT(output),
        OP_ATTR(stride5, pad5, dilation5, groups, dataFormat, outputPad5, 0, paddingP));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        output = nullptr;
        return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
    }

    // 将inputSize刷新成预期的out的shape
    auto outputShapeVector = op::ToShapeVector(output->GetViewShape());
    inputSize = InitializeTensor(outputShapeVector, executor);
    vector<int64_t> caseInfo;
    Conv3DTransPoseV2Prarams params = {input, weight, output, stride, padding, dilation, outputPadding, groups};
    ConstructCaseInfo(params, caseInfo);
    if (groups > 1 || params.input->GetDataType() == DataType::DT_FLOAT ||
        IsConv3DTransposeV2WhiteListCase(caseInfo, CONV3D_TRANSPOSE_V2_WHITE_LIST) ||
        IsConv3DTransposeUseV2(params)) {
        ADD_TO_LAUNCHER_LIST_AICORE(Conv3DTransposeV2, OP_INPUT(inputSize, input, weight, nullptr, nullptr),
                                    OP_OUTPUT(output), OP_ATTR(stride5, pad5, dilation5, groups,
                                    dataFormat, outputPad5, 0, paddingP, hf32));
    } else {
        ADD_TO_LAUNCHER_LIST_AICORE(Conv3DTranspose, OP_INPUT(inputSize, input, weight), OP_OUTPUT(output),
                                    OP_ATTR(stride5, pad5, dilation5, groups,
                                    dataFormat, outputPad5, 0, paddingP));
    }

    return ACLNN_SUCCESS;
}

const aclTensor *ConvTranspose2d5HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose2d5HdFp16, input, weight, stride, padding, dilation, groups, outputPadding);
    auto output =
        executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), input->GetOriginalFormat());
    ConvTranspose2dWithFlag(input, weight, bias, stride, padding, dilation, groups, outputPadding, false, output,
                            executor);
    return output;
}

const aclTensor *ConvTranspose2d5HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        bool useHf32, aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose2d5HdFp32, input, weight, stride, padding, dilation, groups, outputPadding, useHf32);
    auto output = executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), input->GetOriginalFormat());
    ConvTranspose2dWithFlag(input, weight, bias, stride, padding, dilation, groups, outputPadding, useHf32, output,
                            executor);
    return output;
}

const aclTensor *ConvTranspose2d5HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose2d5HdBf16, input, weight, stride, padding, dilation, groups, outputPadding);
    auto output =
        executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), input->GetOriginalFormat());
    ConvTranspose2dWithFlag(input, weight, bias, stride, padding, dilation, groups, outputPadding, false, output,
                            executor);
    return output;
}

const aclTensor *ConvTranspose3d6HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose3d6HdFp16, input, weight, stride, padding, dilation, groups, outputPadding);
    auto output =
        executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), input->GetOriginalFormat());
    ConvTranspose3dWithFlag(input, weight, bias, stride, padding, dilation, groups, outputPadding, false, output,
                            executor);
    return output;
}

const aclTensor *ConvTranspose3d6HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        bool useHf32, aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose3d6HdFp32, input, weight, stride, padding, dilation, groups, outputPadding, useHf32);
    auto output = executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), input->GetOriginalFormat());
    ConvTranspose3dWithFlag(input, weight, bias, stride, padding, dilation, groups, outputPadding, useHf32, output,
                            executor);
    return output;
}

const aclTensor *ConvTranspose3d6HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor)
{
    L0_DFX(ConvTranspose3d6HdBf16, input, weight, stride, padding, dilation, groups, outputPadding);
    auto output =
        executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), input->GetOriginalFormat());
    ConvTranspose3dWithFlag(input, weight, bias, stride, padding, dilation, groups, outputPadding, false, output,
                            executor);
    return output;
}

}  // namespace l0op
