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
 * @file aclnn_convolution_backward.cpp
 */

#include "aclnn_convolution_backward.h"
#include "matmul_util.h"
#include "conv2d_backprop_input_util.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "op_api_def.h"
#include "convolution.h"
#include "convolutionbackward.h"
#include "dilation.h"
#include "fill.h"
#include "reduce_sum_op.h"
#include "squeeze.h"
#include "unsqueeze.h"
#include "zero_op.h"
#include "cube_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "runtime/context.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

struct ConvolutionBackwardOutput {
  aclTensor *gradInput;
  aclTensor *gradWeight;
  aclTensor *gradBias;
};

struct ConvolutionBackwardResult {
  const aclTensor *gradInput;
  const aclTensor *gradWeight;
  const aclTensor *gradBias;
};

struct BatchMatmulInput {
  const aclTensor *leftData;
  const aclTensor *rightData;
  const aclTensor *outputData;
  bool isLeftTranspose;
  bool isRightTranspose;
};

enum class Conv3DBp2MmMode {
  CONV3D_BP_NO_MM = 0,
  CONV3D_BP_MM_1x1_KERNEL = 1,
  CONV3D_BP_MM_STRIDE_EQ_KERNEL = 2,
  CONV3D_BP_MM_FEATURE_MAP_EQ_KERNEL = 3,
};

const int kHDimNC1HWC0Idx = 2;
const int kWDimNC1HWC0Idx = 3;
const int kHDimNCHWIdx = 2;
const int kWDimNCHWIdx = 3;
const int coDimCoCiDHWIdx = 0;
const int ciDimCoCiDHWIdx = 1;
const int nDimNCDHWIdx = 0;
const int cDimNCDHWIdx = 1;
const int dDimNCDHWIdx = 2;
const int hDimNCDHWIdx = 3;
const int wDimNCDHWIdx = 4;
const int kDILATIONHIdx = 0;
const int kDILATIONWIdx = 1;
const int kSTRIDEHIdx = 0;
const int kSTRIDEWIdx = 1;
const int kPADDINGUPIdx = 0;
const int kPADDINGLEFTIdx = 1;
const int kPadding4UpIdx = 0;
const int kPadding4DownIdx = 1;
const int kPadding4LeftIdx = 2;
const int kPadding4RightIdx = 3;
const int kPADDINGDIM = 4;
const int kDILATIONSDIM = 5;
const int kCONV2DDXHWDIM = 2;
const int CONV1DINPUTDIM = 3;
const int CONV2DINPUTDIM = 4;
const int CONV3DINPUTDIM = 5;

static bool IsInputSupportFp32Local() {
  // 判断当前SOC版本是否支持Fp32输入
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
    OP_LOGD("The soc version is Ascend910B or ASCEND910_93, ConvolutionBackward support input tensor with Fp32");
    return true;
  }
  return false;
}


static bool IsInputSupportInsertDilation() {
  // 判断当前SOC版本是否支持前后插Dilation
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
    OP_LOGD("The soc version is Ascend910B or ASCEND910_93, ConvolutionBackward support to insert Dilation");
    return true;
  }
  return false;
}

static bool IsPreInsertDilation(const ConvolutionBackwardInputTensor &inputTensor,
                                const ConvolutionBackwardParams &params) {
  // In order to prevent the workspaceSize from being too large, dilation will not be inserted with special case.
  op::Shape gradOutputSpecialShape = op::Shape({8, 1280, 64, 64});
  op::Shape inputSpecialShape = op::Shape({8, 3, 896, 896});
  op::Shape weightSpecialShape = op::Shape({1280, 3, 14, 14});
  bool isPreInserDiationSpecialFlag = (inputTensor.gradOutput->GetViewShape() == gradOutputSpecialShape &&
                                       inputTensor.input->GetViewShape() == inputSpecialShape &&
                                       inputTensor.weight->GetViewShape() == weightSpecialShape &&
                                       ((*params.stride)[kSTRIDEHIdx] == 14 && (*params.stride)[kSTRIDEWIdx] == 14));
  // When the stride == 2, using set2d yields superior performance, so there is no need for predilation
  return (inputTensor.weight->GetViewShape().GetDim(kHDimNCHWIdx) > 1 ||
          inputTensor.weight->GetViewShape().GetDim(kWDimNCHWIdx) > 1) &&
         ((*params.stride)[kSTRIDEHIdx] > 2 || (*params.stride)[kSTRIDEWIdx] > 2) && !isPreInserDiationSpecialFlag;
}

static bool IsPostInsertDilation(const aclTensor *weight, ConvolutionBackwardParams &params) {
  return (weight->GetViewShape().GetDim(kHDimNCHWIdx) == 1 && weight->GetViewShape().GetDim(kWDimNCHWIdx) == 1) &&
         ((*params.stride)[kSTRIDEHIdx] > 1 || (*params.stride)[kSTRIDEWIdx] > 1);
}

inline static bool CheckTbcNotNull(const aclTensor *self, const aclTensor *input, const aclTensor *weight,
                                   const aclTensor *bias, const aclTensor *gradInput, const aclTensor *gradWeight,
                                   const aclTensor *gradBias) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(gradInput, return false);
    OP_CHECK_NULL(gradWeight, return false);
    OP_CHECK_NULL(gradBias, return false);
    return true;
}

inline static bool CheckNotNull(const ConvolutionBackwardInputTensor &inputTensor,
                                const ConvolutionBackwardOutput &outputTensor, const ConvolutionBackwardParams &params)
{
    OP_CHECK_NULL(inputTensor.gradOutput, return false);
    OP_CHECK_NULL(inputTensor.input, return false);
    OP_CHECK_NULL(inputTensor.weight, return false);
    OP_CHECK_NULL(params.stride, return false);
    OP_CHECK_NULL(params.padding, return false);
    OP_CHECK_NULL(params.dilation, return false);
    OP_CHECK_NULL(params.outputPadding, return false);
    OP_CHECK_NULL(params.outputMask, return false);

    int64_t outputMaskDim = params.outputMask->Size();
    // outputMask的维度必须为3
    OP_CHECK(outputMaskDim == 3, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of outputMask must be equal 3."),
             return false);
    if ((*params.outputMask)[0]) {
        OP_CHECK_NULL(outputTensor.gradInput, return false);
    }

    if ((*params.outputMask)[1]) {
        OP_CHECK_NULL(outputTensor.gradWeight, return false);
    }

    if ((*params.outputMask)[2]) {
        OP_CHECK_NULL(outputTensor.gradBias, return false);
    }

    return true;
}

static bool CheckDtypeValid(const aclTensor *inputTensor, const string &tensorName) {
  // 检查输入aclTensor的数据类型是否在ConvolutionBackward支持列表内
  auto dtypeSupportList = GetDtypeSupportListBySocVersion();
  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor, dtypeSupportList, return false);
  return true;
}

static bool CheckDtypeValidBf16Allowed(const aclTensor *inputTensor, const string &tensorName) {
  // 检查输入aclTensor的数据类型是否在ConvolutionBackward支持列表内
  auto dtypeSupportList = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor, dtypeSupportList, return false);
  return true;
}

static bool CheckFormatValid(const aclTensor *inputTensor, const string &tensorName)
{
    // API在输入Tensor的Format为ND时, 仅支持输入Tensor的维度是3, 并当做NCL格式处理
    op::Format inputFormat = inputTensor->GetStorageFormat();
    auto inputDim = inputTensor->GetViewShape().GetDimNum();
    if (inputDim == CONV1DINPUTDIM) {
        OP_CHECK(inputFormat == op::Format::FORMAT_ND || inputFormat == op::Format::FORMAT_NCL,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In 1D scenes, the %s format only supports ND and NCL.",
                         tensorName.c_str()),
                 return false);
    } else if (inputDim == CONV2DINPUTDIM) {
        OP_CHECK(
            inputFormat == op::Format::FORMAT_NCHW,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In 2D scenes, the %s format only supports NCHW.", tensorName.c_str()),
            return false);
    }

    else if (inputDim == CONV3DINPUTDIM) {
        OP_CHECK(
            inputFormat == op::Format::FORMAT_NCDHW,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In 3D scenes, the %s format only supports NCDHW.", tensorName.c_str()),
            return false);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The %s tensor dimension of this API only supports 3~5 dimensions.",
                tensorName.c_str());
        return false;
    }

    OP_CHECK(!IsPrivateFormat(inputTensor->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s Format only support ND(NCL), NCHW、NCDHW.", tensorName.c_str()),
             return false);
    return true;
}

static bool CheckDeterministic(const int64_t deterministicValue, int groups) {
    OP_CHECK(!((deterministicValue == 1) && (groups > 1)),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilter cannot support groups(%d) > 1 "
                "in deterministic calculations.", groups),
        return false;
    );
    return true;
}

static bool CheckEmptyTensor(ConvolutionBackwardInputTensor inputTensor, ConvolutionBackwardParams params)
{
    // 空tensor场景check
    // 空tensor场景且需要计算gradBias, biasSizes不能为nullptr
    if ((*params.outputMask)[2]) {
        OP_CHECK(params.biasSizes != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The biasSizes cannot be nullptr with empty tensor calculation."),
                 return false);
        OP_CHECK(params.biasSizes->Size() == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The biasSizes size must be 1, actually is %ld.",
                         params.biasSizes->Size()),
                 return false);

        int64_t channelOutDim = 1;  // NCHW
        int64_t Cout = inputTensor.gradOutput->GetViewShape().GetDim(channelOutDim);
        OP_CHECK((*params.biasSizes)[0] == Cout,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The biasSizes should be equal %ld.", Cout), return false);
    }

    auto inputShape = inputTensor.input->GetViewShape();
    // 确保input的shape大于2
    OP_CHECK(inputShape.GetDimNum() > 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input shape must be greater than 2, but now is %ld",
                     inputShape.GetDimNum()),
             return false);

    // 空tensor场景只支持input的N和C维度为0
    if (inputShape[0] == 0 || inputShape[1] == 0) {
        OP_LOGD("The input of ConvolutionBackward is Empty.");
        return true;
    }

    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "ConvolutionBackward only support zero batch or zero channel with input, but got input shape is %s",
            op::ToString(inputShape).GetString());
    return false;
}

static bool CheckTbcFormat(const aclTensor *inputTensor, const string &tensorName)
{
    OP_CHECK(inputTensor->GetStorageFormat() == op::Format::FORMAT_ND ||
                 inputTensor->GetStorageFormat() == op::Format::FORMAT_NCL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s format only support ND or NCL, but got %s.", tensorName.c_str(),
                     op::ToString(inputTensor->GetStorageFormat()).GetString()),
             return false);
    return true;
}

static bool CheckTbcShape(const aclTensor *self, const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                          const int64_t pad, const aclTensor *gradInput, const aclTensor *gradWeight,
                          const aclTensor *gradBias) {
  // input(TBC1) weigth(LC1C0) bias(C0):
  OP_CHECK(input->GetViewShape().GetDim(2) == weight->GetViewShape().GetDim(1),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dim 2 (Input Channels) is not == dim 1 in the weight tensor."),
           return false);
  OP_CHECK(bias->GetViewShape().GetDim(0) == weight->GetViewShape().GetDim(2),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias size must equal dim 2 in the weight tensor (output channel)."),
           return false);
  // pad >= 0
  OP_CHECK(pad >= 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The pad must be greater than or equal to 0."),
           return false);
  // self 与input, weight shape 必须满足约束
  // 约束1：self shape必须与conv_tbc计算出的output match： self(T+2*pad+1-L,B,C0)
  auto t = input->GetViewShape().GetDim(0) + 2 * pad + 1 - weight->GetViewShape().GetDim(0);
  auto b = input->GetViewShape().GetDim(1);
  auto c0 = weight->GetViewShape().GetDim(2);
  OP_CHECK(t >= 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
           "Try to create tensor with negative dimension %ld:[%ld, %ld, %ld]",
           t, t, b, c0),
           return false);
  OP_CHECK(self->GetViewShape().GetDim(0) == t && self->GetViewShape().GetDim(1) == b &&
           self->GetViewShape().GetDim(2) == c0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Mismatch in shape: grad_output has a shape of %s but output has a shape of [%ld, %ld, %ld],"
                   "which output shape is deduced from the input and the weight",
                   op::ToString(self->GetViewShape()).GetString(), t, b, c0),
           return false);

  // input和gradInput的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(input, gradInput, return false);
  // weight和gradWeight的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(weight, gradWeight, return false);
  // bias和gradBias的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(bias, gradBias, return false);
  return true;
}

static inline DataType CalcPromoteType(ConvolutionBackwardInputTensor &inputTensor) {
  auto gradOutputDtype = (inputTensor.gradOutput)->GetDataType();
  auto inputDtype = (inputTensor.input)->GetDataType();
  auto weightDtype = (inputTensor.weight)->GetDataType();

  auto promoteType1 = op::PromoteType(gradOutputDtype, inputDtype);
  auto promoteTypeFinal = op::PromoteType(promoteType1, weightDtype);
  return promoteTypeFinal;
}

static bool SwitchCubeMathType(DataType promoteType, const int8_t cubeMathType) {
  return CheckCubeMathType(promoteType, cubeMathType);
}

static bool CheckCubeMathTypeConvBackward(ConvolutionBackwardInputTensor inputTensor,
                                          ConvolutionBackwardParams params) {
  auto promoteType = CalcPromoteType(inputTensor);
  return SwitchCubeMathType(promoteType, params.cubeMathType);
}

static bool CheckTbcCubeMathType(const aclTensor *self, const aclTensor *input, const aclTensor *weight,
                                 const int8_t cubeMathType) {
  // 计算promote dtype
  auto gradOutputDtype = self->GetDataType();
  auto inputDtype = input->GetDataType();
  auto weightDtype = weight->GetDataType();
  auto promoteType1 = op::PromoteType(gradOutputDtype, inputDtype);
  auto promoteTypeFinal = op::PromoteType(promoteType1, weightDtype);
  // 调用SwitchCubeMathType
  return SwitchCubeMathType(promoteTypeFinal, cubeMathType);
}

static bool ConvBackGoHf32(ConvolutionBackwardInputTensor inputTensor, int8_t cubeMathType) {
  auto promoteType = op::PromoteType(inputTensor.input->GetDataType(), inputTensor.weight->GetDataType());
  promoteType = op::PromoteType(promoteType, inputTensor.gradOutput->GetDataType());
  return NeedCubeGoHF32(promoteType, cubeMathType);
}

// DW V2当前支持白名单case
static bool IsC04WhiteListCase(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params) {
  const int64_t expendedDim = 2;
  for (auto i = 0; i < expendedDim; i++) {
    if ((*params.dilation)[i] != 1) {
      return false;
    }
  }
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  return (socVersion == SocVersion::ASCEND910B &&
          (inputTensor.input->GetDataType() == DataType::DT_FLOAT16 ||
           inputTensor.input->GetDataType() == DataType::DT_BF16) &&
          inputTensor.input->GetViewShape().GetDim(0) == 1024 && inputTensor.input->GetViewShape().GetDim(1) == 3 &&
          inputTensor.input->GetViewShape().GetDim(2) == 224 && inputTensor.input->GetViewShape().GetDim(3) == 224 &&
          inputTensor.gradOutput->GetViewShape().GetDim(0) == 1024 &&
          inputTensor.gradOutput->GetViewShape().GetDim(1) == 1024 &&
          inputTensor.gradOutput->GetViewShape().GetDim(2) == 14 &&
          inputTensor.gradOutput->GetViewShape().GetDim(3) == 14 && params.groups == 1);
}

static aclnnStatus CheckTbcParams(const aclTensor *self, const aclTensor *input, const aclTensor *weight,
                                  const aclTensor *bias, const int64_t pad, const int8_t cubeMathType,
                                  const aclTensor *gradInput, const aclTensor *gradWeight, const aclTensor *gradBias)
{
    // 检查ConvolutionTbcBackward的输入aclTensor是否符合规范
    // 1. 检查输入aclTensor是否为空指针
    CHECK_RET(CheckTbcNotNull(self, input, weight, bias, gradInput, gradWeight, gradBias), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入aclTensor的Dtype是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, "gradOutput"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(input, "input"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(weight, "weight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValidBf16Allowed(bias, "bias"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(gradInput, "gradInput"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(gradWeight, "gradWeight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(gradBias, "gradBias"), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入aclTensor的Format是否在API支持的数据类型范围之内
    CHECK_RET(CheckTbcFormat(self, "Self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(input, "Input"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(weight, "Weight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(gradInput, "GradInput"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(gradWeight, "GradWeight"), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self不能为空
    OP_CHECK(self->Size() != 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The self can not be empty tensor."),
             return ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入aclTensor的shape是否符合约束
    CHECK_RET(CheckTbcShape(self, input, weight, bias, pad, gradInput, gradWeight, gradBias), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查cubeMathType
    CHECK_RET(CheckTbcCubeMathType(self, input, weight, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckParamsValue(const aclIntArray *params, bool isPad)
{
    int64_t minValue = (isPad) ? 0 : 1;
    if (params != nullptr) {
        for (uint64_t i = 0; i < params->Size(); ++i) {
            if ((*params)[i] < minValue) {
                return false;
            }
        }
    }
    return true;
}

static string AclarrayToString(const aclIntArray *array) {
  string str = "";
  if (array == nullptr) {
    return str;
  }
  for (uint64_t i = 0; i < array->Size(); ++i) {
    str += to_string((*array)[i]);
    if (i < array->Size() - 1) {
      str += ",";
    }
  }
  return str;
}

static bool CheckConvParams(const ConvolutionBackwardParams &params, size_t inputDim)
{
    // stride >= 1
    OP_CHECK(CheckParamsValue(params.stride, false),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of stride[%s] must be greater than or equal to 1.",
             AclarrayToString(params.stride).c_str()),
             return false);

    // padding >= 0
    OP_CHECK(CheckParamsValue(params.padding, true),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of padding[%s] must be greater than or equal to 0.",
             AclarrayToString(params.padding).c_str()),
             return false);

    // dilation >= 1
    OP_CHECK(CheckParamsValue(params.dilation, false),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of dilation[%s] must be greater than or equal to 1.",
             AclarrayToString(params.dilation).c_str()),
             return false);
    // outputPadding >= 0
    if (params.transposed) {
      OP_CHECK(CheckParamsValue(params.outputPadding, true),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of outputPadding[%s] must be greater than or equal to 0 if transposed",
             AclarrayToString(params.outputPadding).c_str()),
             return false);
      if (inputDim == CONV3DINPUTDIM) {
        for (uint64_t i = 0; i < params.outputPadding->Size(); ++i) {
          OP_CHECK((*params.outputPadding)[i] < (*params.stride)[i],
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of outputPadding[%s] should smaller than stride[%s]",
              AclarrayToString(params.outputPadding).c_str(), AclarrayToString(params.stride).c_str()),
              return false);
        }
      }
    } else if (params.outputPadding != nullptr) { // !transposed, outputPadding value is unneeded
      for (uint64_t i = 0; i < params.outputPadding->Size(); ++i) {
        OP_CHECK((*params.outputPadding)[i] == 0,
          OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of outputPadding[%s] must be 0 if not transposed",
            AclarrayToString(params.outputPadding).c_str()),
            return false);
      }
    }

    // group >= 1
    OP_CHECK(params.groups >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The group[%ld] must be greater than or equal to 1.", params.groups),
             return false);
    return true;
}

static bool CheckConvChannelAndGroup(const ConvolutionBackwardInputTensor &inputTensor, const ConvolutionBackwardParams &params)
{
    op::Shape inputShape = params.transposed ? inputTensor.gradOutput->GetViewShape() :
        inputTensor.input->GetViewShape();
    op::Shape weightShape = inputTensor.weight->GetViewShape();
    op::Shape gradOutShape = params.transposed ? inputTensor.input->GetViewShape() :
        inputTensor.gradOutput->GetViewShape();
    int64_t channelDim = 1; // NCHW

    OP_CHECK(gradOutShape.GetDim(channelDim) == weightShape.GetDim(0), // 0: NCHW, the order of N(out_channel)
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput_channel(%ld) != weight_N_dim(%ld)",
        gradOutShape.GetDim(channelDim), weightShape.GetDim(0)),
        return false);

    bool channelCheck = weightShape.GetDim(channelDim) == 0 ||
        inputShape.GetDim(channelDim) % weightShape.GetDim(channelDim) != 0;
    OP_CHECK(!channelCheck,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input_channel(%ld) %% weight_channel(%ld) != 0",
        inputShape.GetDim(channelDim), weightShape.GetDim(channelDim)),
        return false);

    int32_t groups = inputShape.GetDim(channelDim) / weightShape.GetDim(channelDim);
    bool groupCheck = groups == params.groups;
    OP_CHECK(groupCheck,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input_channel(%ld) / weight_channel(%ld) != groups(%ld)",
        inputShape.GetDim(channelDim), weightShape.GetDim(channelDim), params.groups),
        return false);

    if (inputShape.GetDim(channelDim) == params.groups){
      auto outChannel = gradOutShape.GetDim(channelDim);
      OP_CHECK(outChannel >= params.groups,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when input_channel(%ld) == groups(%ld), output_channel(%ld) need bigger groups",
        inputShape.GetDim(channelDim),  params.groups, outChannel),
        return false);

      OP_CHECK(gradOutShape.GetDim(channelDim) % params.groups == 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when input_channel(%ld) == groups(%ld), output_channel(%ld) need k times input_channel",
        inputShape.GetDim(channelDim),  params.groups, outChannel),
        return false);
    }
    
    return true;
}

static bool CheckResolutionGEKernelShape(const op::Shape &inputShape, const op::Shape &weightShape,
                                       const ConvolutionBackwardParams &params, int64_t dimIdx)
{
  int64_t dimOrder = dimIdx - 2; // 2: the dim N and D
  int64_t filterDimDilation = (weightShape[dimIdx] - 1) * (*params.dilation)[dimOrder] + 1;
  int64_t dimInput = inputShape.GetDim(dimIdx) + (*params.padding)[dimOrder] * 2 - filterDimDilation; // 2 : pad two dim
  bool dimInputExpect = dimInput >= 0;
  OP_CHECK(dimInputExpect,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "(in_dim(%ld) + pad_dim(%ld) * 2) should >= ((weight_shape(%ld) - 1) * dilation(%ld) + 1)",
      inputShape.GetDim(dimIdx), (*params.padding)[dimOrder], weightShape[dimIdx], (*params.dilation)[dimOrder]),
      return false);
  return true;
}

static int64_t GetExpectNum(const op::Shape &inputShape, const op::Shape &weightShape,
                                       const ConvolutionBackwardParams &params, int64_t dimIdx)
{
  int64_t dimOrder = dimIdx - 2; // 2: the dim N and D
  int64_t filterDimDilation = (weightShape[dimIdx] - 1) * (*params.dilation)[dimOrder] + 1;
  int64_t dimInput = inputShape.GetDim(dimIdx) + (*params.padding)[dimOrder] * 2 - filterDimDilation; // 2 : pad two dim
  int64_t dimExpect = dimInput / (*params.stride)[dimOrder] + 1;
  return dimExpect;
}

static bool CheckConvShapePlus(const ConvolutionBackwardInputTensor &inputTensor,
                               const ConvolutionBackwardParams &params)
{
    op::Shape inputShape = params.transposed ? inputTensor.gradOutput->GetViewShape() :
        inputTensor.input->GetViewShape();
    op::Shape weightShape = inputTensor.weight->GetViewShape();
    op::Shape gradOutShape = params.transposed ? inputTensor.input->GetViewShape() :
        inputTensor.gradOutput->GetViewShape();
    auto inputDim = inputShape.GetDimNum();
    if (inputDim == CONV3DINPUTDIM) {
        int64_t depthIdx = 2; // NCDHW
        int64_t heightIdx = 3; // NCDHW
        int64_t widthIdx = 4; // NCDHW
        if (!CheckResolutionGEKernelShape(inputShape, weightShape, params, depthIdx) ||
            !CheckResolutionGEKernelShape(inputShape, weightShape, params, heightIdx) ||
            !CheckResolutionGEKernelShape(inputShape, weightShape, params, widthIdx)) {
            return false;
        }
        int64_t doExpect = GetExpectNum(inputShape, weightShape, params, depthIdx);
        int64_t hoExpect = GetExpectNum(inputShape, weightShape, params, heightIdx);
        int64_t woExpect = GetExpectNum(inputShape, weightShape, params, widthIdx);
        bool expectCheck = doExpect == gradOutShape.GetDim(depthIdx) &&
            hoExpect == gradOutShape.GetDim(heightIdx) && woExpect == gradOutShape.GetDim(widthIdx);
        OP_CHECK(expectCheck,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "gradOutput's shape%s is not equal with inferred shape[%ld,%ld,%ld,%ld,%ld]",
            op::ToString(gradOutShape).GetString(),
            gradOutShape.GetDim(0), gradOutShape.GetDim(1), doExpect, hoExpect, woExpect),
            return false);
    }

    return true;
}

static bool CheckConvShape(const ConvolutionBackwardInputTensor &inputTensor,
                           const ConvolutionBackwardOutput &outputTensor, const ConvolutionBackwardParams &params)
{
    auto gradOutputDim = inputTensor.gradOutput->GetViewShape().GetDimNum();
    auto inputDim = inputTensor.input->GetViewShape().GetDimNum();
    auto weightDim = inputTensor.weight->GetViewShape().GetDimNum();

    OP_CHECK(gradOutputDim == inputDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of gradOutput and input should be equal."), return false);

    OP_CHECK(inputDim == weightDim, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of input and weight should be equal."),
             return false);
    // 检查gradOutput和weight是否为空tensor
    OP_CHECK(inputTensor.gradOutput->Size() != 0 && inputTensor.weight->Size() != 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gradOutput and weight cannot be empty tensor."),
             return false);
    if (outputTensor.gradInput != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.input, outputTensor.gradInput, return false);
    }

    if (outputTensor.gradWeight != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.weight, outputTensor.gradWeight, return false);
    }

    int64_t channelOutDim = 1;  // NCHW
    int64_t cOut = inputTensor.gradOutput->GetViewShape().GetDim(channelOutDim);
    if (outputTensor.gradBias != nullptr) {
        OP_CHECK(outputTensor.gradBias->GetViewShape().GetDimNum() == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of gradBias only support 1."), return false);
        OP_CHECK(outputTensor.gradBias->GetViewShape().GetDim(0) == cOut,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gradBias shape should be equal [%ld].", cOut), return false);
    }

    int64_t paramsDim = inputDim - 2;  // 参数维度应该等于输入tensor维度-2
    int64_t strideDim = params.stride->Size();
    OP_CHECK(strideDim == paramsDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the input dimension is %ld, the dimension of stride should be %ld.",
                     inputDim, paramsDim),
             return false);

    int64_t dilationDim = params.dilation->Size();
    OP_CHECK(dilationDim == paramsDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "When the input dimension is %ld, the dimension of dilation should be %ld.", inputDim, paramsDim),
             return false);

    int64_t paddingDim = params.padding->Size();
    int64_t outputPaddingDim = params.outputPadding->Size();
    if (inputDim == CONV2DINPUTDIM) {
        // padding类支持4维输入
        OP_CHECK(paddingDim == paramsDim || paddingDim == CONV2DINPUTDIM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "When the input dimension is %ld, the dimension of padding should be %ld or %d.", inputDim,
                         paramsDim, CONV2DINPUTDIM),
                 return false);

        OP_CHECK(outputPaddingDim == paramsDim || outputPaddingDim == CONV2DINPUTDIM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "When the input dimension is %ld, the dimension of outputPadding should be %ld or %d.",
                         inputDim, paramsDim, CONV2DINPUTDIM),
                 return false);
    } else {
        OP_CHECK(
            paddingDim == paramsDim,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the input dimension is %ld, the dimension of padding should be %ld.",
                    inputDim, paramsDim),
            return false);

        OP_CHECK(outputPaddingDim == paramsDim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "When the input dimension is %ld, the dimension of outputPadding should be %ld.", inputDim,
                         paramsDim),
                 return false);
    }

    return true;
}

static aclnnStatus CheckParams(const ConvolutionBackwardInputTensor &inputTensor,
                               const ConvolutionBackwardOutput &outputTensor, const ConvolutionBackwardParams &params)
{
    // 检查ConvolutionBackward的输入aclTensor是否符合规范
    //  1. 检查输入aclTensor是否为空指针
    CHECK_RET(CheckNotNull(inputTensor, outputTensor, params), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入aclTensor的Dtype是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(inputTensor.gradOutput, "gradOutput"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(inputTensor.input, "input"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(inputTensor.weight, "weight"), ACLNN_ERR_PARAM_INVALID);
    if (outputTensor.gradInput != nullptr) {
        CHECK_RET(CheckDtypeValid(outputTensor.gradInput, "gradInput"), ACLNN_ERR_PARAM_INVALID);
    }
    if (outputTensor.gradWeight != nullptr) {
        CHECK_RET(CheckDtypeValid(outputTensor.gradWeight, "gradWeight"), ACLNN_ERR_PARAM_INVALID);
    }
    if (outputTensor.gradBias != nullptr) {
        CHECK_RET(CheckDtypeValid(outputTensor.gradBias, "gradBias"), ACLNN_ERR_PARAM_INVALID);
    }

    // 3. 检查输入aclTensor的Format是否在API支持的数据类型范围之内
    CHECK_RET(CheckFormatValid(inputTensor.gradOutput, "gradOutput"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(inputTensor.input, "input"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(inputTensor.weight, "weight"), ACLNN_ERR_PARAM_INVALID);
    if (outputTensor.gradInput != nullptr) {
        CHECK_RET(CheckFormatValid(outputTensor.gradInput, "gradInput"), ACLNN_ERR_PARAM_INVALID);
    }
    if (outputTensor.gradWeight != nullptr) {
        CHECK_RET(CheckFormatValid(outputTensor.gradWeight, "gradWeight"), ACLNN_ERR_PARAM_INVALID);
    }

    if (!params.transposed && outputTensor.gradBias != nullptr) {
      if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B || GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        OP_CHECK(outputTensor.gradBias->GetStorageFormat() == op::Format::FORMAT_ND, 
          OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "gradBias format only support ND, but get [%s].",
            op::ToString(outputTensor.gradBias->GetStorageFormat()).GetString()), return ACLNN_ERR_PARAM_INVALID);
      }
    }
     
    // 4. 检查输入参数的合法性
    CHECK_RET(CheckConvShape(inputTensor, outputTensor, params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckConvChannelAndGroup(inputTensor, params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckConvParams(params, inputTensor.input->GetViewShape().GetDimNum()), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckConvShapePlus(inputTensor, params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckCubeMathTypeConvBackward(inputTensor, params), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckSupportedForConv3dBackpropFilter(ConvolutionBackwardInputTensor &inputTensor,
                                                  ConvolutionBackwardOutput &outputTensor,
                                                  ConvolutionBackwardParams &params)
{
  if (!(*params.outputMask)[1] || params.groups == 1) {
    return true;
  }

  op::DataType gradWeightDtype = outputTensor.gradWeight->GetDataType();
  if (gradWeightDtype != DataType::DT_FLOAT) {
    return true;
  }

  if (params.cubeMathType == USE_FP16) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is not supported for Conv3DBackpropFilter when promoted input dtype is fp16/bf16, "
      "output dtype is fp32 and group > 1. cubeMathType=%d, please consider to change the cubeMathType to 0, 1 or 3."
      , params.cubeMathType);
    return false;
  }

  auto promoteType = CalcPromoteType(inputTensor);
  if (promoteType == DataType::DT_FLOAT16 || promoteType == DataType::DT_BF16) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is not supported for Conv3DBackpropFilter when promoted input dtype is fp16/bf16, "
      "output dtype is fp32 and group > 1. gradOutputDtype=%d, inputDtype=%d, weightDtype=%d, "
      "please consider to change the input dtype to fp32."
      , (inputTensor.gradOutput)->GetDataType(), (inputTensor.input)->GetDataType(), (inputTensor.weight)->GetDataType());
    return false;
  }

  return true;
}

static const aclTensor *Permute(const aclTensor *input, std::vector<int64_t> dims, aclOpExecutor *executor) {
  // contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);
  // Transpose
  auto *perm = executor->AllocIntArray(dims.data(), dims.size());
  CHECK_RET(perm != nullptr, nullptr);

  auto *result = l0op::Transpose(contiguousInput, perm, executor);
  CHECK_RET(result != nullptr, nullptr);

  return result;
}

static aclnnStatus InputPreProcess(const aclTensor *&inputTensor, const string &tensorName,
                                   ConvolutionBackwardParams &params, const op::DataType promoteDtype,
                                   aclOpExecutor *executor, bool c04Flag = false, bool transDataFlag = true) {
  // API输入预处理 l0ResultTensor -> l0op::Contiguous -> l0op::Cast -> l0op::TransData -> inputTensor
  inputTensor = l0op::Contiguous(inputTensor, executor);
  OP_CHECK(inputTensor != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, %s with Contiguous return nullptr.",
                   tensorName.c_str()),
           return ACLNN_ERR_INNER_NULLPTR);

  bool useFp16Input =
      (params.cubeMathType == USE_FP16 || (!IsInputSupportFp32Local() && params.cubeMathType == ALLOW_FP32_DOWN_PRECISION));
  if (useFp16Input) {
    OP_LOGD("According to the configuration of cubeMathType, use Fp16 to calculation.");
    inputTensor = l0op::Cast(inputTensor, DataType::DT_FLOAT16, executor);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, %s with Cast return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);
  } else {
    inputTensor = l0op::Cast(inputTensor, promoteDtype, executor);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, %s with Cast return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);
  }
  auto inputDim = inputTensor->GetViewShape().GetDimNum();
  if (inputDim == CONV3DINPUTDIM) {
    if (tensorName == "weight") {
      inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_Z_3D, params.groups, executor);
    } else {
      if (tensorName == "input" && !transDataFlag) {
          return ACLNN_SUCCESS;
      }
      inputTensor = l0op::TransData(inputTensor, Format::FORMAT_NDC1HWC0, params.groups, executor);
    }
  } else {
    if (tensorName == "weight") {
      if (c04Flag) {
        inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_Z_C04, params.groups, executor);
      } else {
        inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_Z, params.groups, executor);
      }
    } else {
      inputTensor = l0op::TransData(inputTensor, Format::FORMAT_NC1HWC0, params.groups, executor);
    }
  }
  OP_CHECK(inputTensor != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, %s with TransData return nullptr.",
                   tensorName.c_str()),
           return ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus OutputPostProcess(const aclTensor *&outputTensor, const aclTensor *&l0ResultTensor,
                                     const string &tensorName, const int64_t groups, aclOpExecutor *executor) {
  // API输出后处理 l0ResultTensor -> L0op::Cast(定制) -> L0op::Transdata
  OP_LOGD("%s l0ResultTensor is: %s", tensorName.c_str(), l0ResultTensor->ToString().GetString());
  int64_t storageShapeDimSize = (int64_t) l0ResultTensor->GetStorageShape().GetDimNum();
  bool needSpecialCast = (l0ResultTensor->GetDataType() == op::DataType::DT_FLOAT) &&
    (l0ResultTensor->GetStorageShape().GetDim(storageShapeDimSize - 1) == 16) && (groups > 1);
  // 特殊场景，先cast 再transdata规避
  if (needSpecialCast) {
    l0ResultTensor = l0op::CastOnlyForConvBackward(l0ResultTensor, outputTensor->GetDataType(), executor);
    OP_CHECK(l0ResultTensor != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess fail, l0ResultTensor with cast return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);
    outputTensor = l0op::TransData(l0ResultTensor, GetPrimaryFormat(outputTensor->GetOriginalFormat()), groups, executor);
    OP_CHECK(outputTensor != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess fail, outputTensor with transdata return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);
  } else {
    auto transTensor = l0op::TransData(l0ResultTensor, GetPrimaryFormat(outputTensor->GetOriginalFormat()), groups,
                                       executor);
    OP_CHECK(transTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with TransData return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("%s outputTensor dtype: %s", tensorName.c_str(), op::ToString(outputTensor->GetDataType()).GetString());
    outputTensor = l0op::Cast(transTensor, outputTensor->GetDataType(), executor);
    OP_CHECK(outputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with Cast return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("%s outputTensor is: %s", tensorName.c_str(), outputTensor->ToString().GetString());
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus OutputPostProcessWithoutTransdata(const aclTensor *&outputTensor, const aclTensor *&l0ResultTensor,
                                     const string &tensorName, aclOpExecutor *executor) {
  OP_LOGD("Output process without transdata.");
  OP_LOGD("%s l0ResultTensor is: %s", tensorName.c_str(), l0ResultTensor->ToString().GetString());
  outputTensor = l0op::Cast(l0ResultTensor, outputTensor->GetDataType(), executor);
  OP_CHECK(outputTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with Cast return nullptr.",
                    tensorName.c_str()),
            return ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("%s outputTensor is: %s", tensorName.c_str(), outputTensor->ToString().GetString());
  return ACLNN_SUCCESS;
}

static aclnnStatus OutputViewProcess(ConvolutionBackwardResult &ResultTensor, ConvolutionBackwardOutput &outputTensor,
                                     const aclBoolArray *outputMask, aclOpExecutor *executor) {
  // API进行ViewCopy处理最终输出
  // Index 为 0：ViewCopy gradInput
  if ((*outputMask)[0]) {
    auto result = l0op::ViewCopy(ResultTensor.gradInput, outputTensor.gradInput, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, gradInput with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }
  // Index 为 1：ViewCopy gradWeight
  if ((*outputMask)[1]) {
    auto result = l0op::ViewCopy(ResultTensor.gradWeight, outputTensor.gradWeight, executor);
    OP_CHECK(
        result != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, gradWeight with ViewCopy return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
  }
  // Index 为 2：ViewCopy gradBias
  if ((*outputMask)[2]) {
    auto result = l0op::ViewCopy(ResultTensor.gradBias, outputTensor.gradBias, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, gradBias with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclIntArray *View1dAs2d(const aclIntArray *intArray, int64_t expendValue, aclOpExecutor *executor,
                               const string &tensorName) {
  const uint64_t newDimSize = 2;
  int64_t data[newDimSize];
  uint64_t arraySize = intArray->Size();
  OP_CHECK(arraySize == 1,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The %s's dimension can only be set to 1 with Conv1D, actually is %ld.",
                   tensorName.c_str(), arraySize),
           return nullptr);

  data[0] = expendValue;
  data[1] = (*intArray)[0];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  return newArray;
}

static const aclTensor *View4d(const aclTensor *input, aclOpExecutor *executor, const string &tensorName) {
  // input NCL->contigious->unsqueeze(2)->reformat NCHW
  // 非连续转连续contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);

  // unsqeeze(2)/unsqueeze(2,3)
  auto inputDim = input->GetViewShape().GetDimNum();
  const int64_t expendedDim = 4;
  int64_t append_dim[expendedDim];
  for (auto i = inputDim - 1; i < expendedDim - 1; i++) {
    append_dim[i - inputDim + 1] = i;
  }
  aclIntArray *dim = executor->AllocIntArray(append_dim, expendedDim - inputDim);
  auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dim, executor);
  OP_CHECK(
      unsqueezedInput != nullptr,
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view4d failed, %s with UnsqueezeNd return nullptr.", tensorName.c_str()),
      return nullptr);

  // reformat
  auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
  OP_CHECK(reformatInput != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view4d failed, %s with ReFormat return nullptr.", tensorName.c_str()),
           return nullptr);

  return reformatInput;
}

static const aclTensor *View3d(const aclTensor *input, aclOpExecutor *executor, const string &tensorName) {
  // input NCL->contigious->unsqueeze(2)->reformat NCHW
  // 非连续转连续contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);
  // unsqeeze(2)
  const int64_t append_dim[] = {2};
  aclIntArray *dim = executor->AllocIntArray(append_dim, 1);
  auto squeezedInput = l0op::SqueezeNd(contiguousInput, dim, executor);
  OP_CHECK(squeezedInput != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view3d failed, %s with SqueezeNd return nullptr.", tensorName.c_str()),
           return nullptr);

  // reformat
  auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCL);
  OP_CHECK(reformatInput != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view3d failed, %s with ReFormat return nullptr.", tensorName.c_str()),
           return nullptr);

  return reformatInput;
}

static const aclTensor *PreDilation(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params,
                                    aclOpExecutor *executor) {
  const aclTensor *preDilationGradOutputNC1HWC0 = nullptr;
  int64_t preDilationDilationsVector[] = {1, 1, (*params.stride)[kSTRIDEHIdx], (*params.stride)[kSTRIDEWIdx], 1};
  /* 当输出的h/w为1时，把传给Dilation算子的dilation值修正为fmap_h/w，避免在dilation值超大时，Dilation算子超时 */
  if (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEHIdx] > inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx)) {
    preDilationDilationsVector[kHDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx);
  }
  if (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEWIdx] > inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx)) {
    preDilationDilationsVector[kWDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx);
  }
  aclIntArray *preDilationDilations = executor->AllocIntArray(preDilationDilationsVector, kDILATIONSDIM);
  int64_t preDilationPadUp = 0;
  int64_t preDilationPadLeft = 0;
  int64_t preDilationPadH = 0;
  int64_t preDilationPadW = 0;
  if (params.padding->Size() == 4) {
    preDilationPadH = (*params.padding)[kPadding4UpIdx] + (*params.padding)[kPadding4DownIdx];
    preDilationPadW = (*params.padding)[kPadding4LeftIdx] + (*params.padding)[kPadding4RightIdx];
  } else {
    preDilationPadH = 2 * (*params.padding)[kPADDINGUPIdx];
    preDilationPadW = 2 * (*params.padding)[kPADDINGLEFTIdx];
  }
  int64_t preDilationPadDown =
      inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx) -
      (inputTensor.weight->GetViewShape().GetDim(kHDimNCHWIdx) - 1) * (*params.dilation)[kDILATIONHIdx] +
      preDilationPadH -
      (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEHIdx] - 1;
  int64_t preDilationPadRight =
      inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx) -
      (inputTensor.weight->GetViewShape().GetDim(kWDimNCHWIdx) - 1) * (*params.dilation)[kDILATIONWIdx] +
      preDilationPadW -
      (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEWIdx] - 1;

  int64_t preDilationPadsVector[] = {preDilationPadUp, preDilationPadDown, preDilationPadLeft, preDilationPadRight};
  aclIntArray *preDilationPads = executor->AllocIntArray(preDilationPadsVector, kPADDINGDIM);

  float paddingValue = 0;

  preDilationGradOutputNC1HWC0 =
      l0op::Dilation(inputTensor.gradOutput, preDilationDilations, preDilationPads, paddingValue, executor);
  return preDilationGradOutputNC1HWC0;
}

static const aclTensor *PostDilation(const aclTensor *dxGradInputNC1HWC0, ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  const aclTensor *postDilationGradInputNC1HWC0 = nullptr;
  int64_t postDilationDilationsVector[] = {1, 1, (*params.stride)[kSTRIDEHIdx], (*params.stride)[kSTRIDEWIdx], 1};
  /* 当输出的h/w为1时，把传给Dilation算子的dilation值修正为fmap_h/w，避免在dilation值超大时，Dilation算子超时 */
  if (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEHIdx] > inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx)) {
    postDilationDilationsVector[kHDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx);
  }
  if (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEWIdx] > inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx)) {
    postDilationDilationsVector[kWDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx);
  }
  aclIntArray *post_dilation_dilations = executor->AllocIntArray(postDilationDilationsVector, kDILATIONSDIM);

  int64_t postDilationPadUp = 0;
  int64_t postDilationPadLeft = 0;
  int64_t padUp = 0;
  int64_t padLeft = 0;
  if (params.padding->Size() == 4) {
    postDilationPadUp = -(*params.padding)[kPadding4UpIdx];
    postDilationPadLeft = -(*params.padding)[kPadding4LeftIdx];
    padUp = (*params.padding)[kPadding4UpIdx];
    padLeft = (*params.padding)[kPadding4LeftIdx];
  } else {
    postDilationPadUp = -(*params.padding)[kPADDINGUPIdx];
    postDilationPadLeft = -(*params.padding)[kPADDINGLEFTIdx];
    padUp = (*params.padding)[kPADDINGUPIdx];
    padLeft = (*params.padding)[kPADDINGLEFTIdx];
  }
  int64_t postDilationPadDown =
      inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx) + padUp -
      (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEHIdx] - 1;
  int64_t postDilationPadRight =
      inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx) + padLeft -
      (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEWIdx] - 1;
  int64_t postDilationPadsVector[] = {postDilationPadUp, postDilationPadDown, postDilationPadLeft,
                                      postDilationPadRight};
  aclIntArray *postDilationPads = executor->AllocIntArray(postDilationPadsVector, kPADDINGDIM);

  float paddingValue = 0;

  postDilationGradInputNC1HWC0 =
      l0op::Dilation(dxGradInputNC1HWC0, post_dilation_dilations, postDilationPads, paddingValue, executor);
  return postDilationGradInputNC1HWC0;
}

static const aclTensor *PerformConv2DBackpropInput(ConvolutionBackwardInputTensor &inputTensor,
                                            ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  const aclTensor *gradInputNC1HWC0 = nullptr;
  bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  if (useHf32) {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputHf32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                      params.padding, params.dilation, params.groups, executor);
  } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputFp322Fp32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                           params.padding, params.dilation, params.groups, executor);
  } else if (inputTensor.weight->GetDataType() == DataType::DT_BF16) {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputBf162Bf16(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                           params.padding, params.dilation, params.groups, executor);
  } else {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputFp162Fp16(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                           params.padding, params.dilation, params.groups, executor);
  }
  return gradInputNC1HWC0;
}

const aclTensor *CalculateConv2DBackpropInput(ConvolutionBackwardInputTensor &inputTensor,
                                                     ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  const aclTensor *dxGradInputNC1HWC0Res = nullptr;
  if (IsPreInsertDilation(inputTensor, params) && IsInputSupportInsertDilation() &&
      inputTensor.input->GetDataType() != DataType::DT_BF16) {
    const aclTensor *preDilationGradOutputNC1HWC0 = nullptr;
    preDilationGradOutputNC1HWC0 = PreDilation(inputTensor, params, executor);
    int64_t newstrideVector[] = {1, 1};
    aclIntArray *newstride = executor->AllocIntArray(newstrideVector, kCONV2DDXHWDIM);
    ConvolutionBackwardInputTensor newInputTensor = {preDilationGradOutputNC1HWC0, inputTensor.input,
                                                     inputTensor.weight};
    ConvolutionBackwardParams newparams = {params.biasSizes, newstride,         params.padding,
                                           params.dilation,  params.transposed, params.outputPadding,
                                           params.groups,    params.outputMask, params.cubeMathType};
    dxGradInputNC1HWC0Res = PerformConv2DBackpropInput(newInputTensor, newparams, executor);
  } else if (IsPostInsertDilation(inputTensor.weight, params) && IsInputSupportInsertDilation() &&
             inputTensor.input->GetDataType() != DataType::DT_BF16) {
    const aclTensor *dxGradInputNC1HWC0 = nullptr;
    int64_t newstrideVector[] = {1, 1};
    int64_t newpaddingVector[] = {0, 0, 0, 0};
    aclIntArray *newstride = executor->AllocIntArray(newstrideVector, kCONV2DDXHWDIM);
    aclIntArray *newpadding = executor->AllocIntArray(newpaddingVector, kPADDINGDIM);
    op::Shape newInputStorageShape;
    op::Shape newInputOrishape;
    for (size_t i = 0; i < inputTensor.input->GetStorageShape().GetDimNum(); i++) {
      newInputStorageShape.AppendDim(i == kHDimNC1HWC0Idx || i == kWDimNC1HWC0Idx
                                         ? inputTensor.gradOutput->GetStorageShape().GetDim(i)
                                         : inputTensor.input->GetStorageShape().GetDim(i));}
    for (size_t i = 0; i < inputTensor.input->GetViewShape().GetDimNum(); i++) {
      newInputOrishape.AppendDim(i == kHDimNCHWIdx || i == kWDimNCHWIdx
                                     ? inputTensor.gradOutput->GetViewShape().GetDim(i)
                                     : inputTensor.input->GetViewShape().GetDim(i));}
    auto newInput =
        executor->AllocTensor(newInputStorageShape, newInputOrishape, inputTensor.weight->GetDataType(),
                              inputTensor.input->GetStorageFormat(), inputTensor.input->GetOriginalFormat());
    ConvolutionBackwardInputTensor newInputTensor = {inputTensor.gradOutput, newInput, inputTensor.weight};
    ConvolutionBackwardParams newparams = {params.biasSizes, newstride,         newpadding,
                                           params.dilation,  params.transposed, params.outputPadding,
                                           params.groups,    params.outputMask, params.cubeMathType};
    dxGradInputNC1HWC0 = PerformConv2DBackpropInput(newInputTensor, newparams, executor);
    OP_CHECK(dxGradInputNC1HWC0 != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with PerformConv2DBackpropInput failed, Conv2dBackpropInput return nullptr."),
             return nullptr);
    dxGradInputNC1HWC0Res = PostDilation(dxGradInputNC1HWC0, inputTensor, params, executor);
  } else {
    dxGradInputNC1HWC0Res = PerformConv2DBackpropInput(inputTensor, params, executor);
  }
  OP_CHECK(dxGradInputNC1HWC0Res != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The calculation with dxGradInputNC1HWC0Res failed, return nullptr."),
           return nullptr);
  return dxGradInputNC1HWC0Res;
}

static aclnnStatus CalculateBiasGrad(ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                     aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  if ((*params.outputMask)[2]) {
    OP_LOGD("Enter bias grad Calculate");
    auto gradContiguous = l0op::Contiguous(inputTensor.gradOutput, executor);
    const int64_t reshapeList[] = {inputTensor.gradOutput->GetViewShape().GetDim(0),
                                   inputTensor.gradOutput->GetViewShape().GetDim(1), -1};
    auto gradReshape = l0op::Reshape(gradContiguous, executor->AllocIntArray(reshapeList, 3), executor);
    OP_CHECK(gradReshape != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The Reshape with gradOutput failed."),
           return ACLNN_ERR_INNER_NULLPTR);
    auto gradReshapeND = l0op::ReFormat(gradReshape, op::Format::FORMAT_ND);
    OP_CHECK(gradReshapeND != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ReFormat with gradOutput failed."),
           return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("gradReshapeND: %s", gradReshapeND->ToString().GetString());
    const int64_t dim[] = {0, 2};
    auto gradBiasResult = l0op::ReduceSumOp(gradReshapeND, executor->AllocIntArray(dim, 2), false, executor);
    OP_CHECK(gradBiasResult != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ReduceSumOp with gradOutput failed."),
            return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("gradBiasResult: %s", gradBiasResult->ToString().GetString());
    CHECK_RET(
        OutputPostProcess(outputTensor.gradBias, gradBiasResult, "gradBias", params.groups, executor) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv2DBackward(ConvolutionBackwardInputTensor &inputTensor,
                                           ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                           aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  auto promoteType = CalcPromoteType(inputTensor);
  CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  bool c04Flag = IsC04WhiteListCase(inputTensor, params);
  CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor, c04Flag) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());

  // Index 为 0：进行dx运算
  if ((*params.outputMask)[0]) {
    OP_LOGD("Enter dx Calculate");
    const aclTensor *gradInputNC1HWC0 = nullptr;
    gradInputNC1HWC0 = CalculateConv2DBackpropInput(inputTensor, params, executor);
    OP_CHECK(gradInputNC1HWC0 != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropInput return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    bool outputTransdataFlag = true;
    l0op::ConvBackpropParams conv2DBackpropPrarams = {inputTensor.input, inputTensor.weight, inputTensor.gradOutput,
      params.stride, params.padding, params.dilation, params.groups};
    bool useV2Flag = l0op::IsConv2DBackpropInputV2(conv2DBackpropPrarams);
    bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
    if (useV2Flag && !useHf32 && inputTensor.weight->GetDataType() != DataType::DT_FLOAT) {
      outputTransdataFlag = false; // V2，非HF32，非FP32，不在黑名单
    }
    aclnnStatus status = outputTransdataFlag ?
      OutputPostProcess(outputTensor.gradInput, gradInputNC1HWC0, "gradInput", params.groups, executor)
      : OutputPostProcessWithoutTransdata(outputTensor.gradInput, gradInputNC1HWC0, "gradInput", executor);
    CHECK_RET(status == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  }

  // Index 为 1：进行dw运算
  if ((*params.outputMask)[1]) {
    OP_LOGD("Enter dw Calculate");
    const aclTensor *gradWeightFZ = nullptr;
    bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
    auto inputDtype = inputTensor.input->GetDataType();
    if (IsC04WhiteListCase(inputTensor, params)) {
      if (inputDtype == DataType::DT_FLOAT16) {
        gradWeightFZ = l0op::Conv2DBackpropFilterV2Fp162Fp32(inputTensor.input, inputTensor.weight,
                                                             inputTensor.gradOutput, params.stride, params.padding,
                                                             params.dilation, params.groups, executor);
      }  else if (inputDtype == DataType::DT_BF16) {
        gradWeightFZ = l0op::Conv2DBackpropFilterV2Bf162Fp32(inputTensor.input, inputTensor.weight,
                                                             inputTensor.gradOutput, params.stride, params.padding,
                                                             params.dilation, params.groups, executor);
      }
    } else {
      if (useHf32) {
        gradWeightFZ =
            l0op::Conv2DBackpropFilterHf32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                           params.padding, params.dilation, params.groups, executor);
      } else if (inputDtype == DataType::DT_FLOAT) {
        gradWeightFZ = l0op::Conv2DBackpropFilterFp322Fp32(inputTensor.input, inputTensor.weight,
                                                           inputTensor.gradOutput, params.stride, params.padding,
                                                           params.dilation, params.groups, executor);
      } else if (inputDtype == DataType::DT_BF16) {
        gradWeightFZ = l0op::Conv2DBackpropFilterBf162Fp32(inputTensor.input, inputTensor.weight,
                                                           inputTensor.gradOutput, params.stride, params.padding,
                                                           params.dilation, params.groups, executor);
      } else {
        gradWeightFZ = l0op::Conv2DBackpropFilterFp162Fp32(inputTensor.input, inputTensor.weight,
                                                           inputTensor.gradOutput, params.stride, params.padding,
                                                           params.dilation, params.groups, executor);
      }
    }

    OP_CHECK(gradWeightFZ != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropFilter return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ, "gradWeight", params.groups, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }

  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv2DTransposeBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

  auto promoteType = CalcPromoteType(inputTensor);
  CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());
  bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  if ((*params.outputMask)[0]) {
    OP_LOGD("Enter dx Calculate");
    const aclTensor *gradInputNC1HWC0 = nullptr;
    if (useHf32) {
      gradInputNC1HWC0 = l0op::Conv2d5HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                             params.padding, params.dilation, params.groups, true, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
      gradInputNC1HWC0 = l0op::Conv2d5HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                             params.padding, params.dilation, params.groups, false, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradInputNC1HWC0 = l0op::Conv2d5HdBf16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                             params.padding, params.dilation, params.groups, executor);
    } else {
      gradInputNC1HWC0 = l0op::Conv2d5HdFp16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                             params.padding, params.dilation, params.groups, executor);
    }
    OP_CHECK(
        gradInputNC1HWC0 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The calculation with empty tensor failed, Conv2d5HdFp16 return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradInput, gradInputNC1HWC0, "gradInput", params.groups, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }

  if ((*params.outputMask)[1]) {
    OP_LOGD("Enter dw Calculate");
    const aclTensor *gradWeightFZ = nullptr;
    if (useHf32) {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterHf32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input, params.stride,
                                         params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_FLOAT) {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterFp322Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterBf162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    } else {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterFp162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    }

    OP_CHECK(gradWeightFZ != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropFilter return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ, "gradWeight", params.groups, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv1DBackward(ConvolutionBackwardInputTensor &inputTensor,
                                           ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                           aclOpExecutor *executor) {
  params.stride = View1dAs2d(params.stride, 1, executor, "stride");
  CHECK_RET(params.stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

  params.padding = View1dAs2d(params.padding, 0, executor, "padding");
  CHECK_RET(params.padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

  params.dilation = View1dAs2d(params.dilation, 1, executor, "dilation");
  CHECK_RET(params.dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.input = View4d(inputTensor.input, executor, "input");
  CHECK_RET(inputTensor.input != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.weight = View4d(inputTensor.weight, executor, "weight");
  CHECK_RET(inputTensor.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.gradOutput = View4d(inputTensor.gradOutput, executor, "gradOutput");
  CHECK_RET(inputTensor.gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  outputTensor.gradInput = View4d(outputTensor.gradInput, executor, "gradInput");
  CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  outputTensor.gradWeight = View4d(outputTensor.gradWeight, executor, "gradWeight");
  CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclnnStatus ret = CalculateConv2DBackward(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

  if ((*params.outputMask)[0]) {
    outputTensor.gradInput = View3d(outputTensor.gradInput, executor, "gradInput");
    CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if ((*params.outputMask)[1]) {
    outputTensor.gradWeight = View3d(outputTensor.gradWeight, executor, "gradWeight");
    CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv1DTransposeBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  params.stride = View1dAs2d(params.stride, 1, executor, "stride");
  CHECK_RET(params.stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

  params.padding = View1dAs2d(params.padding, 0, executor, "padding");
  CHECK_RET(params.padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

  params.dilation = View1dAs2d(params.dilation, 1, executor, "dilation");
  CHECK_RET(params.dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.input = View4d(inputTensor.input, executor, "input");
  CHECK_RET(inputTensor.input != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.weight = View4d(inputTensor.weight, executor, "weight");
  CHECK_RET(inputTensor.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.gradOutput = View4d(inputTensor.gradOutput, executor, "gradOutput");
  CHECK_RET(inputTensor.gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  outputTensor.gradInput = View4d(outputTensor.gradInput, executor, "gradInput");
  CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  outputTensor.gradWeight = View4d(outputTensor.gradWeight, executor, "gradWeight");
  CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclnnStatus ret = CalculateConv2DTransposeBackward(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

  if ((*params.outputMask)[0]) {
    outputTensor.gradInput = View3d(outputTensor.gradInput, executor, "gradInput");
    CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if ((*params.outputMask)[1]) {
    outputTensor.gradWeight = View3d(outputTensor.gradWeight, executor, "gradWeight");
    CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv1DBackwardSpecial(ConvolutionBackwardInputTensor &inputTensor,
                                                  ConvolutionBackwardResult &outputTensor,
                                                  ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  // to do
  return ACLNN_SUCCESS;
}

static bool IsConv3DBpSupportMatmulMode(const aclTensor *inputTensor, ConvolutionBackwardParams &params)
{
  // must be NCDHW format
  op::Format format = inputTensor->GetStorageFormat();
  if (format != op::Format::FORMAT_NCDHW) {
    return false;
  }
  // padding
  int64_t paddingVecSize = params.padding->Size();
  for (int64_t paddingIdx = 0; paddingIdx < paddingVecSize; ++paddingIdx) {
    if ((*params.padding)[paddingIdx] != 0) {
      return false;
    }
  }
  // outputpadding
  paddingVecSize = params.outputPadding->Size();
  for (int64_t paddingIdx = 0; paddingIdx < paddingVecSize; ++paddingIdx) {
    if ((*params.outputPadding)[paddingIdx] != 0) {
      return false;
    }
  }
  // dilation
  int64_t dilationVecSize = params.dilation->Size();
  for (int64_t dilationIdx = 0; dilationIdx < dilationVecSize; ++dilationIdx) {
    if ((*params.dilation)[dilationIdx] != 1) {
      return false;
    }
  }
  // attribute
  if (params.transposed || params.groups != 1) {
    return false;
  }
  return true;
}

static Conv3DBp2MmMode GetConv3DBp2MmMode(ConvolutionBackwardInputTensor &inputTensor,
  ConvolutionBackwardParams &params)
{
  const auto &inputShape = inputTensor.input->GetViewShape();
  const auto &weightShape = inputTensor.weight->GetViewShape();
  const auto &gradOutputShape = inputTensor.gradOutput->GetViewShape();
  if (!IsConv3DBpSupportMatmulMode(inputTensor.input, params)) {
    return Conv3DBp2MmMode::CONV3D_BP_NO_MM;
  }

  bool is1x1Kernel = true;
  bool isFmEqKernel = true;
  bool isStrideEqKernel = true;
  const vector<int64_t> dimIdxVec {dDimNCDHWIdx, hDimNCDHWIdx, wDimNCDHWIdx};
  const vector<int64_t> strideIdxVec {0, 1, 2}; // 0 : depth 1 : height 2 : width
  const int64_t resolutionDimSize = (int64_t)(dimIdxVec.size());
  for (int64_t idx = 0; idx < resolutionDimSize; ++idx) {
    int64_t dimIdx = dimIdxVec[idx];
    int64_t strideIdx = strideIdxVec[idx];
    bool isDim1x1Kernel = (weightShape[dimIdx] == 1) && (gradOutputShape[dimIdx] == inputShape[dimIdx]);
    if (!isDim1x1Kernel) {
      is1x1Kernel = false;
    }
    bool isDimFmEqKernel = (weightShape[dimIdx] == inputShape[dimIdx]) && (gradOutputShape[dimIdx] == 1);
    if (!isDimFmEqKernel) {
      isFmEqKernel = false;
    }
    if (weightShape[dimIdx] != (*params.stride)[strideIdx]) {
      isStrideEqKernel = false;
    }
  }
  if (isFmEqKernel) {
    return Conv3DBp2MmMode::CONV3D_BP_MM_FEATURE_MAP_EQ_KERNEL;
  }
  return Conv3DBp2MmMode::CONV3D_BP_NO_MM;
}

static int64_t CalcCountByAxisVec(const op::Shape &dataShape, const vector<int64_t> &axisVec)
{
  int64_t count = 1;
  for (auto axis : axisVec) {
    count *= dataShape[axis];
  }
  return count;
}

static const aclTensor *ViewWithShape(const aclTensor *tensor, const op::Shape &shape, aclOpExecutor *executor)
{
  if (shape == tensor->GetViewShape() && shape == tensor->GetStorageShape()) {
    return tensor;
  }
  return executor->CreateView(tensor, shape, tensor->GetViewOffset());
}

static const aclTensor *ViewWithShapeAndReformatND(const aclTensor *tensor, const std::initializer_list<int64_t> &shape,
  aclOpExecutor *executor)
{
  op::Shape shape2d = op::Shape(shape);
  auto tensor2d = ViewWithShape(tensor, shape2d, executor);
  CHECK_RET(tensor2d != nullptr, nullptr);
  return l0op::ReFormat(tensor2d, op::Format::FORMAT_ND);
}

static aclnnStatus GenDxInOutByConvBp2MmMode(BatchMatmulInput &batchMmInput,
                                             ConvolutionBackwardInputTensor &inputTensor,
                                             ConvolutionBackwardResult &outputTensor,
                                             aclOpExecutor *executor,
                                             Conv3DBp2MmMode conv2MmMode)
{
  auto gradOutput = inputTensor.gradOutput;
  auto weight = inputTensor.weight;
  auto gradInput = outputTensor.gradInput;
  vector<int64_t> gradShapeVec {0, 0, 0};
  const int64_t bDimBMNIdx = 0; // batch in [batch, m, n] for mamtul
  const int64_t mDimBMNIdx = 1; // m in [batch, m, n] for mamtul
  const int64_t nDimBMNIdx = 2; // n in [batch, m, n] for mamtul
  const vector<int64_t> dhwIdxUnionVec {dDimNCDHWIdx, hDimNCDHWIdx, wDimNCDHWIdx};
  const vector<int64_t> cidhwIdxUnionVec {ciDimCoCiDHWIdx, dDimNCDHWIdx, hDimNCDHWIdx, wDimNCDHWIdx};
  // input : gradOutput
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ||
    conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL) {
    gradShapeVec = {gradOutput->GetViewShape()[nDimNCDHWIdx], gradOutput->GetViewShape()[cDimNCDHWIdx],
      CalcCountByAxisVec(gradOutput->GetViewShape(), dhwIdxUnionVec)};
  } else if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_FEATURE_MAP_EQ_KERNEL) {
    gradShapeVec = {gradOutput->GetViewShape()[nDimNCDHWIdx], 1, gradOutput->GetViewShape()[1]};
  }
  auto gradOutputND = ViewWithShapeAndReformatND(gradOutput,
    {gradShapeVec[bDimBMNIdx], gradShapeVec[mDimBMNIdx], gradShapeVec[nDimBMNIdx]}, executor);
  CHECK_RET(gradOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // input : weight -- 1, Co, CinDHW
  std::initializer_list<int64_t> weightShape {1, weight->GetViewShape()[coDimCoCiDHWIdx],
    CalcCountByAxisVec(weight->GetViewShape(), cidhwIdxUnionVec)};
  auto weightND = ViewWithShapeAndReformatND(weight, weightShape, executor);
  CHECK_RET(weightND != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // output
  vector<int64_t> outShapeVec {0, 0, 0};
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL) {
    // CONV3D_BP_MM_STRIDE_EQ_KERNEL's matmul output is N, DoHoWo, CinDkHkWk
    outShapeVec = {gradInput->GetViewShape()[nDimNCDHWIdx],
      CalcCountByAxisVec(gradOutput->GetViewShape(), dhwIdxUnionVec),
      CalcCountByAxisVec(weight->GetViewShape(), cidhwIdxUnionVec)};
  } else {
    outShapeVec = {gradInput->GetViewShape()[nDimNCDHWIdx],
      gradInput->GetViewShape()[cDimNCDHWIdx],
      CalcCountByAxisVec(gradInput->GetViewShape(), dhwIdxUnionVec)};
  }
  auto mmDxOutputND = ViewWithShapeAndReformatND(gradInput,
    {outShapeVec[bDimBMNIdx], outShapeVec[mDimBMNIdx], outShapeVec[nDimBMNIdx]}, executor);
  CHECK_RET(mmDxOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);

  batchMmInput.leftData = conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ? weightND : gradOutputND;
  batchMmInput.isLeftTranspose = conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ||
    conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL;
  batchMmInput.rightData = conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ? gradOutputND : weightND;
  batchMmInput.isRightTranspose = false;
  batchMmInput.outputData = mmDxOutputND;
  return ACLNN_SUCCESS;
}

static const aclTensor *DoPostMatmulForConv3dBpInput(const aclTensor *gradInputND,
  ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardResult &outputTensor,
  aclOpExecutor *executor, Conv3DBp2MmMode conv2MmMode)
{
  // permute : to recover data arrangement : NDoHoWoCinDkHkWk -> NCinDoDkHoHkWoWk
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL) {
    const auto &gradOutputShape = inputTensor.gradOutput->GetViewShape();
    const auto &weightShape = inputTensor.weight->GetViewShape();
    op::Shape mmOutputExtendShape = op::Shape({
      gradOutputShape[0], gradOutputShape[dDimNCDHWIdx], gradOutputShape[hDimNCDHWIdx], gradOutputShape[wDimNCDHWIdx],
      weightShape[ciDimCoCiDHWIdx], weightShape[dDimNCDHWIdx], weightShape[hDimNCDHWIdx], weightShape[wDimNCDHWIdx],
    });
    auto mmOutputExtendND = ViewWithShape(gradInputND, mmOutputExtendShape, executor);
    CHECK_RET(mmOutputExtendND != nullptr, nullptr);
    std::vector<int64_t> permDimVec {0, 4, 1, 5, 2, 6, 3, 7}; // NDoHoWoCinDkHkWk -> NCinDoDkHoHkWoWk
    gradInputND = Permute(mmOutputExtendND, permDimVec, executor);
  }
  // ND -> NCDHW
  CHECK_RET(gradInputND != nullptr, nullptr);
  auto gradInputNCDHW = ViewWithShape(gradInputND, outputTensor.gradInput->GetViewShape(), executor);
  CHECK_RET(gradInputNCDHW != nullptr, nullptr);
  gradInputNCDHW = l0op::ReFormat(gradInputNCDHW, op::Format::FORMAT_NCDHW);
  return gradInputNCDHW;
}

// FM=KERNEL
static aclnnStatus GenConvMmDwInputByMode(BatchMatmulInput &batchMmInput,
                                       ConvolutionBackwardInputTensor &inputTensor,
                                       aclOpExecutor *executor,
                                       Conv3DBp2MmMode conv2MmMode)
{
  auto gradOutput = inputTensor.gradOutput; // dy
  auto input = inputTensor.input; //x
  auto weight = inputTensor.weight; 
  op::Shape gradOutputShape2d = op::Shape({
    1, 
    gradOutput->GetViewShape()[0],
    CalcCountByAxisVec(gradOutput->GetViewShape(), {1, 2, 3, 4})});
  auto gradOutput2d = ViewWithShape(gradOutput, gradOutputShape2d, executor); 
  CHECK_RET(gradOutput2d != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradOutputND = l0op::ReFormat(gradOutput2d, op::Format::FORMAT_ND);

  op::Shape inputShape2d = op::Shape({
      1,
      input->GetViewShape()[0],
      CalcCountByAxisVec(input->GetViewShape(), {1, 2, 3, 4})});
  auto input2d = ViewWithShape(input, inputShape2d, executor);
  CHECK_RET(input2d != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto inputND = l0op::ReFormat(input2d, op::Format::FORMAT_ND);

  batchMmInput.leftData = gradOutputND;
  batchMmInput.isLeftTranspose = true;
  batchMmInput.rightData = inputND;
  batchMmInput.isRightTranspose = false;
  return ACLNN_SUCCESS;
}

static aclnnStatus GenConvMmDwOutputByMode(aclTensor *&mmDwOutputNDFp32,
                                       ConvolutionBackwardResult &outputTensor,
                                       aclOpExecutor *executor,
                                       Conv3DBp2MmMode conv2MmMode)
{   
    auto gradInput = outputTensor.gradInput;
    auto gradWeight = outputTensor.gradWeight;
    op::Shape mmDwOutShape2d = op::Shape({
      1,
      gradWeight->GetViewShape()[0],  //cout
      CalcCountByAxisVec(gradInput->GetViewShape(), {1, 2, 3, 4})});
    auto mmDwOutput2d = ViewWithShape(gradInput, mmDwOutShape2d, executor);
    CHECK_RET(mmDwOutput2d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mmDwOutputND = l0op::ReFormat(mmDwOutput2d, op::Format::FORMAT_ND);
    mmDwOutputNDFp32 = executor->CreateView(mmDwOutputND, mmDwOutputND->GetViewShape(),
      mmDwOutputND->GetViewOffset());
    CHECK_RET(mmDwOutputNDFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    mmDwOutputNDFp32->SetDataType(DataType::DT_FLOAT);
    
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv3DBackwardDwByMmMode(ConvolutionBackwardInputTensor &inputTensor,
                                                   ConvolutionBackwardResult &outputTensor,
                                                   ConvolutionBackwardParams &params,
                                                   aclOpExecutor *executor,
                                                   Conv3DBp2MmMode conv2MmMode)
{
  BatchMatmulInput batchMmInput;
  auto status = GenConvMmDwInputByMode(batchMmInput, inputTensor, executor, conv2MmMode);
  if (status != ACLNN_SUCCESS) {
    return status;
  }
  OP_LOGD("Enter backprop filter Calculate with matmul mode");
  aclTensor *mmDwOutputNDFp32 = nullptr;
  status = GenConvMmDwOutputByMode(mmDwOutputNDFp32, outputTensor, executor, conv2MmMode);
  if (status != ACLNN_SUCCESS) {
    OP_LOGD("GenConvMmDwOutputByMode False");
    return status;
  }
  auto gradWeightNND = ExecBatchMatmulOp(batchMmInput.leftData, batchMmInput.rightData, mmDwOutputNDFp32, batchMmInput.isLeftTranspose,
    batchMmInput.isRightTranspose, params.cubeMathType, executor);
  OP_CHECK(gradWeightNND != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ExecBatchMatmulOp for 3ddw return nullptr."),
            return ACLNN_ERR_INNER_NULLPTR);

  auto gradWeightNCDHW = ViewWithShape(gradWeightNND, outputTensor.gradWeight->GetViewShape(), executor);
  CHECK_RET(gradWeightNCDHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
  gradWeightNCDHW = l0op::ReFormat(gradWeightNCDHW, op::Format::FORMAT_NCDHW);

  OP_LOGD("gradWeightNCDHW: %s", gradWeightNCDHW->ToString().GetString());
  CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightNCDHW, "gradWeight", params.groups, executor) ==
            ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv3DBackwardByMatmulImpl(ConvolutionBackwardInputTensor &inputTensor,
                                                       ConvolutionBackwardResult &outputTensor,
                                                       ConvolutionBackwardParams &params,
                                                       aclOpExecutor *executor,
                                                       vector<bool> &conv3DBp2MatmulMask)
{
  OP_LOGD("Enter CalculateConv3DBackwardByMatmulImpl");
  auto conv2MmMode = GetConv3DBp2MmMode(inputTensor, params);
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_NO_MM) {
    return ACLNN_SUCCESS;
  }
  if ((*params.outputMask)[0]) {
    BatchMatmulInput batchMmInput;
    auto status = GenDxInOutByConvBp2MmMode(batchMmInput, inputTensor, outputTensor, executor, conv2MmMode);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    OP_LOGD("Enter Conv3DBackpropInput Calculation By Matmul Implementation.");
    auto gradInputND = ExecBatchMatmulOp(batchMmInput.leftData, batchMmInput.rightData, batchMmInput.outputData,
      batchMmInput.isLeftTranspose, batchMmInput.isRightTranspose, params.cubeMathType, executor);
    OP_CHECK(
        gradInputND != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The Mamtul In Conv3DBackpropInput Return Nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    auto gradInputNCDHW = DoPostMatmulForConv3dBpInput(gradInputND, inputTensor, outputTensor, executor, conv2MmMode);
    CHECK_RET(gradInputNCDHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
    status = OutputPostProcess(outputTensor.gradInput, gradInputNCDHW, "gradInput", params.groups, executor);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    conv3DBp2MatmulMask[0] = true;
  }
    if ((*params.outputMask)[1]) {
    auto status = CalculateConv3DBackwardDwByMmMode(inputTensor, outputTensor, params, executor, conv2MmMode);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    conv3DBp2MatmulMask[1] = true;
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv3DBackward(ConvolutionBackwardInputTensor &inputTensor,
                                           ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                           aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

  vector<bool> conv3DBp2MatmulMask {false, false};
  auto mmStatus = CalculateConv3DBackwardByMatmulImpl(inputTensor, outputTensor, params, executor, conv3DBp2MatmulMask);
  CHECK_RET(mmStatus == ACLNN_SUCCESS, mmStatus);
  // 如果BpInput和BpFilter都用matmul做完，则提前返回，避免transData耗时.
  if ((*params.outputMask)[0] == conv3DBp2MatmulMask[0] && (*params.outputMask)[1] == conv3DBp2MatmulMask[1]) {
    return ACLNN_SUCCESS;
  }

  l0op::ConvBackpropParams conv3DBackpropPrarams = {inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride, params.padding, params.dilation, params.groups};
  bool useV2Flag = l0op::IsConv3DBackpropInputV2(conv3DBackpropPrarams);
  bool inputTransDataFlag = !((*params.outputMask)[1] && l0op::IsConv3DBackpropFilterV2(conv3DBackpropPrarams) && l0op::IsInputTransdataWhiteListCase(conv3DBackpropPrarams)); // dw、v2、白名单同时满足，输入x融合transdata;
  OP_LOGD("Input trans data flag is %d", inputTransDataFlag ? 1 : 0);

  auto promoteType = CalcPromoteType(inputTensor);
  CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor, false, inputTransDataFlag) == ACLNN_SUCCESS,
                ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());

  bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  bool outputTransdataFlag = true;

  if ((*params.outputMask)[0] && !conv3DBp2MatmulMask[0]) {
    OP_LOGD("Enter dx Calculate");
    const aclTensor *gradInputNDC1HWC0 = nullptr;
    if (useV2Flag && !useHf32 && inputTensor.weight->GetDataType() != DataType::DT_FLOAT) {
      outputTransdataFlag = false; // V2，非HF32，非FP32，不在黑名单
    }
    if (useHf32) {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputHf32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputFp322Fp32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputBf162Bf16(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    } else {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputFp162Fp16(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    }
    OP_CHECK(
        gradInputNDC1HWC0 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The calculation with empty tensor failed, Conv3DBackpropInput return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    aclnnStatus status = outputTransdataFlag ? OutputPostProcess(outputTensor.gradInput, gradInputNDC1HWC0, "gradInput", params.groups, executor)
      : OutputPostProcessWithoutTransdata(outputTensor.gradInput, gradInputNDC1HWC0, "gradInput", executor);
    CHECK_RET(status == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  }

  if ((*params.outputMask)[1] && !conv3DBp2MatmulMask[1]) {
    OP_LOGD("Enter dw Calculate");
    const aclTensor *gradWeightFZ3D = nullptr;
    if (useHf32) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterHf32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                         params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_FLOAT) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp322Fp32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                         params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterBf162Fp32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                         params.padding, params.dilation, params.groups, executor);
    } else {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp162Fp32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                         params.padding, params.dilation, params.groups, executor);
    }

    OP_CHECK(gradWeightFZ3D != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropInput return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ3D, "gradWeight", params.groups, executor) ==
              ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv3DTransposeBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  DataType promoteType;
  bool useHf32 = false;
  if ((*params.outputMask)[1]) {
    promoteType = CalcPromoteType(inputTensor);
    CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  } else if ((*params.outputMask)[0]) {
    promoteType = CalcPromoteType(inputTensor);
    CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  }
  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());
  if ((*params.outputMask)[0]) {
    OP_LOGD("Enter dx Calculate");
    const aclTensor *gradInputNDC1HWC0 = nullptr;
    if (useHf32) {
      gradInputNDC1HWC0 = l0op::Conv3d6HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, true, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
      gradInputNDC1HWC0 = l0op::Conv3d6HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, false, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradInputNDC1HWC0 = l0op::Conv3d6HdBf16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    } else {
      gradInputNDC1HWC0 = l0op::Conv3d6HdFp16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    }
    OP_CHECK(
        gradInputNDC1HWC0 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The calculation with empty tensor failed, Conv2d5HdFp16 return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradInput, gradInputNDC1HWC0, "gradInput", params.groups, executor) ==
              ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }

  if ((*params.outputMask)[1]) {
    OP_LOGD("Enter dw Calculate");
    const aclTensor *gradWeightFZ3D = nullptr;
    if (useHf32) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterHf32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input, params.stride,
                                         params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_FLOAT) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp322Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterBf162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    } else {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    }

    OP_CHECK(gradWeightFZ3D != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropFilter return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ3D, "gradWeight", params.groups, executor) ==
              ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConvolutionBackwardWithEmpty(ConvolutionBackwardInputTensor &inputTensor,
                                                         ConvolutionBackwardOutput &outputTensor,
                                                         ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  // Index 为 1：进行gradWeight空tensor运算
  if ((*params.outputMask)[1]) {
    auto weightContiguous = l0op::Contiguous(inputTensor.weight, executor);
    auto gradWeightZeros = l0op::ZerosLike(weightContiguous, executor);
    OP_CHECK(gradWeightZeros != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, weight with ZerosLike return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::ViewCopy(gradWeightZeros, outputTensor.gradWeight, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, weight with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }

  // Index 为 2：进行bias空tensor运算
  if ((*params.outputMask)[2]) {
    op::Shape biasGradShape = {(*params.biasSizes)[0]};
    auto biasTensor = executor->AllocTensor(biasGradShape, inputTensor.weight->GetDataType());
    const_cast<aclTensor *>(biasTensor)->SetStorageFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor *>(biasTensor)->SetViewFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor *>(biasTensor)->SetOriginalFormat(op::Format::FORMAT_ND);

    auto gradBiasZeros = l0op::ZerosLike(biasTensor, executor);
    OP_CHECK(gradBiasZeros != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, bias with ZerosLike return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ViewCopy(gradBiasZeros, outputTensor.gradBias, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, bias with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckCubeMathTypeFor3D(ConvolutionBackwardInputTensor &inputTensor,
  const ConvolutionBackwardParams &params)
{
  DataType promoteDtype = CalcPromoteType(inputTensor);
  int8_t cubeMathType = params.cubeMathType;
  if (cubeMathType == USE_FP16 && promoteDtype == DataType::DT_BF16) {
    OP_LOGW("When promoteDtype is bfloat16, the cubeMathType can not be USE_FP16.");
    return ACLNN_SUCCESS;
  }
  if (cubeMathType == USE_HF32 && promoteDtype == DataType::DT_BF16) {
    OP_LOGW("When promoteDtype is bfloat16, the cubeMathType can not be USE_HF32.");
    return ACLNN_SUCCESS;
  }
  if (cubeMathType == USE_HF32 && promoteDtype == DataType::DT_FLOAT16) {
    OP_LOGW("When promoteDtype is float16, the cubeMathType can not be USE_HF32.");
    return ACLNN_SUCCESS;
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConvolutionBackward(ConvolutionBackwardInputTensor &inputTensor,
  ConvolutionBackwardOutput &outputTensor, ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  ConvolutionBackwardResult resultTensor = {outputTensor.gradInput, outputTensor.gradWeight, outputTensor.gradBias};
  auto inputDim = inputTensor.input->GetViewShape().GetDimNum();
  // inputDim 为 3 时，进行1D卷积反向
  if (inputDim == CONV1DINPUTDIM) {
    if (!params.transposed) {
      OP_LOGD("Entering CalculateConv1DBackward");
      auto ret = CalculateConv1DBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    } else {
      OP_LOGD("Entering CalculateConv1DTransposeBackward");
      auto ret = CalculateConv1DTransposeBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    // inputDim 为 4 时，进行2D卷积反向
  } else if (inputDim == CONV2DINPUTDIM) {
    if (!params.transposed) {
      OP_LOGD("Entering CalculateConv2DBackward");
      auto ret = CalculateConv2DBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    } else {
      OP_LOGD("Entering CalculateConv2DTransposeBackward");
      auto ret = CalculateConv2DTransposeBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    // inputDim 为 5 时，进行3D卷积反向
  } else if (inputDim == CONV3DINPUTDIM) {
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not implemented for %s", op::ToString(socVersion).GetString()),
             return ACLNN_ERR_PARAM_INVALID);
    auto ret = CheckCubeMathTypeFor3D(inputTensor, params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSupportedForConv3dBackpropFilter(inputTensor, outputTensor, params), ACLNN_ERR_PARAM_INVALID);
    if (!params.transposed) {
      OP_LOGD("Entering CalculateConv3DBackward");
      ret = CalculateConv3DBackward(inputTensor, resultTensor, params, executor);
    } else {
      OP_LOGD("Entering CalculateConv3DTransposeBackward");
      ret = CalculateConv3DTransposeBackward(inputTensor, resultTensor, params, executor);
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "ConvolutionBackward only support input with dimensions 3, 4, or 5, Actually is %ld.", inputDim);
    return ACLNN_ERR_PARAM_INVALID;
  }
  auto ret = OutputViewProcess(resultTensor, outputTensor, params.outputMask, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("After CalculateConvolutionBackward");
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConvolutionTbcBackwardWithEmpty(aclTensor *&gradInput, aclTensor *&gradWeight,
                                                            aclTensor *&gradBias, const int64_t pad,
                                                            aclOpExecutor *executor) {
  if (gradInput->Size() != 0) {
    auto inputValueScalar = executor->AllocScalar(0);
    auto inputValueTensor = executor->ConvertToTensor(inputValueScalar, gradInput->GetDataType());
    auto inputFillShape = op::ToShapeVector(gradInput->GetViewShape());
    aclIntArray *inputShapeArray = executor->AllocIntArray(inputFillShape.data(), inputFillShape.size());
    const aclTensor *inputDims = executor->ConvertToTensor(inputFillShape.data(),
                                                           inputFillShape.size(), op::DataType::DT_INT64);
    auto gradInputZeros = l0op::Fill(inputDims, inputValueTensor, inputShapeArray, executor);
    CHECK_RET(gradInputZeros != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradInputResult = l0op::ViewCopy(gradInputZeros, gradInput, executor);
    CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (gradWeight->Size() != 0) {
    auto weightValueScalar = executor->AllocScalar(0);
    auto weightValueTensor = executor->ConvertToTensor(weightValueScalar, gradWeight->GetDataType());
    auto weightFillShape = op::ToShapeVector(gradWeight->GetViewShape());
    aclIntArray *weightShapeArray = executor->AllocIntArray(weightFillShape.data(), weightFillShape.size());
    const aclTensor *weightDims = executor->ConvertToTensor(weightFillShape.data(),
                                                           weightFillShape.size(), op::DataType::DT_INT64);
    auto gradWeightZeros = l0op::Fill(weightDims, weightValueTensor, weightShapeArray, executor);
    CHECK_RET(gradWeightZeros != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradWeightResult = l0op::ViewCopy(gradWeightZeros, gradWeight, executor);
    CHECK_RET(gradWeightResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (gradBias->Size() != 0) {
    auto t = gradInput->GetViewShape().GetDim(0) + 2 * pad + 1 - gradWeight->GetViewShape().GetDim(0);
    float value = 1.0f * gradInput->GetViewShape().GetDim(1) * t;
    auto valueScalar = executor->AllocScalar(value);
    auto valueTensor = executor->ConvertToTensor(valueScalar, gradBias->GetDataType());
    auto fillShape = op::ToShapeVector(gradBias->GetOriginalShape());
    aclIntArray *shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclTensor *dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    auto gradBiasOnes = l0op::Fill(dims, valueTensor, shapeArray, executor);
    CHECK_RET(gradBiasOnes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradBiasResult = l0op::ViewCopy(gradBiasOnes, gradBias, executor);
    CHECK_RET(gradBiasResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConvolutionTbcBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                   ConvolutionBackwardOutput &finalTensor,
                                                   ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  aclTensor *gradInputNcl = executor->AllocTensor(inputTensor.input->GetViewShape(), inputTensor.input->GetDataType());
  aclTensor *gradWeightNcl = executor->AllocTensor(inputTensor.weight->GetViewShape(),
                                                   inputTensor.weight->GetDataType());
  aclTensor *gradBiasNcl = executor->AllocTensor(finalTensor.gradBias->GetViewShape(),
                                                 finalTensor.gradBias->GetDataType());
  ConvolutionBackwardResult resultTensor = {gradInputNcl, gradWeightNcl, gradBiasNcl};
  auto inputDim = inputTensor.input->GetViewShape().GetDimNum();
  // inputDim 为 3 时，进行1D卷积反向
  if (inputDim == 3) {
    if (!params.transposed) {
      OP_LOGD("Entering CalculateConv1DBackward");
      auto ret = CalculateConv1DBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    } else {
      OP_LOGD("Entering CalculateConv1DTransposeBackward");
      auto ret = CalculateConv1DTransposeBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    }
  } else {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "ConvolutionTbcBackward only support input with dimensions 3, Actually is %ld.", inputDim);
    return ACLNN_ERR_INNER_NULLPTR;
  }
  // recover NCL to TBC
  vector<int64_t> gradInputDims = {2, 0, 1};
  vector<int64_t> weightDims = {2, 1, 0};
  auto gradInputTbc = Permute(resultTensor.gradInput, gradInputDims, executor);
  CHECK_RET(gradInputTbc != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradWeightTbc = Permute(resultTensor.gradWeight, weightDims, executor);
  CHECK_RET(gradWeightTbc != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradBiasTbc = resultTensor.gradBias;
  // transdata: change result format --> out format
  if (gradInputTbc->GetStorageFormat() != finalTensor.gradInput->GetStorageFormat()) {
    gradInputTbc = l0op::ReFormat(gradInputTbc, finalTensor.gradInput->GetStorageFormat());
  }
  if (gradWeightTbc->GetStorageFormat() != finalTensor.gradWeight->GetStorageFormat()) {
    gradWeightTbc = l0op::ReFormat(gradWeightTbc, finalTensor.gradWeight->GetStorageFormat());
  }

  // viewcopy
  auto gradInputResult = l0op::ViewCopy(gradInputTbc, finalTensor.gradInput, executor);
  auto gradWeightResult = l0op::ViewCopy(gradWeightTbc, finalTensor.gradWeight, executor);
  auto gradBiasResult = l0op::ViewCopy(gradBiasTbc, finalTensor.gradBias, executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(
    const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight, const aclIntArray *biasSizes,
    const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed,
    const aclIntArray *outputPadding, int groups, const aclBoolArray *outputMask, int8_t cubeMathType,
    aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnConvolutionBackward,
                   DFX_IN(gradOutput, input, weight, biasSizes, stride, padding, dilation, transposed, outputPadding,
                          groups, outputMask, cubeMathType),
                   DFX_OUT(gradInput, gradWeight, gradBias));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ConvolutionBackwardInputTensor inputTensor = {gradOutput, input, weight};
    ConvolutionBackwardOutput outputTensor = {gradInput, gradWeight, gradBias};
    ConvolutionBackwardParams params = {biasSizes,     stride, padding,    dilation,    transposed,
                                        outputPadding, groups, outputMask, cubeMathType};

    // 固定写法，参数检查
    auto ret = CheckParams(inputTensor, outputTensor, params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查conv3ddw确定性计算
    if ((*outputMask)[1] && (input->GetViewShape().GetDimNum() == CONV3DINPUTDIM)) {
      int64_t deterministicValue = 0;
      rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
      if (retRts != RT_ERROR_NONE) {
        deterministicValue = 0;
      }
      CHECK_RET(CheckDeterministic(deterministicValue, groups), ACLNN_ERR_PARAM_INVALID);
    }

    if (input->Size() == 0 || weight->Size() == 0 || gradOutput->Size() == 0) {
        OP_LOGD("Entering CalculateConvolutionBackwardWithEmpty");
        CHECK_RET(CheckEmptyTensor(inputTensor, params), ACLNN_ERR_PARAM_INVALID);
        ret = CalculateConvolutionBackwardWithEmpty(inputTensor, outputTensor, params, uniqueExecutor.get());
    } else {
        OP_LOGD("Entering CalculateConvolutionBackward");
        ret = CalculateConvolutionBackward(inputTensor, outputTensor, params, uniqueExecutor.get());
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(const aclTensor *self, const aclTensor *input, const aclTensor *weight,
                                                 const aclTensor *bias, int64_t pad, int8_t cubeMathType,
                                                 aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnConvTbcBackward, DFX_IN(self, input, weight, bias, pad, cubeMathType),
                   DFX_OUT(gradInput, gradWeight, gradBias));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckTbcParams(self, input, weight, bias, pad, cubeMathType, gradInput, gradWeight, gradBias);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 设置param
    FVector<int64_t> newStride = {1};
    FVector<int64_t> newPadding = {pad};
    FVector<int64_t> newDilation = {1};
    FVector<int64_t> newOutputPadding = {0};
    FVector<bool> newOutputMask = {1, 1, 1};
    FVector<int64_t> newBias = {bias->Size()};
    auto *stride = uniqueExecutor.get()->AllocIntArray(newStride.data(), 1);
    auto *padding = uniqueExecutor.get()->AllocIntArray(newPadding.data(), 1);
    auto *dilation = uniqueExecutor.get()->AllocIntArray(newDilation.data(), 1);
    const bool transposed = 0;
    auto *outputPadding = uniqueExecutor.get()->AllocIntArray(newOutputPadding.data(), 1);
    int64_t groups = 1;
    auto *outputMask = uniqueExecutor.get()->AllocBoolArray(newOutputMask.data(), 3);
    auto biasSizes = uniqueExecutor.get()->AllocIntArray(newBias.data(), 1);

    if (input->Size() == 0 || weight->Size() == 0 || bias->Size() == 0) {
        OP_LOGD("Entering CalculateConvolutionTbcBackwardWithEmpty");
        ret = CalculateConvolutionTbcBackwardWithEmpty(gradInput, gradWeight, gradBias, pad, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        // transpose TBC to NCL
        vector<int64_t> dims = {1, 2, 0};
        vector<int64_t> weightDims = {2, 1, 0};
        auto selfPermuted = Permute(self, dims, uniqueExecutor.get());
        CHECK_RET(selfPermuted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputPermuted = Permute(input, dims, uniqueExecutor.get());
        CHECK_RET(inputPermuted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto weightPermuted = Permute(weight, weightDims, uniqueExecutor.get());
        CHECK_RET(weightPermuted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        ConvolutionBackwardInputTensor inputTensorNcl = {selfPermuted, inputPermuted, weightPermuted};
        ConvolutionBackwardParams params = {biasSizes,     stride, padding,    dilation,    transposed,
                                            outputPadding, groups, outputMask, cubeMathType};
        ConvolutionBackwardOutput finalTensor = {gradInput, gradWeight, gradBias};
        OP_LOGD("Entering CalculateConvolutionTbcBackward");
        ret = CalculateConvolutionTbcBackward(inputTensorNcl, finalTensor, params, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvolutionBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConvolutionBackward);
    OP_LOGD("Entering aclnnConvolutionBackward");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnConvTbcBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConvTbcBackward);
    OP_LOGD("Entering aclnnConvTbcBackward");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
