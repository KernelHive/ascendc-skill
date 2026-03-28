/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "resize_grad_d.h"
#include "upsample_bicubic2d_grad_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_upsample_bicubic_2d_backward.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "runtime/context.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// According to the API definition, all supported dtypes need to be listed
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910 = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {op::DataType::DT_BF16,
                                                                           op::DataType::DT_FLOAT16,
                                                                           op::DataType::DT_FLOAT};

static const int64_t DIM_LIMIT = 4;
static const int64_t EXPECT_SIZE = 2;
static const int64_t FOURDIMS = 4;
static const int64_t FP16_DETERMINSTIC_CAST = 1;
static const int64_t BF16_CAST = 2;
static const float MAX_SCALE = 50;

static const std::string CUBIC_MODE = "cubic";

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

// 循环判断size是否相等
static bool CheckSizeLoop(size_t dimNum, Shape gradOutShape, FVector<int64_t> &fullOutputSize)
{
    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected grad_output to have the same shape as output;"
                " output.size(%zu) = %ld but got grad_output.size(%zu) = %ld",
                i,
                fullOutputSize[i],
                i,
                gradOutShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static bool CheckNotNull(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, const aclTensor *gradInput) {
  OP_CHECK_NULL(inputSize, return false);
  OP_CHECK_NULL(gradInput, return false);
  OP_CHECK_NULL(gradOut, return false);
  OP_CHECK_NULL(outputSize, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *gradInput) {
  // Check if the data type of gradOut is in the supported list of
  // the ResizeNearestNeighborV2Grad operator
  bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E &&
                           GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B);
  const std::initializer_list<op::DataType> dtypeSupportList =
        is910BSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;

  OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, dtypeSupportList, return false);

  // Check that the data type of gradInput is consistent with gradOut
  OP_CHECK_DTYPE_NOT_MATCH(gradOut, gradInput->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize) {
  // NCHW和ND格式判断
  auto srcOriginFormat = gradOut->GetOriginalFormat();
  bool supportFormat = srcOriginFormat == op::Format::FORMAT_ND || srcOriginFormat == op::Format::FORMAT_NCHW;
  OP_CHECK(supportFormat, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input format only support NCHW or ND"), return false);
  size_t inputSizeNum = inputSize->Size();
  size_t outputSizeNum = outputSize->Size();
  OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);

  OP_CHECK(inputSizeNum == DIM_LIMIT,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input_size equals to 4, but got size %zu", inputSizeNum),
           return false);
  OP_CHECK(outputSizeNum == EXPECT_SIZE,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 2, but got size %zu", outputSizeNum),
           return false);
  
  return true;
}

static bool CheckInputElement(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize) {
  int64_t batch = (*inputSize)[DIM_ZERO];
  int64_t channels = (*inputSize)[DIM_ONE];
  int64_t inputH = (*inputSize)[DIM_TWO];
  int64_t inputW = (*inputSize)[DIM_THREE];
  int64_t outH = (*outputSize)[DIM_ZERO];
  int64_t outW = (*outputSize)[DIM_ONE];
  auto gradOutShape = gradOut->GetViewShape();
  size_t dimNum = gradOutShape.GetDimNum();
  FVector<int64_t> fullOutputSize = {batch, channels, outH, outW};

  OP_CHECK(inputW > 0 && inputH > 0 && outW > 0 && outH > 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input and output sizes should greater than 0, bug got input (H: %ld,"
                   " W: %ld) output (H: %ld, W: %ld)", inputH, inputW, outH, outW),
          return false);

  auto res = CheckSizeLoop(dimNum, gradOutShape, fullOutputSize);
  CHECK_RET(res == true, res);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, const aclTensor *gradInput) {
  // check input args
  CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_NULLPTR);

  // check input element
  CHECK_RET(CheckInputElement(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

  // check input shape
  CHECK_RET(CheckShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

  // check input dtype
  CHECK_RET(CheckDtypeValid(gradOut, gradInput), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor *ContiguousCast(const aclTensor *input, op::DataType dstDtype, aclOpExecutor *executor)
{
  auto inputContiguous = l0op::Contiguous(input, executor);
  auto outputContiguous = l0op::Cast(inputContiguous, dstDtype, executor);
  return outputContiguous;
}

static const aclTensor *adaptOutput(const aclTensor *gradOut, const aclTensor * ResizeGradDOut,
                                         aclIntArray *permuteHWNCArray, aclOpExecutor *executor,
                                         const int64_t *newReverseShape, int64_t castFp32Condition) {
  const aclTensor *resizeGradDOutContiguous = nullptr;
  if (castFp32Condition == FP16_DETERMINSTIC_CAST) {
    resizeGradDOutContiguous = ContiguousCast(ResizeGradDOut, op::DataType::DT_FLOAT16, executor);
  } else if (castFp32Condition == BF16_CAST) {
    resizeGradDOutContiguous = ContiguousCast(ResizeGradDOut, op::DataType::DT_BF16, executor);
  } else {
    resizeGradDOutContiguous = l0op::Contiguous(ResizeGradDOut, executor);
  }
  CHECK_RET(resizeGradDOutContiguous != nullptr, nullptr);

  aclIntArray* shapeArrayReverse = executor->AllocIntArray(newReverseShape, 4);
  resizeGradDOutContiguous = l0op::Reshape(resizeGradDOutContiguous, shapeArrayReverse, executor);
  CHECK_RET(resizeGradDOutContiguous != nullptr, nullptr);

  // ResizeGradDOut transpose
  auto resizeGradDOutTranspose = l0op::Transpose(resizeGradDOutContiguous, permuteHWNCArray, executor);
  CHECK_RET(resizeGradDOutTranspose != nullptr, nullptr);

  return resizeGradDOutTranspose;
}

aclnnStatus aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclIntArray *outputSize,
                                                           const aclIntArray *inputSize, const bool alignCorners,
                                                           double scalesH, double scalesW, aclTensor *gradInput,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnUpsampleBicubic2dBackward, DFX_IN(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW), DFX_OUT(gradInput));
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CheckParams(gradOut, outputSize, inputSize, gradInput);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (gradOut->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto computeScale = [](double scale, int64_t inputSizes, 
    int64_t outputSizes, bool alignCorner) -> float {
    if(alignCorner){
      if(outputSizes > 1) { return static_cast<double>(inputSizes - 1) / static_cast<double>(outputSizes - 1); } 
      else {
        return 0;
      }
    }
    if(scale > 0){
      return 1 / scale;
    } else {
      return static_cast<double>(inputSizes) / static_cast<double>(outputSizes);
    }
  };
  const float scalesWidth = computeScale(
    scalesW, (*inputSize)[DIM_THREE], (*outputSize)[DIM_ONE], alignCorners);
  const float scalesHeight = computeScale(
    scalesH, (*inputSize)[DIM_TWO], (*outputSize)[DIM_ZERO], alignCorners);

  if (GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E &&
      GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      CheckType(gradOut->GetDataType(), DTYPE_SUPPORT_LIST_910B) &&
      scalesWidth < MAX_SCALE && scalesWidth >= (1 / MAX_SCALE) &&
      scalesHeight < MAX_SCALE && scalesHeight >= (1 / MAX_SCALE)) {
    const aclTensor* gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if ((*inputSize)[DIM_TWO] == (*outputSize)[DIM_ZERO] && (*inputSize)[DIM_THREE] == (*outputSize)[DIM_ONE]) {
        auto viewCopyResult = l0op::ViewCopy(gradOutContiguous, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 将fp16/bf16类型cast成fp32处理，保证精度
        const aclTensor* gradInputFp32 = l0op::Contiguous(gradInput, uniqueExecutor.get());
        CHECK_RET(gradInputFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto dtype = gradOut->GetDataType();
        if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
          gradOutContiguous = l0op::Cast(gradOutContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
          gradInputFp32 = l0op::Cast(gradInputFp32, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor *upsampleBicubic2dGradOut = 
          l0op::UpsampleBicubic2dGrad(gradOutContiguous, alignCorners, scalesHeight, 
          scalesWidth, gradInputFp32, uniqueExecutor.get());
        CHECK_RET(upsampleBicubic2dGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // CAST回bf16
        if (dtype == op::DataType::DT_BF16) {
          upsampleBicubic2dGradOut = l0op::Cast(upsampleBicubic2dGradOut, op::DataType::DT_BF16, uniqueExecutor.get());
        } else if (dtype == op::DataType::DT_FLOAT16){
          upsampleBicubic2dGradOut = l0op::Cast(upsampleBicubic2dGradOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        }
        CHECK_RET(upsampleBicubic2dGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(upsampleBicubic2dGradOut, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  } else {
    // judge deterministic
    auto dtype = gradOut->GetDataType();
    int64_t castFp32Condition = 0;
    int64_t deterministicValue = 1;
    rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
    if (retRts != RT_ERROR_NONE) {
      deterministicValue = 0;
      OP_LOGD("Unable to get system param determinstic = %ld.", deterministicValue);
    }
    OP_LOGD("deterministic is = %ld.", deterministicValue);

    // gradout transpose
    bool fp16Cast = (deterministicValue == 1) || (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910);
    const aclTensor* gradOutContiguous = nullptr;
    if (fp16Cast && dtype == op::DataType::DT_FLOAT16) {
      castFp32Condition = FP16_DETERMINSTIC_CAST;
      gradOutContiguous = ContiguousCast(gradOut, op::DataType::DT_FLOAT, uniqueExecutor.get());
    } else if (dtype == op::DataType::DT_BF16){
      castFp32Condition = BF16_CAST;
      gradOutContiguous = ContiguousCast(gradOut, op::DataType::DT_FLOAT, uniqueExecutor.get());
    } else {
      gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    }
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t permuteHWNCList[] = {2, 3, 0, 1};
    auto permuteHWNCArray = uniqueExecutor.get()->AllocIntArray(permuteHWNCList, FOURDIMS);
    CHECK_RET(permuteHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradOutTranspose = l0op::Transpose(gradOutContiguous, permuteHWNCArray, uniqueExecutor.get());
    CHECK_RET(gradOutTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];

    // constuct aclFloatArray
    const float scalesList[] = {scalesH, scalesW};
    const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList, EXPECT_SIZE);
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // gradinput transpose
    const aclTensor* gradInputContiguous = nullptr;
    if ((fp16Cast && dtype == op::DataType::DT_FLOAT16) || (dtype == op::DataType::DT_BF16)) {
      gradInputContiguous = ContiguousCast(gradInput, op::DataType::DT_FLOAT, uniqueExecutor.get());
    } else {
      gradInputContiguous = l0op::Contiguous(gradInput, uniqueExecutor.get());
    }
    CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t newShape[4] = {batch, channels, inputH, inputW};
    aclIntArray* shapeArray = uniqueExecutor.get()->AllocIntArray(newShape, 4);
    gradInputContiguous = l0op::Reshape(gradInputContiguous, shapeArray, uniqueExecutor.get());
    CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradInputTranspose = l0op::Transpose(gradInputContiguous, permuteHWNCArray, uniqueExecutor.get());
    CHECK_RET(gradInputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // call l0op resizeGradD
    const aclTensor *ResizeGradDOut = l0op::ResizeGradD(gradOutTranspose, inputSize, scales, alignCorners, gradInputTranspose, CUBIC_MODE, uniqueExecutor.get());
    CHECK_RET(ResizeGradDOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const int64_t newReverseShape[4] = {inputH, inputW, batch, channels};
    auto adaptResult = adaptOutput(gradOut, ResizeGradDOut, permuteHWNCArray, uniqueExecutor.get(), newReverseShape, castFp32Condition);

    // view copy out
    auto viewCopyResult = l0op::ViewCopy(adaptResult, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnUpsampleBicubic2dBackward);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
