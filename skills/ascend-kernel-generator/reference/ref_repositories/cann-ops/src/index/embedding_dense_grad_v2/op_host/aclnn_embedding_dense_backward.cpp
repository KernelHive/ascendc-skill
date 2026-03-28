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

#include "aclnn_embedding_dense_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "sort.h"
#include "embedding_dense_grad.h"
#include "zero_op.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "runtime/context.h"

using namespace op;

static const int64_t MAX_SUPPORT_DIM = 8;
static const int64_t LIMIT_EMBEDDING_DIM_NUM = 14336;
static const uint64_t INT32_MAX_LIMIT = 2147483647;
static const uint64_t INT32_INF = 2139095040;
static const uint64_t SINGLE_CORE_SORT_ROW_NUM = 192;
static const uint64_t CAST_MAX_NUM = 16777216;
static const int MIN_SORT_CORE_NUM = 1;
static const int MAX_SORT_CORE_NUM = 4;
static const int OUT_SHAPE = 2;
static const int SMALL_DIM_THRESH = 512;
static const int BLOCK_SIZE = 32;

static const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_BOOL};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_910 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *grad, const aclTensor *indices, const aclTensor *out) {
  OP_CHECK_NULL(grad, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *grad, const aclTensor *indices, const aclTensor *out) {
  bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
  const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST =
    is910BSocVersion ? GRAD_DTYPE_SUPPORT_LIST_910B : GRAD_DTYPE_SUPPORT_LIST_910;
  const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST =
    is910BSocVersion ? OUT_DTYPE_SUPPORT_LIST_910B : OUT_DTYPE_SUPPORT_LIST_910;

  // 检查grad的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(grad, GRAD_DTYPE_SUPPORT_LIST, return false);

  // 检查indices的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);

  return true;
}

static bool CheckMaxDimension(const aclTensor *tensor) {
  OP_CHECK_MAX_DIM(tensor, MAX_SUPPORT_DIM, return false);
  return true;
}

static bool CheckDimension(const aclTensor *grad, const aclTensor *indices) {
  auto gradShape = grad->GetViewShape();
  auto indicesShape = indices->GetViewShape();
  if (gradShape.GetDim(gradShape.GetDimNum() - 1) == 0) {
    return true;
  }

  int64_t gradShapeSum = 1;
  int64_t indicesShapeSum = 1;
  for (size_t i = 0; i < gradShape.GetDimNum() - 1; i++) {
    gradShapeSum *= gradShape.GetDim(i);
  }
  for (size_t i = 0; i < indicesShape.GetDimNum(); i++) {
    indicesShapeSum *= indicesShape.GetDim(i);
  }
  if (indicesShapeSum != gradShapeSum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad shape [%s] is not match with indices shape [%s].",
              op::ToString(grad->GetViewShape()).GetString(), op::ToString(indices->GetViewShape()).GetString());
    return false;
  }
  return true;
}

static bool CheckOutShape(const aclTensor *out, const aclTensor *grad, const uint64_t numWeights)
{
  auto outShape = out->GetViewShape();
  auto gradShape = grad->GetViewShape();
  auto outDimNum = outShape.GetDimNum();
  if (outDimNum != OUT_SHAPE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outDim shape is not equal to 2");
    return false;
  }
  if (static_cast<uint64_t>(outShape.GetDim(0)) != numWeights ||
      outShape.GetDim(1) != gradShape.GetDim(gradShape.GetDimNum() - 1)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outshape [%s] is not match with infershape {%lu, %lu}.",
            op::ToString(out->GetViewShape()).GetString(), numWeights, gradShape.GetDim(gradShape.GetDimNum() - 1));
    return false;
  }
  return true;
}

static bool CheckFormat(const aclTensor *grad, const aclTensor *indices, const aclTensor *out)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(grad->GetStorageFormat()) || op::IsPrivateFormat(indices->GetStorageFormat()) ||
        op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *grad, const aclTensor *indices, const aclTensor *out,
                               const uint64_t numWeights) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(grad, indices, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(grad, indices, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查最大维度是否超过8
  CHECK_RET(CheckMaxDimension(grad) && CheckMaxDimension(indices)
            && CheckMaxDimension(out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查grad和indices的维度匹配关系
  CHECK_RET(CheckDimension(grad, indices), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查out的shape是否符合推导规则
  CHECK_RET(CheckOutShape(out, grad, numWeights), ACLNN_ERR_PARAM_INVALID);

  // 6. 检查参数数据格式是否合法
  CHECK_RET(CheckFormat(grad, indices, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckIsSmallDimMode(const uint64_t embeddingDim, const uint64_t numWeights)
{
  int64_t deterministicValue = 0;
  rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
  if (retRts != RT_ERROR_NONE) {
    deterministicValue = 0;
  }
  return deterministicValue == 0 && embeddingDim <= SMALL_DIM_THRESH && numWeights <= CAST_MAX_NUM;
}

static bool IsComputeByV2(const aclTensor *grad, const uint64_t numWeights)
{
  bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
  int64_t gradRow = 1;
  auto gradShape = grad->GetViewShape();
  for (uint32_t i = 0; i < gradShape.GetDimNum() - 1; i++) {
    gradRow *= gradShape.GetDim(i);
  }
  auto embeddingDim = gradShape.GetDim(gradShape.GetDimNum() - 1);
  return is910BSocVersion &&
         (gradRow <= INT32_MAX_LIMIT) && (numWeights <= INT32_MAX_LIMIT);
}

static int64_t ComputeGradRow(const aclTensor *grad)
{
  int64_t gradRow = 1;
  auto gradShape = grad->GetViewShape();
  for (uint32_t i = 0; i < gradShape.GetDimNum() - 1; i++) {
    gradRow *= gradShape.GetDim(i);
  }
  return gradRow;
}

static void ViewDataType(const aclTensor *input, const op::DataType dtype)
{
  if (input == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "view data type error!!");
    return;
  }
  auto tmpTensor = const_cast<aclTensor *>(input);
  tmpTensor->SetDataType(dtype);
  input = tmpTensor; 
}

static std::pair<const aclTensor*, const aclTensor*> PorcessIndices(const aclTensor *indicesContiguous, const aclTensor *grad,
                                                                    aclOpExecutor *executor)
{
    int64_t gradRow = ComputeGradRow(grad);
    if (gradRow == 1) {
      FVector<int32_t> posIdxVec = {0};
      auto posIdx = executor->ConvertToTensor(posIdxVec.data(), posIdxVec.size(), DataType::DT_INT32);
      return std::make_pair(indicesContiguous, posIdx);
    }

    const aclTensor *indiceViewFloat = executor->CreateView(indicesContiguous, {gradRow}, indicesContiguous->GetViewOffset());
    if (indiceViewFloat == nullptr) { return {nullptr, nullptr}; }

    // 修改读取的数据类型
    if (gradRow < INT32_INF) {
      ViewDataType(indiceViewFloat, op::DataType::DT_FLOAT);
      OP_LOGD("aclnnEmbeddingDenseGradV2: indice sort by aicore");
    }
    // 对index进行sort操作
    indiceViewFloat = l0op::Reshape(indiceViewFloat, {gradRow}, executor);
    if (indiceViewFloat == nullptr) { return {nullptr, nullptr}; }
    auto sortResult = l0op::Sort(indiceViewFloat, -1, false, true, executor);
    const aclTensor *sortIdxOut = std::get<0>(sortResult);
    const aclTensor *posIdx = std::get<1>(sortResult);
    if (sortIdxOut == nullptr || posIdx == nullptr) {
      return {nullptr, nullptr};
    }
    // 将sort的结果已INT32的数据类型来进行读取
    auto sortIndice = executor->CreateView(sortIdxOut, {gradRow}, sortIdxOut->GetViewOffset());
    if (sortIndice == nullptr) { return {nullptr, nullptr}; }
    sortIndice->SetDataType(op::DataType::DT_INT32);
    return std::make_pair(sortIndice, posIdx);
}

aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *indices,
                                                        uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq,
                                                        const aclTensor *out, uint64_t *workspaceSize,
                                                        aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnEmbeddingDenseBackward,
                 DFX_IN(grad, indices, numWeights, paddingIdx, scaleGradByFreq),
                 DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(grad, indices, out, numWeights);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (grad->IsEmpty() || indices->IsEmpty()) {
    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用ZerosLike算子kernel
    auto zeroOut = l0op::ZerosLike(outContiguous, uniqueExecutor.get());
    CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyOut = l0op::ViewCopy(zeroOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // grad如果非连续，需要转连续
  auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
  CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // grad如果是float16/bfloat16，需要cast为float32
  auto gradCasted = l0op::Cast(gradContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
  CHECK_RET(gradCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // indices如果非连续，需要转连续
  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor *embeddingDenseBackwardResult = nullptr;
  // 判断是走V1还是V2， 芯片为910或者embeddingDim过大时就走V1，否则走V2
  if (IsComputeByV2(gradCasted, numWeights)) {
    // v2只支持int32的数据类型
    indicesContiguous = l0op::Cast(indicesContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradShape = gradCasted->GetViewShape();
    auto embeddingDim = gradShape.GetDim(gradShape.GetDimNum() - 1);
    if (!CheckIsSmallDimMode(embeddingDim, numWeights)) {
      auto result = PorcessIndices(indicesContiguous, gradCasted, uniqueExecutor.get());
      auto sortIndice = result.first;
      auto posIdx = result.second;
      CHECK_RET(sortIndice != nullptr, ACLNN_ERR_INNER_NULLPTR);
      embeddingDenseBackwardResult = l0op::EmbeddingDenseGradV2(gradCasted, sortIndice, posIdx,
                                                                numWeights, paddingIdx, scaleGradByFreq, uniqueExecutor.get());
    } else {
      embeddingDenseBackwardResult = l0op::EmbeddingDenseGradV2(gradCasted, indicesContiguous, indicesContiguous,
                                                                numWeights, paddingIdx, scaleGradByFreq, uniqueExecutor.get());
    }
  } else {
    // indices如果非int32,int64类型，需要强转
    auto castDtype = indicesContiguous->GetDataType();
    if (castDtype != op::DataType::DT_INT32 && castDtype != op::DataType::DT_INT64) {
      indicesContiguous = l0op::Cast(indicesContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
    }
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    embeddingDenseBackwardResult = l0op::EmbeddingDenseGrad(gradCasted, indicesContiguous, numWeights, paddingIdx,
                                                            scaleGradByFreq, uniqueExecutor.get());
  }
  CHECK_RET(embeddingDenseBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(embeddingDenseBackwardResult, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnEmbeddingDenseBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
