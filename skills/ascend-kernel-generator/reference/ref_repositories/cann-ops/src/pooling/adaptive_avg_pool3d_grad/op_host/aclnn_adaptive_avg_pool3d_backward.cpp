/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <vector>
#include "adaptive_avg_pool3d_backward.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "fill.h"
#include "mul.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/fast_vector.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_adaptive_avg_pool3d_backward.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/transpose.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t N_DIM_INDEX = -3;
static const int64_t H_DIM_INDEX = -2;
static const int64_t W_DIM_INDEX = -1;
static const int64_t D_DIM_INDEX_FROM_LAST = 3;
static const int64_t H_DIM_INDEX_FROM_LAST = 2;
static const int64_t W_DIM_INDEX_FROM_LAST = 1;
static const int64_t DHW_SHAPE_SIZES = 3;
static const int64_t CDHW_SHAPE_SIZES = 4;
static const int64_t NCDHW_SHAPE_SIZES = 5;

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckInputOutDims(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out) {
  auto selfDimNum = self->GetViewShape().GetDimNum();
  auto gradOutputDimNum = gradOutput->GetViewShape().GetDimNum();
  auto outDimNum = out->GetViewShape().GetDimNum();

  const op::Shape selfShape = self->GetViewShape();
  const op::Shape gradOutputShape = gradOutput->GetViewShape();
  const op::Shape outShape = out->GetViewShape();

  if (selfDimNum != CDHW_SHAPE_SIZES && selfDimNum != NCDHW_SHAPE_SIZES) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "non-empty 4D or 5D tensor expected for inputs.");
    return false;
  }
  if (selfDimNum != gradOutputDimNum || outDimNum != selfDimNum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputs and out must have same DimNum.");
    return false;
  }

  for (size_t i = 0; i < selfDimNum; i++) {
    if (selfShape.GetDim(i) <= 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self'dims is invalid, self No.[%lu] dim is not bigger than [%d].", i + 1, 0);
      return false;
    }
    if (selfShape.GetDim(i) != outShape.GetDim(i)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dims No.[%lu] must match self dims.", i);
      return false;
    }
  }
  size_t offset = gradOutputDimNum - DHW_SHAPE_SIZES;
  for (size_t i = 0; i < offset; i++) {
    if (selfShape.GetDim(i) != gradOutputShape.GetDim(i)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput dims No.[%lu] must match self dims.", i);
      return false;
    }
  }
  for (size_t i = offset; i < gradOutputDimNum; i++) {
    if (gradOutputShape.GetDim(i) <= 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput dims is invalid, self No.[%lu] dim is not bigger than [%d].", i + 1, 0);
      return false;
    }
  }
  return true;
}

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                       op::DataType::DT_DOUBLE,
                                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_ALLONE_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                               op::DataType::DT_DOUBLE,
                                                                                               op::DataType::DT_FLOAT16,
                                                                                               op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910B_NOTALLONE_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                               op::DataType::DT_FLOAT16,
                                                                                               op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList(const aclTensor* gradOutput) {
  op::Shape gradOutputShape = gradOutput->GetViewShape();
  auto gradOutputDimNum = gradOutputShape.GetDimNum();
  int64_t dValue = gradOutputShape.GetDim(gradOutputDimNum - D_DIM_INDEX_FROM_LAST);
  int64_t hValue = gradOutputShape.GetDim(gradOutputDimNum - H_DIM_INDEX_FROM_LAST);
  int64_t wValue = gradOutputShape.GetDim(gradOutputDimNum - W_DIM_INDEX_FROM_LAST);
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
      if (dValue == 1 && hValue == 1 && wValue == 1) {
        return ASCEND910B_ALLONE_DTYPE_DTYPE_SUPPORT_LIST;
      }
      else {
        return ASCEND910B_NOTALLONE_DTYPE_DTYPE_SUPPORT_LIST;
      }
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out) {
  const auto& supportList = GetDtypeSupportList(gradOutput);
  // 检查gradOutput、self的数据类型是否在AdaptiveAvgPool3dBackward算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  
  // 检查gradOutput和self数据类型是否一致
  OP_CHECK_DTYPE_NOT_SAME(gradOutput, self, return false);

  // 检查self和out数据类型是否一致
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
  return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out) {
  if (gradOutput->GetStorageFormat() != self->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of inputs should be equal, gradOutput [%s], self [%s]",
            op::ToString(gradOutput->GetStorageFormat()).GetString(),
            op::ToString(self->GetStorageFormat()).GetString());
    return false;
  }
  if (out->GetStorageFormat() != self->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of self and out should be equal, out [%s], self [%s]",
            op::ToString(out->GetStorageFormat()).GetString(),
            op::ToString(self->GetStorageFormat()).GetString());
    return false;
  }
  // 如果输入格式是私有格式，记录日志，直接报错
  if (op::IsPrivateFormat(gradOutput->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCDHW、ND.");
    return false;
  }

  return true;
}

// 检查参数是否满足910A芯片算子的运算逻辑
static aclnnStatus CheckParamsLogic(const aclTensor* gradOutput) {
  auto gradOutputShapes = gradOutput->GetViewShape();
  auto gradOutputDim = gradOutputShapes.GetDimNum();
  // 输入参数gradOutput的shape最后三个的大小必须为[1, 1, 1]
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910 &&
      !(gradOutputShapes.GetDim(gradOutputDim - D_DIM_INDEX_FROM_LAST) == 1 &&
      gradOutputShapes.GetDim(gradOutputDim - H_DIM_INDEX_FROM_LAST) == 1 &&
      gradOutputShapes.GetDim(gradOutputDim - W_DIM_INDEX_FROM_LAST) == 1)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "adaptive_avg_pool3d_backward only support D=1 && H=1 && W=1 on 910A!");
    return ACLNN_ERR_PARAM_INVALID;
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 校验输入参数维度
  CHECK_RET(CheckInputOutDims(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入的数据类型是否在支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查数据形状是否支持
  CHECK_RET(CheckFormat(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);

   // 5. 检查参数是否满足910A芯片算子的运算逻辑
  CHECK_RET(CheckParamsLogic(gradOutput) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static int64_t adaptive_avg_pool3d_backward_safe_size(const aclTensor* self) {
  FVector<int64_t, DHW_SHAPE_SIZES> dims = {N_DIM_INDEX, H_DIM_INDEX, W_DIM_INDEX};

  int64_t size = 1;
  if (self->IsEmpty()) {
    return size;
  }
  int64_t dim_post_expr = self->GetViewShape().GetDimNum();
  for (int64_t ndim : dims) {
    if (dim_post_expr <= 0) {
      dim_post_expr = 1;  // this will make range [-1, 0]
    }
    ndim += dim_post_expr;
    size *= self->GetViewShape().GetDim(ndim);
  }

  return size;
}

//NC111 --->  NCDHW
static aclnnStatus DoAllOneAdapativeAvgPool3dBackward(const aclTensor* gradOutput, const aclTensor* self, aclTensor* out,
                                                      aclOpExecutor* executor) {
  // fill value tensor
  float fillVaule = 1.0 / (adaptive_avg_pool3d_backward_safe_size(self));
  aclScalar* value = executor->AllocScalar(fillVaule);
  const aclTensor* castTensor = executor->ConvertToTensor(value, static_cast<op::DataType>(self->GetDataType()));

  // fill dims tensor
  size_t dimNum = self->GetViewShape().GetDimNum();
  FVector<int64_t> dimTmp;
  for (size_t idx = 0; idx < dimNum; idx++) {
    int64_t tmpVal = self->GetViewShape().GetDim(idx);
    dimTmp.push_back(tmpVal);
  }

  aclIntArray* shapeArray = executor->AllocIntArray(dimTmp.data(), dimTmp.size());
  const aclTensor* dims = executor->ConvertToTensor(dimTmp.data(), dimTmp.size(),
                                                    static_cast<op::DataType>(ACL_INT64));
  auto fillOut = l0op::Fill(dims, castTensor, shapeArray, executor);
  CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto mulOpOut = l0op::Mul(fillOut, gradOutput, executor);
  CHECK_RET(mulOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(mulOpOut, out->GetDataType(), executor);
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto view_copy_result = l0op::ViewCopy(castOut, out, executor);
  CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

//不为[1,1,1]
static aclnnStatus DoCommonAdaptiveAvgPool3dBackward(const aclTensor* gradOutput, const aclTensor* self, aclTensor* out,
                                                     aclOpExecutor* executor) {
  //进行NC维度的对齐，reshape(展平NC为1维dhwl)
  auto gradOutputReshape = gradOutput;
  int64_t preDims = self->GetViewShape().GetDimNum();
  if (preDims == NCDHW_SHAPE_SIZES) {
    op::Shape gradOutputShape = gradOutput->GetViewShape();
    FVector<int64_t> valueShape;
    valueShape.push_back(-1);
    for (int64_t i = 2; i < NCDHW_SHAPE_SIZES; i++) {
        valueShape.push_back(gradOutputShape.GetDim(i));
    }
    auto reshapeShape = executor->AllocIntArray(valueShape.data(), CDHW_SHAPE_SIZES);        
    CHECK_RET(reshapeShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    gradOutputReshape = l0op::Reshape(gradOutputReshape, reshapeShape, executor);
  }
  //将DHW维度转置到前面
  std::vector<int64_t> valuePerm(CDHW_SHAPE_SIZES);
  for (int64_t i = 0; i < CDHW_SHAPE_SIZES; i++) {
    valuePerm[i] = (i + CDHW_SHAPE_SIZES - DHW_SHAPE_SIZES) % CDHW_SHAPE_SIZES;
  }
  auto perm = executor->AllocIntArray(valuePerm.data(), CDHW_SHAPE_SIZES);
  CHECK_RET(perm!=nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradOutputReshapeTran = l0op::Transpose(gradOutputReshape, perm, executor);
  CHECK_RET(gradOutputReshapeTran != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto poolResult = l0op::AdaptiveAvgPool3dGrad(gradOutputReshapeTran, self, executor);
  CHECK_RET(poolResult != nullptr, ACLNN_ERR_INNER_NULLPTR); 
  //将DHW维度转置回后面
  std::vector<int64_t> valuePermRes(CDHW_SHAPE_SIZES);
  for (int64_t i = 0; i < CDHW_SHAPE_SIZES; i++) {
    valuePermRes[i] = (i + DHW_SHAPE_SIZES) % CDHW_SHAPE_SIZES;
  }
  auto permRes = executor->AllocIntArray(valuePermRes.data(), CDHW_SHAPE_SIZES);
  CHECK_RET(permRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto resTran = l0op::Transpose(poolResult, permRes, executor);
  CHECK_RET(resTran != nullptr, ACLNN_ERR_INNER_NULLPTR); 
  auto resTranReshape = resTran;
  if (self->GetViewShape().GetDimNum() == NCDHW_SHAPE_SIZES) {
    FVector<int64_t> resShapeValue;
    for (int64_t i = 0; i < NCDHW_SHAPE_SIZES; i++){
      resShapeValue.push_back(self->GetViewShape().GetDim(i));
    }
    auto resShape = executor->AllocIntArray(resShapeValue.data(), NCDHW_SHAPE_SIZES);
    CHECK_RET(resShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    resTranReshape = l0op::Reshape(resTran, resShape, executor);
  }
  CHECK_RET(resTranReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyResult = l0op::ViewCopy(resTranReshape, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus SelectAdaptiveAvgPool3dBackward(const aclTensor* gradOutput, const aclTensor* self, aclTensor* out,
                                                   aclOpExecutor* executor) {
  op::Shape gradOutputShape = gradOutput->GetViewShape();
  auto gradOutputDimNum = gradOutputShape.GetDimNum();
  int64_t dValue = gradOutputShape.GetDim(gradOutputDimNum - D_DIM_INDEX_FROM_LAST);
  int64_t hValue = gradOutputShape.GetDim(gradOutputDimNum - H_DIM_INDEX_FROM_LAST);
  int64_t wValue = gradOutputShape.GetDim(gradOutputDimNum - W_DIM_INDEX_FROM_LAST);
  if (dValue == 1 && hValue == 1 && wValue == 1) {
    return DoAllOneAdapativeAvgPool3dBackward(gradOutput, self, out, executor);
  }                                        
  return DoCommonAdaptiveAvgPool3dBackward(gradOutput, self, out, executor);
}

aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                           aclTensor* out, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnAdaptiveAvgPool3dBackward, DFX_IN(gradOutput, self), DFX_OUT(out));

  // 参数检查
  auto ret = CheckParams(gradOutput, self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (self->IsEmpty() || gradOutput->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入gradOutput转换成连续的tensor
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto rets = SelectAdaptiveAvgPool3dBackward(gradOutputContiguous, selfContiguous, out, uniqueExecutor.get());
  CHECK_RET(rets == ACLNN_SUCCESS, rets);

  //固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveAvgPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnAdaptiveAvgPool3dBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
