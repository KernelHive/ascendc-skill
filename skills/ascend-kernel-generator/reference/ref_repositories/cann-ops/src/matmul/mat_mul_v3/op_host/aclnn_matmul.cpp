/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul.cpp
 * \brief
 */

#include "aclnn_matmul.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "dot.h"
#include "fill.h"
#include "matmul_l0.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "unsqueeze.h"

#include "cube_util.h"
#include "matmul_util.h"
#include "op_api_def.h"

#define OP_LOGI(...) do {std::printf(__VA_ARGS__); std::printf("\n");} while(0)

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t MAX_SUPPORT_MATMUL_DIMS_NUMS = 6;
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                       DataType::DT_FLOAT16,
                                                                       DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = {DataType::DT_FLOAT,
                                                                                    DataType::DT_FLOAT16};

inline static bool CheckNotNull(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(mat2, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckSocVersionIsSupportBf16(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static inline bool CheckMathType(const aclTensor *self, const aclTensor *mat2, int8_t cubeMathType) {
  bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
  bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
  auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
  return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* mat2, const aclTensor* out,
                                   int8_t cubeMathType)
{
  bool bf16flag = CheckSocVersionIsSupportBf16();
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto dtypeList = bf16flag ? DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST_WITHOUT_BF16;
  OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(mat2, dtypeList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false);
  if (!bf16flag &&
      (self->GetDataType() == op::DataType::DT_BF16 ||
       mat2->GetDataType() == op::DataType::DT_BF16 ||
       out->GetDataType() == op::DataType::DT_BF16)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "Bfloat16 is unsupported by the current SOC version [%s], now self is %s, mat2 is %s, out is %s",
              op::ToString(socVersion).GetString(),
              op::ToString(self->GetDataType()).GetString(),
              op::ToString(mat2->GetDataType()).GetString(),
              op::ToString(out->GetDataType()).GetString());
      return false;
  }

  // keeptype模式支持类型检查
  if (cubeMathType == KEEP_DTYPE && IsInputSupportFp32() == false &&
      (self->GetDataType() == DataType::DT_FLOAT || mat2->GetDataType() == DataType::DT_FLOAT)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s or mat2 dtype %s not support under keep type mode.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
    return false;
  }
  if (cubeMathType == KEEP_DTYPE && out->GetDataType() == op::DataType::DT_FLOAT16 &&
      self->GetDataType() == op::DataType::DT_FLOAT) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input tensor's dtype[DT_FLOAT] should be same with output's dtype[DT_FLOAT16].");
    return false;
  }

  return true;
}

// 获取broadcast shape
inline static op::Shape GetBroadcastShape(const aclTensor* tensor)
{
  op::Shape shape;
  size_t dimNum = tensor->GetViewShape().GetDimNum();
  size_t loopDims = dimNum - 2; // the dims except the last two
  for (size_t idx = 0; idx < loopDims; idx++) {
    int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
    shape.AppendDim(tmpVal);
  }
  if (shape.GetDimNum() == 0) {
    shape.AppendDim(1);
  }
  return shape;
}

static bool CheckShapeValid(const aclTensor* self, const aclTensor* mat2)
{
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  auto dimTensor1 = selfShape.GetDimNum();
  auto dimTensor2 = mat2Shape.GetDimNum();
  int64_t selfKDim = 0;
  int64_t mat2KDim = 0;

  // 超出最大支持维度返回
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_MATMUL_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(mat2, MAX_SUPPORT_MATMUL_DIMS_NUMS, return false);

  // Tensor1 dims number is 0 OR error dims number is 0
  if (dimTensor1 == 0 || dimTensor2 == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Matmul not support %s, %s", op::ToString(mat2Shape).GetString(),
            op::ToString(mat2Shape).GetString());
    return false;

    // tensor1 dims number is 1 OR tensor2 dims number is 2
  } else if (dimTensor2 == 1 || dimTensor2 == 2) {
    selfKDim = selfShape.GetDim(dimTensor1 - 1);  // the rear dim 1
    mat2KDim = mat2Shape.GetDim(0);               // the front 0
    if (selfKDim != mat2KDim) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
              op::ToString(selfShape).GetString(), op::ToString(mat2Shape).GetString());
      return false;
    }

    // tensor2 dims number > 3
  } else if (dimTensor2 >= 3) {
    selfKDim = selfShape.GetDim(dimTensor1 - 1);  // the rear dim 1
    mat2KDim = mat2Shape.GetDim(dimTensor2 - 2);  // the rear dim 2
    if (selfKDim != mat2KDim) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
              op::ToString(selfShape).GetString(), op::ToString(mat2Shape).GetString());
      return false;
    }
  }

  // 检查是否满足broadcast规则
  if (dimTensor1 >= 2 && dimTensor2 >= 2) { // the dims larger than 2
    op::Shape broadcastShape;
    auto selfBroadcastShape = GetBroadcastShape(self);
    auto mat2BroadcastShape = GetBroadcastShape(mat2);
    if (!BroadcastInferShape(selfBroadcastShape, mat2BroadcastShape, broadcastShape)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
              op::ToString(mat2->GetViewShape()).GetString());
      return false;
    }
  }

  return true;
}

inline static aclnnStatus CheckParam(const aclTensor* self, const aclTensor* mat2, const aclTensor* out,
                                     int8_t cubeMathType)
{
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, mat2, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查Shape是否支持
  CHECK_RET(CheckShapeValid(self, mat2), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查cubeMathType
  CHECK_RET(CheckMathType(self, mat2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}


static const aclTensor* ProcessEmptyTensor(const aclTensor* self, const aclTensor* mat2, const aclTensor* out,
                                           aclOpExecutor* executor)
{
  // 获取shape信息
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  op::Shape outShape = out->GetViewShape();
  auto output = executor->AllocTensor(outShape, self->GetDataType());
  if (output->IsEmpty()) {
    OP_LOGI("Returning an empty tensor without actually doing calculation.");
    return output;
  }
  FVector<int64_t> fillShape = GetShape(output);
  const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
  aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
  const aclScalar* valueScalar = executor->AllocScalar(0);
  const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
  auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
  return fillTensor;
}

static inline const aclTensor* ContiguousUnsqueezeNd(const aclTensor* input, FVector<int64_t>& dim_data,
                                                     aclOpExecutor* executor)
{
  auto inputContiguous = l0op::Contiguous(input, executor);
  CHECK_RET(inputContiguous != nullptr, nullptr);

  auto dims = executor->AllocIntArray(dim_data.data(), dim_data.size());
  auto output = l0op::UnsqueezeNd(inputContiguous, dims, executor);
  CHECK_RET(output != nullptr, nullptr);

  return output;
}

static const aclTensor* BuildDotGraph(const aclTensor* self, const aclTensor* mat2, const aclTensor* out,
                                      aclOpExecutor* executor)
{
  // 检查输入size是否相等
  auto dimSize1 = self->GetViewShape().GetDim(0);
  auto dimSize2 = mat2->GetViewShape().GetDim(0);
  if (dimSize1 != dimSize2) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dimSize [%ld] should be same as mat2 dimSize [%ld].", dimSize1, dimSize2);
    return nullptr;
  }

  // 连续性转换
  self = l0op::Contiguous(self, executor);
  CHECK_RET(self != nullptr, nullptr);
  mat2 = l0op::Contiguous(mat2, executor);
  CHECK_RET(mat2 != nullptr, nullptr);

  // 全部转成ND
  self = l0op::ReFormat(self, op::Format::FORMAT_ND);
  CHECK_RET(self != nullptr, nullptr);
  mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
  CHECK_RET(mat2 != nullptr, nullptr);

  // 提升精度
  auto promoteType = op::PromoteType(self->GetDataType(), mat2->GetDataType());
  self = l0op::Cast(self, promoteType, executor);
  CHECK_RET(self != nullptr, nullptr);
  mat2 = l0op::Cast(mat2, promoteType, executor);
  CHECK_RET(mat2 != nullptr, nullptr);

  // 点乘运算
  auto dotOut = l0op::Dot(self, mat2, executor);
  CHECK_RET(dotOut != nullptr, nullptr);

  // 转到输出类型
  auto dotCast = l0op::Cast(dotOut, out->GetDataType(), executor);
  return dotCast;
}

static inline const aclTensor* BuildBatchMatmulGraph(const aclTensor* self, const aclTensor* mat2, const aclTensor* out,
                                                     int8_t cubeMathType, aclOpExecutor* executor)
{
  auto matmulOut = ExecBmmOp(self, mat2, out, cubeMathType, executor);
  CHECK_RET(matmulOut != nullptr, nullptr);
  return matmulOut;
}

static const aclTensor* BuildMatMulGraph(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
                                         int8_t cubeMathType, aclOpExecutor* executor)
{
  // 空tensor 处理
  if (self->IsEmpty() || mat2->IsEmpty()) {
    auto emptyOut = ProcessEmptyTensor(self, mat2, out, executor);
    CHECK_RET(emptyOut != nullptr, nullptr);
    return emptyOut;
  }

  auto dimTensor1 = self->GetViewShape().GetDimNum();
  auto dimTensor2 = mat2->GetViewShape().GetDimNum();
  const aclTensor* matmulOut = nullptr;

  // Tensor1 dims number 1  && Tensor2 dims number 1
  if (dimTensor1 == 1 && dimTensor2 == 1) {
    // dot_out
    matmulOut = BuildDotGraph(self, mat2, out, executor);

    // Tensor1 dims number 2  && Tensor2 dims number 1
  } else if (dimTensor1 == 2 && dimTensor2 == 1) {
    // mv_out to check
    FVector<int64_t> dim_data{-1};
    auto mat2_unsqueeze = ContiguousUnsqueezeNd(mat2, dim_data, executor);
    CHECK_RET(mat2_unsqueeze != nullptr, nullptr);
    matmulOut = ExecMmOp(self, mat2_unsqueeze, cubeMathType, executor);

    // Tensor1 dims number 1 && Tensor2 dims number 2
  } else if (dimTensor1 == 1 && dimTensor2 == 2) {
    FVector<int64_t> dim_data{0};
    auto self_unsqueeze = ContiguousUnsqueezeNd(self, dim_data, executor);
    CHECK_RET(self_unsqueeze != nullptr, nullptr);
    matmulOut = ExecMmOp(self_unsqueeze, mat2, cubeMathType, executor);

    // Tensor1 dims number 2 && Tensor2 dims number 2
  } else if (dimTensor1 == 2 && dimTensor2 == 2) {
    matmulOut = ExecMmOp(self, mat2, cubeMathType, executor);

    // Tensor1 dims number > 3 & Tensor2 dim number is 1 or 2
  } else if (dimTensor1 >= 3 && (dimTensor2 == 1 || dimTensor2 == 2)) {
    // t1:(N, n, m) * t2:(m, p)
    auto mat2_unsqueeze = mat2;
    if (dimTensor2 == 1) {
      FVector<int64_t> dim_data{-1};
      mat2_unsqueeze = ContiguousUnsqueezeNd(mat2, dim_data, executor);
      CHECK_RET(mat2_unsqueeze != nullptr, nullptr);
    }
    // Fold the batch into the first dimension
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, nullptr);
    op::Shape shape{-1, selfContiguous->GetViewShape().GetDim(dimTensor1 - 1)};
    auto self_reshape = l0op::Reshape(selfContiguous, shape, executor);
    CHECK_RET(self_reshape != nullptr, nullptr);
    matmulOut = ExecMmOp(self_reshape, mat2_unsqueeze, cubeMathType, executor);

    // Tensor1 dims number is 1 or 2 && Tensor2 dim number > 3
  } else if ((dimTensor1 == 1 || dimTensor1 == 2) && dimTensor2 >= 3) {
    // t1:(n, m) * t2:(N, m, p)
    // mm_out to check
    FVector<int64_t> dim_data;
    if (dimTensor1 == 1) {
      dim_data = FVector<int64_t>{0};  // unsquee dim 0
    } else {
      dim_data = FVector<int64_t>{0, 1};  //  unsquee dim 0,1
    }
    auto self_unsqueeze = ContiguousUnsqueezeNd(self, dim_data, executor);
    CHECK_RET(self_unsqueeze != nullptr, nullptr);
    matmulOut = BuildBatchMatmulGraph(self_unsqueeze, mat2, out, cubeMathType, executor);

    // Tensor1 dims number > 3 && Tensor2 dim number > 3
  } else if (dimTensor1 >= 3 && dimTensor2 >= 3) {
    matmulOut = BuildBatchMatmulGraph(self, mat2, out, cubeMathType, executor);

    // Impossible cases.
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Internal error self: %s, mat2: %s",
            op::ToString(self->GetViewShape()).GetString(), op::ToString(mat2->GetViewShape()).GetString());

    return nullptr;
  }

  CHECK_RET(matmulOut != nullptr, nullptr);
  // Reshape to out shape
  auto matReshape = l0op::Reshape(matmulOut, out->GetViewShape(), executor);
  CHECK_RET(matReshape != nullptr, nullptr);

  return matReshape;
}

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
                                        int8_t cubeMathType, size_t* workspaceSize, aclOpExecutor** executor)
{
  L2_DFX_PHASE_1(aclnnMatmul, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto unique_executor = CREATE_EXECUTOR();
  CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 入参检查
  auto ret = CheckParam(self, mat2, out, cubeMathType);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 构建matmul计算图
  auto matmulOut = BuildMatMulGraph(self, mat2, out, cubeMathType, unique_executor.get());
  CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (matmulOut->IsEmpty()) {
    // 当输出为空tensor的场景，空tensor处理
    *workspaceSize = 0;
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto viewCopyResult = l0op::ViewCopy(matmulOut, out, unique_executor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取workspace
  *workspaceSize = unique_executor->GetWorkspaceSize();
  unique_executor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnMatmul);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
