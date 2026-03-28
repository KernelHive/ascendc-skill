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
 * \file aclnn_embedding_bag.cpp
 * \brief
 */
#include "aclnn_embedding_bag.h"
#include "embedding_bag_l0.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_SUPPORT_DIM = 2;
static const int64_t MIN_SUPPORT_DIM = 1;
static const int64_t BAG_SIZE_DIM = 2;
static const int64_t MAX_INDICES_DIM = 3;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> WEIGHT_DTYPE_SUPPORT_LIST_910B = {op::DataType::DT_FLOAT,
                                                                                   op::DataType::DT_FLOAT16,
                                                                                   op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> WEIGHT_DTYPE_SUPPORT_LIST_910 = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_FLOAT16};

static inline const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
    return WEIGHT_DTYPE_SUPPORT_LIST_910;
  } else {
    return WEIGHT_DTYPE_SUPPORT_LIST_910B;
  }
}

static const std::initializer_list<op::DataType> INT_DTYPE_LIST = {op::DataType::DT_INT64, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> INT_DTYPE_LIST_ALL = {op::DataType::DT_INT64, op::DataType::DT_INT32,
                                                                       op::DataType::DT_INT16, op::DataType::DT_INT8,
                                                                       op::DataType::DT_UINT8};

static inline bool CheckNotNull(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets,
                                aclTensor* output, aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices) {
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(offsets, return false);
  OP_CHECK_NULL(output, return false);
  OP_CHECK_NULL(offset2bag, return false);
  OP_CHECK_NULL(bagSize, return false);
  OP_CHECK_NULL(maxIndices, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets,
                            const aclTensor *perSampleWeights, aclTensor* output, aclTensor* offset2bag,
                            aclTensor* bagSize, aclTensor* maxIndices) {
  // 检查weight的数据类型是否在算子的支持列表内
  if (weight != nullptr) {
    auto weightDtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, weightDtypeSupportList, return false);
  }

  // 检查indices的数据类型是否在算子的支持列表内
  if (indices != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INT_DTYPE_LIST_ALL, return false);
  }

  // 检查offsets的数据类型是否在算子的支持列表内
  if (offsets != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(offsets, INT_DTYPE_LIST_ALL, return false);
  }

  // 检查indices和offsets的数据类型是否有一个达到了INT32/INT64
  if (!CheckType(indices->GetDataType(), INT_DTYPE_LIST) && !CheckType(offsets->GetDataType(), INT_DTYPE_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices or offsets must has one dtype in [int32, int64], "
            "but get indices dtype %s, offsets dtype %s.", op::ToString(indices->GetDataType()).GetString(),
            op::ToString(offsets->GetDataType()).GetString());
  }

  // 检查perSampleWeights的数据类型是否与weight相同
  if (perSampleWeights != nullptr && weight->GetDataType() != perSampleWeights->GetDataType()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "perSampleWeights dtype %s should be in same with weight dtype %s.",
            op::ToString(perSampleWeights->GetDataType()).GetString(), op::ToString(output->GetDataType()).GetString());
    return false;
  }

  // 检查weight和output的数据类型一致
  if (weight->GetDataType() != output->GetDataType()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output dtype %s should be in same with weight dtype %s.",
            op::ToString(output->GetDataType()).GetString(), op::ToString(weight->GetDataType()).GetString());
    return false;
  }

  // 检查offset2bag的数据类型是否在算子的支持列表内
  if (offset2bag != nullptr){
    OP_CHECK_DTYPE_NOT_SUPPORT(offset2bag, INT_DTYPE_LIST, return false);
  }

  // 检查bagSize的数据类型是否在算子支持列表内
  if (bagSize != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(bagSize, INT_DTYPE_LIST, return false);
  }

  // 检查maxIndices的数据类型是否在算子支持列表内
  if (maxIndices != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(maxIndices, INT_DTYPE_LIST, return false);
  }
  return true;
}

static bool CheckDims(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets,
                      const aclTensor *perSampleWeights, const std::string& modeStr, aclTensor* output,
                      aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices) {
  if (weight != nullptr) {
    OP_CHECK_MAX_DIM(weight, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MIN_DIM(weight, MAX_SUPPORT_DIM, return false);
  }

  if (indices != nullptr) {
    OP_CHECK_MAX_DIM(indices, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MIN_DIM(indices, MIN_SUPPORT_DIM, return false);
  }

  if (offsets != nullptr){
    OP_CHECK_MAX_DIM(offsets, MAX_SUPPORT_DIM, return false);
    if (indices->GetViewShape().GetDimNum() == 1) {
      OP_CHECK_MAX_DIM(offsets, MIN_SUPPORT_DIM, return false);
    }
  }

  if(perSampleWeights != nullptr) {
    OP_CHECK_MAX_DIM(perSampleWeights, MIN_SUPPORT_DIM, return false);
    OP_CHECK_MIN_DIM(perSampleWeights, MIN_SUPPORT_DIM, return false);
  }

  if (output != nullptr) {
    OP_CHECK_MAX_DIM(output, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MIN_DIM(output, MAX_SUPPORT_DIM, return false);
  }

  if (offset2bag != nullptr) {
    OP_CHECK_MAX_DIM(offset2bag, MIN_SUPPORT_DIM, return false);
  }

  if (bagSize != nullptr) {
    OP_CHECK_MAX_DIM(bagSize, MIN_SUPPORT_DIM, return false);
    OP_CHECK_MIN_DIM(bagSize, MIN_SUPPORT_DIM, return false);
  }

  if (maxIndices != nullptr) {
    if(modeStr == "max") {
      OP_CHECK_MAX_DIM(maxIndices, MAX_SUPPORT_DIM, return false);
      OP_CHECK_MIN_DIM(maxIndices, MAX_SUPPORT_DIM, return false);
    }else{
      OP_CHECK_MAX_DIM(maxIndices, MIN_SUPPORT_DIM, return false);
      OP_CHECK_MIN_DIM(maxIndices, MIN_SUPPORT_DIM, return false);
    }
  }
  return true;
}

static op::Shape GetOutPutShape(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets,
                                bool includeLastOffset) {
  op::Shape outputShape;
  int64_t offsetSize = offsets->GetViewShape().GetShapeSize();
  if (includeLastOffset) {
    offsetSize -= 1;
  }
  outputShape.AppendDim(offsetSize);
  outputShape.AppendDim((weight->GetViewShape())[1]);
  return outputShape;
}

static bool CheckShape(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets,
                       const aclTensor *perSampleWeights, const std::string& modeStr, bool includeLastOffset,
                       aclTensor* output, aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices) {
  if(modeStr != "sum" && perSampleWeights != nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "per_sample_weights only supported with mode='sum'");
    return false;
  }

  if(perSampleWeights != nullptr && indices->GetViewShape().GetShapeSize() != perSampleWeights->GetViewShape().GetShapeSize()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected indices->GetViewShape().GetShapeSize() == perSampleWeights"
            "->GetViewShape().GetShapeSize() to be true, but got false");
    return false;
  }

  auto outputShape = GetOutPutShape(weight, indices, offsets, includeLastOffset);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outputShape, return false);

  if (offset2bag->GetViewShape().GetShapeSize() != 0 &&
      offset2bag->GetViewShape().GetShapeSize() != indices->GetViewShape().GetDim(0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offset2bag shape size should be %ld,but got %ld.",
            indices->GetViewShape().GetDim(0), offset2bag->GetViewShape().GetShapeSize());
    return false;
  }

  if(includeLastOffset) {
    if(bagSize->GetViewShape().GetShapeSize() != offsets->GetViewShape().GetDim(0) -1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bagSize shape size should be %ld,but got %ld.", 
            offsets->GetViewShape().GetShapeSize() -1, bagSize->GetViewShape().GetShapeSize());
      return false;
    }
  }else{
    if(bagSize->GetViewShape().GetShapeSize() != offsets->GetViewShape().GetDim(0)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bagSize shape size should be %ld, but got %ld.", 
            offsets->GetViewShape().GetShapeSize(), bagSize->GetViewShape().GetShapeSize());
      return false;
    }
  }

  if (modeStr == "max") {
    auto maxIndicesShape = GetOutPutShape(weight, indices, offsets, includeLastOffset);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(maxIndices, maxIndicesShape, return false);
  }else{
    if(includeLastOffset) {
      if(maxIndices->GetViewShape().GetShapeSize() != offsets->GetViewShape().GetDim(0) -1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maxIndices shape size should be %ld, but got %ld.", 
            offsets->GetViewShape().GetShapeSize() -1, maxIndices->GetViewShape().GetShapeSize());
        return false;
      }
    }else{
      if(maxIndices->GetViewShape().GetShapeSize() != offsets->GetViewShape().GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maxIndices shape size should be %ld,but got %ld.", 
            offsets->GetViewShape().GetShapeSize(), maxIndices->GetViewShape().GetShapeSize());
        return false;
      }
    }
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets,
                               const std::string& modeStr, const aclTensor *perSampleWeights, bool includeLastOffset,
                               aclTensor* output, aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(weight, indices, offsets, output, offset2bag, bagSize, maxIndices), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(weight, indices, offsets, perSampleWeights, output, offset2bag, bagSize, maxIndices),
            ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入的dim 是否满足
  CHECK_RET(CheckDims(weight, indices, offsets, perSampleWeights, modeStr, output,
                      offset2bag, bagSize, maxIndices), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入形状是否满足
  CHECK_RET(CheckShape(weight, indices, offsets, perSampleWeights, modeStr, includeLastOffset,
                       output, offset2bag, bagSize, maxIndices), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static std::string getModeStr(const int64_t mode)
{
    std::string modeStr = "max";
    if (mode == 0) {
        modeStr = "sum";
    } else if (mode == 1) {
        modeStr = "mean";
    }
    return modeStr;
}

aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor *weight, const aclTensor *indices,
                                              const aclTensor *offsets, bool scaleGradByFreq,
                                              int64_t mode, bool sparse, const aclTensor* perSampleWeights,
                                              bool includeLastOffset, int64_t paddingIdx, aclTensor* output,
                                              aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices,
                                              uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnEmbeddingBag, DFX_IN(weight, indices, offsets, scaleGradByFreq, mode, sparse,
                 perSampleWeights, includeLastOffset, paddingIdx),
                 DFX_OUT(output, offset2bag, bagSize, maxIndices));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(weight, indices, offsets, getModeStr(mode), perSampleWeights,
                         includeLastOffset, output, offset2bag, bagSize, maxIndices);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (weight->IsEmpty() || indices->IsEmpty() || offsets->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入weight,indices,offsets转换成连续的tensor
  auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  auto offsetsContiguous = l0op::Contiguous(offsets, uniqueExecutor.get());
  CHECK_RET(weightContiguous != nullptr && indicesContiguous != nullptr &&
            offsetsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入per_sample_weights转换成连续的tensor
  auto perSampleWeightsContiguous = perSampleWeights;
  if (perSampleWeights != nullptr) {
    perSampleWeightsContiguous = l0op::Contiguous(perSampleWeights, uniqueExecutor.get());
    CHECK_RET(perSampleWeightsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  //indices、offsets类型转换。
  auto indicesCast = l0op::Cast(indicesContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
  auto offsetsCast = l0op::Cast(offsetsContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
  CHECK_RET(indicesCast != nullptr && offsetsCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  static const std::string modeStr[] = {"sum", "mean"};
  static const std::string modeMax = "max";
  auto result = l0op::EmbeddingBag(weightContiguous, indicesCast, offsetsCast, scaleGradByFreq,
                                   (mode != 0 && mode != 1) ? modeMax : modeStr[mode], sparse,
                                   perSampleWeightsContiguous, includeLastOffset, paddingIdx,
                                   uniqueExecutor.get());
  CHECK_RET(std::get<0>(result) != nullptr && std::get<1>(result) != nullptr &&
            std::get<BAG_SIZE_DIM>(result) != nullptr && std::get<MAX_INDICES_DIM>(result) != nullptr,
            ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyOutputResult = l0op::ViewCopy(std::get<0>(result), output, uniqueExecutor.get());

  if (offset2bag->GetViewShape().GetShapeSize() != 0) {
    auto offset2bagL0Cast = l0op::Cast(std::get<1>(result), offset2bag->GetDataType(), uniqueExecutor.get());
    CHECK_RET(offset2bagL0Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyoffset2bagL0CastResult = l0op::ViewCopy(offset2bagL0Cast, offset2bag, uniqueExecutor.get());
    CHECK_RET(viewCopyoffset2bagL0CastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  auto bagSizeL0Cast = l0op::Cast(std::get<2>(result), bagSize->GetDataType(), uniqueExecutor.get());
  CHECK_RET(bagSizeL0Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyBagSizeL0CastResult = l0op::ViewCopy(bagSizeL0Cast, bagSize, uniqueExecutor.get());

  auto maxIndicesL0Cast = l0op::Cast(std::get<3>(result), maxIndices->GetDataType(), uniqueExecutor.get());
  CHECK_RET(maxIndicesL0Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyMaxIndicesL0CastResult = l0op::ViewCopy(maxIndicesL0Cast, maxIndices, uniqueExecutor.get());
  CHECK_RET(viewCopyOutputResult != nullptr && viewCopyMaxIndicesL0CastResult != nullptr &&
            viewCopyBagSizeL0CastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnEmbeddingBag(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnEmbeddingBag);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
