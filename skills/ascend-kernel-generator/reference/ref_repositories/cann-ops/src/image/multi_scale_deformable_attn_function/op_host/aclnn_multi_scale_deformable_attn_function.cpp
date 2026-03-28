/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_multi_scale_deformable_attn_function.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "multi_scale_deformable_attn_function_l0.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const size_t FOURTH_DIM = 3;
static const size_t FIFTH_DIM = 4;
static const size_t SIXTH_DIM = 5;

static const size_t VALUE_SIZE = 4;
static const size_t SPATIAL_SHAPES_SIZE = 2;
static const size_t LEVEL_START_INDEX_SIZE = 1;
static const size_t LOCATION_SIZE = 6;
static const size_t ATTN_WEIGHT_SIZE = 5;
static const size_t GRAD_OUT_SIZE = 4;
static const size_t GRAD_VALUE_SIZE = 4;
static const size_t GRAD_LOCATION_SIZE = 6;
static const size_t GRAD_ATTN_SIZE = 5;

static const int64_t NUM_KEYS = 1;
static const int64_t NUM_HEADS = 2;
static const int64_t NUM_LEVELS = 4;
static const int64_t NUM_POINTS = 5;
static const int64_t VALUE_DIM_LIMIT = 4;
static const int64_t SPATIAL_SHAPES_DIM_LIMIT = 2;
static const int64_t LEVEL_START_DIM_LIMIT = 1;
static const int64_t LOCATION_DIM_LIMIT = 6;
static const int64_t ATTN_WEIGHT_DIM_LIMIT = 5;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_FIVE = 5;

static const std::initializer_list<DataType> VALUE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<DataType> INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};
static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex, 
                         const aclTensor *location, const aclTensor *attnWeight, const aclTensor *output,
                         uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(spatialShape, return false);
    OP_CHECK_NULL(levelStartIndex, return false);
    OP_CHECK_NULL(location, return false);
    OP_CHECK_NULL(attnWeight, return false);
    OP_CHECK_NULL(output, return false);
    if (workspaceSize == nullptr) {
        return false;
    }
    if (executor == nullptr) {
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex, 
                            const aclTensor *location, const aclTensor *attnWeight, const aclTensor *output) {
    // 检查value的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(value, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查spatialShape的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(spatialShape, INDEX_DTYPE_SUPPORT_LIST, return false);

    // 检查levelStartIndex的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(levelStartIndex, INDEX_DTYPE_SUPPORT_LIST, return false);

    // 检查location的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(location, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查attnWeight的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(attnWeight, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查output的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(output, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查value和output的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(value, output->GetDataType(), return false);

    return true;
}

static bool CheckShape(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex, 
                            const aclTensor *location, const aclTensor *attnWeight) {
  OP_CHECK_WRONG_DIMENSION(value, VALUE_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(spatialShape, SPATIAL_SHAPES_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(levelStartIndex, LEVEL_START_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(location, LOCATION_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(attnWeight, ATTN_WEIGHT_DIM_LIMIT, return false);

  auto spatialShapeShape = spatialShape->GetViewShape();
  int64_t spatialLastDim = spatialShapeShape.GetDim(DIM_ONE);
  OP_CHECK(spatialLastDim == 2,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last axis of spatialShape should be 2, bug got input %ld", spatialLastDim),
        return false);

  auto locationShape = location->GetViewShape();
  int64_t locationLastDim = locationShape.GetDim(DIM_FIVE);
  OP_CHECK(locationLastDim == 2,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last axis of location should be 2, bug got input %ld", spatialLastDim),
        return false);

  auto valueShape = value->GetViewShape();
  int64_t numQueries = locationShape.GetDim(1);
  OP_CHECK(numQueries >= 32,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numQueries must be greater than or equal to 32, bug got input %ld", numQueries),
      return false);

  int64_t numHeads = locationShape.GetDim(2);
  OP_CHECK(numHeads <= 16,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numHeads must be less than or equal to 16, bug got input %ld", numHeads),
      return false);

  int64_t numLevels = locationShape.GetDim(3);
  OP_CHECK(numLevels <= 16,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numLevels must be less than or equal to 16, bug got input %ld", numLevels),
      return false);

  int64_t numPoints = locationShape.GetDim(4);
  OP_CHECK(numPoints <= 16,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numPoints must be less than or equal to 16, bug got input %ld", numPoints),
      return false);

  int64_t embedDims =  valueShape.GetDim(3);
  OP_CHECK(((embedDims % 8 == 0) && (embedDims <= 256)),
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "embedDims must be divisible by 8 and less than or equal to 256"),
      return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex, 
                               const aclTensor *location, const aclTensor *attnWeight, aclTensor *output,
                               uint64_t *workspaceSize, aclOpExecutor **executor) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(value, spatialShape, levelStartIndex, location, attnWeight,
                             output, workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(value, spatialShape, levelStartIndex, location, attnWeight,
                              output), ACLNN_ERR_PARAM_INVALID);
    
    // 3. 检查输入的shape
    CHECK_RET(CheckShape(value, spatialShape, levelStartIndex, location, attnWeight), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(const aclTensor *value,
                                                                   const aclTensor *spatialShape,
                                                                   const aclTensor *levelStartIndex,
                                                                   const aclTensor *location,
                                                                   const aclTensor *attnWeight,
                                                                   aclTensor *output,
                                                                   uint64_t *workspaceSize,
                                                                   aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnMultiScaleDeformableAttnFunction,
        DFX_IN(value, spatialShape, levelStartIndex, location, attnWeight),
        DFX_OUT(output));
    // 固定写法，创建OpExecutor

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(value, spatialShape, levelStartIndex, location, attnWeight, output, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (value->IsEmpty() || spatialShape->IsEmpty() || levelStartIndex->IsEmpty() ||
        location->IsEmpty() || attnWeight->IsEmpty()) {
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
    }

    // 将输入转为连续tensor
    auto valueContiguous = l0op::Contiguous(value, uniqueExecutor.get());
    CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto spatialShapeContiguous = l0op::Contiguous(spatialShape, uniqueExecutor.get());
    CHECK_RET(spatialShapeContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto levelStartIndexContiguous = l0op::Contiguous(levelStartIndex, uniqueExecutor.get());
    CHECK_RET(levelStartIndexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto locationContiguous = l0op::Contiguous(location, uniqueExecutor.get());
    CHECK_RET(locationContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto attnWeightContiguous = l0op::Contiguous(attnWeight, uniqueExecutor.get());
    CHECK_RET(attnWeightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueShape = value->GetViewShape();
    auto locationShape = location->GetViewShape();
    uint64_t numQueries =  locationShape.GetDim(1);
    uint64_t numHeads =  locationShape.GetDim(2);
    uint64_t numLevels =  locationShape.GetDim(3);
    uint64_t numPoints =  locationShape.GetDim(4);
    uint64_t embedDims =  valueShape.GetDim(3);
    uint64_t noTranspose = numLevels <= 8 && numHeads <= 8 && (embedDims == 16 || embedDims == 32) &&
                        (numPoints % 2 == 0 || numPoints == 1);
    
    if (noTranspose == 0){
        // value transpose
        const int64_t permuteValueList[] = {0, 2, 1, 3};
        int64_t valueDimNum = static_cast<int64_t>(valueContiguous->GetViewShape().GetDimNum());
        auto valueAxes = uniqueExecutor.get()->AllocIntArray(permuteValueList, valueDimNum);
        CHECK_RET(valueAxes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        valueContiguous = l0op::Transpose(valueContiguous, valueAxes, uniqueExecutor.get());
        CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // loc transpose
        const int64_t permuteLocList[] = {0, 2, 3, 4, 5, 1};
        int64_t locDimNum = static_cast<int64_t>(locationContiguous->GetViewShape().GetDimNum());
        auto locAxes = uniqueExecutor.get()->AllocIntArray(permuteLocList, locDimNum);
        CHECK_RET(locAxes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        locationContiguous = l0op::Transpose(locationContiguous, locAxes, uniqueExecutor.get());
        CHECK_RET(locationContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // attnWeight transpose
        const int64_t permuteAttnList[] = {0, 2, 3, 4, 1};
        int64_t attnDimNum = static_cast<int64_t>(attnWeightContiguous->GetViewShape().GetDimNum());
        auto attnAxes = uniqueExecutor.get()->AllocIntArray(permuteAttnList, attnDimNum);
        CHECK_RET(attnAxes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        attnWeightContiguous = l0op::Transpose(attnWeightContiguous, attnAxes, uniqueExecutor.get());
        CHECK_RET(attnWeightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 输入如果是float16/bfloat16，需要cast为float32
    auto valueCasted = l0op::Cast(valueContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(valueCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto locationCasted = l0op::Cast(locationContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(locationCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto attnWeightCasted = l0op::Cast(attnWeightContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(attnWeightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 索引和spatialShape如果非int32类型，需要强转
    auto spatialShapeCasted = l0op::Cast(spatialShapeContiguous, DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(spatialShapeCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto levelStartIndexCasted = l0op::Cast(levelStartIndexContiguous, DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(levelStartIndexCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子MultiScaleDeformableAttnFunction进行计算
    auto multiScaleDeformableAttnFunctionResult = l0op::MultiScaleDeformableAttnFunction(valueCasted,
        spatialShapeCasted, levelStartIndexCasted, locationCasted, attnWeightCasted, uniqueExecutor.get());
    auto outputTensor = std::get<0>(multiScaleDeformableAttnFunctionResult);
    CHECK_RET(outputTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto outputCast = l0op::Cast(outputTensor, output->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
   
    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyOutput = l0op::ViewCopy(outputCast, output, uniqueExecutor.get());
    CHECK_RET(viewCopyOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMultiScaleDeformableAttnFunction(void *workspace, uint64_t workspaceSize,
                                                  aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnMultiScaleDeformableAttnFunction);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif