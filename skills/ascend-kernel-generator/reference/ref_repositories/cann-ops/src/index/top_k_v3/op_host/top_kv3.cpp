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
 * \file top_kv3_tiling.cc
 * \brief
 */
#include <iostream>
#include "top_kv3_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling
namespace optiling {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t FP16_BLK_SIZE = 16;
constexpr uint32_t UB_FACTOR_B16 = 3072;
constexpr uint32_t ROW_FACTOR = 2;

static const size_t INDEX_ATTR_LARGEST = 2;
static const size_t BLOCK_SIZE = 32;

template <typename T>
static T CeilDiv(T x, T y) {
    return y == 0 ? x : (x + y - 1) / y;
}

static ge::graphStatus GetShapeParameters(gert::TilingContext* context, uint32_t& numRow, uint32_t& numCol,
                                          int32_t& kValue, bool& largest) {
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape xShape = xStorageShape->GetStorageShape();

    size_t xDimNum = xShape.GetDimNum();
    numRow = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        numRow *= xShape.GetDim(i);
    }
    numCol = xShape.GetDim(xDimNum - 1);
    auto tensorK = context->GetInputTensor(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tensorK);
    const int32_t* constDataPtr = tensorK->GetData<int32_t>();
    OP_TILING_CHECK(constDataPtr == nullptr, OP_LOGE("TopKV3", "Get const data k failed."), return ge::GRAPH_FAILED);
    kValue = static_cast<int32_t>(*constDataPtr);
    OP_TILING_CHECK(kValue == 0, OP_LOGE("TopKV3", "Get k equals to zero."), return ge::GRAPH_FAILED);
    auto* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrLargest = attrs->GetAttrPointer<bool>(INDEX_ATTR_LARGEST);
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrLargest);
    largest = *attrLargest;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4TopKV3(gert::TilingContext* context) {
    OP_LOGD(context->GetNodeName(), " Tiling4TopKV3 is running.");
    TopKV3TilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAic();
    uint32_t ubFactor = UB_FACTOR_B16;

    uint32_t numRow, numCol;
    int32_t kValue = 0;
    bool largest;
    OP_TILING_CHECK(GetShapeParameters(context, numRow, numCol, kValue, largest) != ge::GRAPH_SUCCESS,
                    OP_LOGE("TopKV3", "TopKV3 get parameters failed."), return ge::GRAPH_FAILED);
    uint32_t rowFactor = 1;
    while (rowFactor * kValue % FP16_BLK_SIZE != 0) {
        rowFactor = rowFactor * ROW_FACTOR;  // find a minimum make rowFactor * kValue % 16 = 0;
    }

    uint32_t blockFactor = rowFactor;
    uint32_t tileNum = CeilDiv(numRow, totalCoreNum * blockFactor);
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

    context->SetBlockDim(useCoreNum);

    auto *dataDesc = context->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dataDesc);
    auto dataType = dataDesc->GetDataType();
    uint32_t dtypeKey = DTYPE_KEY_FP16;
    OP_TILING_CHECK(dataType != ge::DT_FLOAT16, OP_LOGE(context->GetNodeName(), "TopKV3 only support float16."),
                    return ge::GRAPH_FAILED);

    uint32_t tilingKey = dtypeKey;
    context->SetTilingKey(tilingKey);

    tiling.set_numRow(numRow);
    tiling.set_numCol(numCol);
    tiling.set_blockFactor(blockFactor);
    tiling.set_rowFactor(rowFactor);
    tiling.set_ubFactor(ubFactor);
    tiling.set_kValue(kValue);
    tiling.set_largest(largest ? 1 : 0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = BLOCK_SIZE;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4TopKV3(gert::TilingParseContext* context) {
    OP_LOGD(context->GetNodeName(), "TilingPrepare4TopKV3 running.");
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    // no vector core enabled
    OP_TILING_CHECK((coreNum <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "TilingPrepare4TopKV3 fail to get core num."),
                    return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    OP_LOGD(context->GetNodeName(), "TilingPrepare4TopKV3 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TopKV3)
    .Tiling(Tiling4TopKV3)
    .TilingInputsDataDependency({1});

}  // namespace optiling

namespace ge{
namespace ops {
static constexpr int OUTPUT_VALUES_INDEX = 0;
static constexpr int OUTPUT_INDICES_INDEX = 1;
static constexpr int INPUT_X_INDEX = 0;
static bool InferShapeForTopKCommon(gert::InferShapeContext* context, int64_t k, const int64_t* dim) {
  const gert::Shape *input_x_shape = context->GetInputShape(INPUT_X_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);
  size_t dim_size = input_x_shape->GetDimNum();
  if (dim_size <= 0) {
    OP_LOGE(context->GetNodeName(), "The dims_in size should more than 0!");
    return GRAPH_FAILED;
  }
  int64_t sorted_axis = dim_size - 1;

  if (dim != nullptr) {
    sorted_axis = *dim;
    if (sorted_axis < 0) {
      sorted_axis += dim_size;
    }
    if (sorted_axis >= static_cast<int64_t>(dim_size)) {
      OP_LOGE(context->GetNodeName(), "Dim is out of shape size.");
      return GRAPH_FAILED;
    }
  }

  gert::Shape *output_values_shape = context->GetOutputShape(OUTPUT_VALUES_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_values_shape);
  gert::Shape *output_indices_shape = context->GetOutputShape(OUTPUT_INDICES_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_indices_shape);

  output_values_shape->SetDimNum(dim_size);
  output_indices_shape->SetDimNum(dim_size);
  for (size_t i = 0; i < dim_size; i++) {
    if (static_cast<int64_t>(i) == sorted_axis) {
      output_values_shape->SetDim(i, k);
      output_indices_shape->SetDim(i, k);
      continue;
    }
    output_values_shape->SetDim(i, input_x_shape->GetDim(i));
    output_indices_shape->SetDim(i, input_x_shape->GetDim(i));
  }
  return GRAPH_SUCCESS;
}

static graphStatus InferShapeForTopKV2D(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do TopKV2DInferShape");
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t *dim = attrs->GetInt(1);
  const gert::Tensor *input_k_tensor = context->GetInputTensor(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_k_tensor);
  DataType input_k_dtype = input_k_tensor->GetDataType();
  if (input_k_dtype == DT_INT32) {
    const int32_t *k = input_k_tensor->GetData<int32_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, k);
    return InferShapeForTopKCommon(context, *k, dim);
  } else if (input_k_dtype == DT_INT64) {
    const int64_t *k = input_k_tensor->GetData<int64_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, k);
    return InferShapeForTopKCommon(context, *k, dim);
  } else {
    OP_LOGE(context->GetNodeName(), "The type of k Error!");
    return GRAPH_FAILED;
  }
}

static graphStatus InferDataType4TopKV3(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4TopKV3");
  context->SetOutputDataType(OUTPUT_VALUES_INDEX, context->GetInputDataType(INPUT_X_INDEX));
  context->SetOutputDataType(OUTPUT_INDICES_INDEX, ge::DT_INT32);
  OP_LOGD(context->GetNodeName(), "End to do InferDataType4TopKV3");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TopKV3)
    .InferShape(InferShapeForTopKV2D)
    .InputsDataDependency({1})
    .InferDataType(InferDataType4TopKV3);
}  // namespace ops
}  // namespace ge