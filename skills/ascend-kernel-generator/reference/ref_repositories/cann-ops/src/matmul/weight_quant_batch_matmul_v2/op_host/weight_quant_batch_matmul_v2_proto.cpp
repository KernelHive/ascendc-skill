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
 * \file weight_quant_batch_matmul_v2_infer.cc
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
using namespace std;

namespace ge {
template <typename T>
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}
}

#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define OP_CHECK(cond, log_func, ...) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      return ge::GRAPH_FAILED;                                 \
    }                                         \
  } while (0)
}  // namespace ops

namespace ops {
constexpr size_t MINIMUM_SHAPE_SIZE = 2UL;
constexpr size_t QUANT_SCALE_IDX = 4UL;
constexpr size_t ATTR_DTYPE_IDX = 3UL;
constexpr int64_t INT4_NUMS_IN_INT32 = 8UL;
const char* NODE_NAME = "WeightQuantBatchMatmulV2";

static ge::graphStatus InferDataTypeForWeightQuantBatchMatmulV2(gert::InferDataTypeContext *context)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *outputDtype = attrs->GetAttrPointer<int64_t>(ATTR_DTYPE_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputDtype);
    auto xDtype = context->GetInputDataType(0);
    auto quantScaleDtype = context->GetOptionalInputDataType(QUANT_SCALE_IDX);
    // 非onnx场景outputDtype为-1, C8时quantScaleDtype为ge::DT_UNDEFINED
    // onnx场景outputDtype不为-1，C8时outputDtype为ge::DT_INT8
    bool isOutputDtypeINT8 = (*outputDtype == -1 && quantScaleDtype != ge::DT_UNDEFINED) ||
                             (*outputDtype != -1 && static_cast<ge::DataType>(*outputDtype) == ge::DT_INT8);
    context->SetOutputDataType(0, isOutputDtypeINT8 ? ge::DT_INT8 : xDtype);
    OP_LOGD(context->GetNodeName(), "get x dtype %s, quant scale dtype %s and output dtype %ld, set y dtype %s",
            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(quantScaleDtype).GetString(),
            *outputDtype,
            ge::TypeUtils::DataTypeToAscendString(context->GetOutputDataType(0)).GetString());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShapeCheckAndInfer(const gert::Shape* xShape, const gert::Shape* weightShape,
    const ge::DataType weightDtype, gert::Shape* yShape, const bool transposeX, const bool transposeWeight)
{
    size_t numDimA = xShape->GetDimNum();
    size_t numDimB = weightShape->GetDimNum();
    OP_CHECK(std::min(numDimA, numDimB) < MINIMUM_SHAPE_SIZE ||
             (std::min(numDimA, numDimB) > MINIMUM_SHAPE_SIZE && numDimA != numDimB),
             OP_LOGE(NODE_NAME, "The shape of x and weight must >= 2 and with same batch or no batch "
                     "which are %s and %s", ge::Shape2String(*xShape).c_str(), ge::Shape2String(*weightShape).c_str()),
             return ge::GRAPH_FAILED);
    size_t mIdx = transposeX ? 1UL : 2UL;
    size_t kaIdx = transposeX ? 2UL : 1UL;
    size_t kbIdx = transposeWeight ? 1UL : 2UL;
    size_t nIdx = transposeWeight ? 2UL : 1UL;
    int64_t xK = xShape->GetDim(numDimA - kaIdx);
    int64_t weightK = weightShape->GetDim(numDimB - kbIdx);
    int64_t weightN = weightShape->GetDim(numDimB - nIdx);
    // weight int32输入图融合前infershape weight输入情况： weight前存在transpose节点：x （m, k), weight (k/8, n);
    //                                                   weight前不存在transpose节点：x （m, k), weight (k, n/8)。
    // weight int32输入图融合后infershape weight输入最后一维大小为实际值除以8。
    if (weightDtype == ge::DT_INT32 && xK > 0 && weightK > 0) {
        bool transWeightInt32 = false;
        if (!transposeX && !transposeWeight) {
            transWeightInt32 = (weightK * INT4_NUMS_IN_INT32 == xK);
        }
        if (transposeWeight || transWeightInt32) {
            weightK *= INT4_NUMS_IN_INT32;
        } else {
            weightN *= INT4_NUMS_IN_INT32;
        }
    }
    OP_CHECK(xK != weightK && xK > 0 && weightK > 0, OP_LOGE(NODE_NAME, "Ka[%ld] != Kb[%ld]", xK, weightK),
             return ge::GRAPH_FAILED);
    size_t outDimNum = std::max(numDimA, numDimB);
    yShape->SetDimNum(outDimNum);
    yShape->SetDim(outDimNum - 2, xShape->GetDim(numDimA - mIdx)); // 2:设置yShape m
    yShape->SetDim(outDimNum - 1, weightN); // 1:设置yShape n
    if (numDimA == numDimB) {
        for (size_t i = 0; i < outDimNum - MINIMUM_SHAPE_SIZE; i++) {
            OP_CHECK(xShape->GetDim(i) != weightShape->GetDim(i),
                     OP_LOGE(NODE_NAME, "batch of xShape is diff from batch of wShape"),
                     return ge::GRAPH_FAILED);
            yShape->SetDim(i, xShape->GetDim(i));
        }
    } else {
        auto longerShape = numDimA > numDimB ? xShape : weightShape;
        for (size_t i = 0; i < outDimNum - MINIMUM_SHAPE_SIZE; i++) {
            yShape->SetDim(i, longerShape->GetDim(i));
        }
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferShapeForWeightQuantBatchMatmulV2(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferShapeForWeightQuantBatchMatmulV2 begin");
    auto xShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto weightShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    auto weight = context->GetInputTensor(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weight);
    auto weightDtype = weight->GetDataType();
    auto yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *transposeX = attrs->GetAttrPointer<bool>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeX);
    const bool *transposeWeight = attrs->GetAttrPointer<bool>(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeWeight);
    OP_LOGD(context->GetNodeName(),
            "x_shape: %s, weight_shape: %s, transpose_x: %d, transpose_weight: %d, weight_dtype: %s",
            ge::Shape2String(*xShape).c_str(), ge::Shape2String(*weightShape).c_str(), *transposeX, *transposeWeight,
            ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString());

    OP_CHECK(ShapeCheckAndInfer(xShape, weightShape, weightDtype, yShape, *transposeX, *transposeWeight) !=
                ge::GRAPH_SUCCESS,
             OP_LOGE(context->GetNodeName(), "The Shape Check failed "), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(WeightQuantBatchMatmulV2)
    .InferShape(InferShapeForWeightQuantBatchMatmulV2)
    .InferDataType(InferDataTypeForWeightQuantBatchMatmulV2);
}  // namespace ops
