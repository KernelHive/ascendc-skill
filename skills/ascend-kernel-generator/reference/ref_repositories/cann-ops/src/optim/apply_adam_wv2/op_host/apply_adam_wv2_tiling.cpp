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
 * @file apply_adam_wv2_tiling.cpp
 */
#include "apply_adam_wv2_tiling.h"
#include "graph/utils/type_utils.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include <cmath>

using namespace std;
using namespace ge;
namespace {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)        \
if ((ptr) == nullptr)                                    \
{                                                        \
    std::printf("nullptr error!");                       \
    return ge::GRAPH_SUCCESS;                            \
}                                                        \

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
  
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
}

namespace optiling {

static const int64_t WORKSPACE_SIZE = static_cast<int64_t>(16) * 1024 * 1024;
static const int64_t ONE_BLK_SIZE = 32;
static const int64_t ONE_BLK_NUM = 16;
static const int64_t ONE_BLK_NUM_FP32 = 8;

static const size_t INDEX_IN_VAR = 0;
static const size_t INDEX_IN_M = 1;
static const size_t INDEX_IN_V = 2;
static const size_t INDEX_IN_GRAD = 3;
static const size_t INDEX_IN_STEP = 4;
static const size_t INDEX_IN_MAX_GRAD_NORM = 5;
static const size_t INDEX_ATTR_LR = 0;
static const size_t INDEX_ATTR_BETA1 = 1;
static const size_t INDEX_ATTR_BETA2 = 2;
static const size_t INDEX_ATTR_WEIGHT_DECAY = 3;
static const size_t INDEX_ATTR_EPS = 4;
static const size_t INDEX_ATTR_AMSGRAD = 5;
static const size_t INDEX_ATTR_MAXIMIZE = 6;

inline static ge::graphStatus ApplyAdamWV2SetTilingData(gert::TilingContext* context,
                                                        ApplyAdamWV2TilingData& tilingData) {
  if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
      return ge::GRAPH_FAILED;
  }
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(ApplyAdamWV2TilingData& tilingData) {
  OP_LOGD("ApplyAdamWV2", "totalCoreNum: %ld", tilingData.get_totalCoreNum());
  OP_LOGD("ApplyAdamWV2", "handleExtraLoopCoreNum: %ld", tilingData.get_handleExtraLoopCoreNum());
  OP_LOGD("ApplyAdamWV2", "usedCoreNum: %ld", tilingData.get_usedCoreNum());
  OP_LOGD("ApplyAdamWV2", "numPerLoop: %ld", tilingData.get_numPerLoop());
  OP_LOGD("ApplyAdamWV2", "loopNumPerCore: %ld", tilingData.get_loopNumPerCore());
  OP_LOGD("ApplyAdamWV2", "numLastLoop: %ld", tilingData.get_numLastLoop());
  OP_LOGD("ApplyAdamWV2", "isBfloat16: %ld", tilingData.get_isBfloat16());
  OP_LOGD("ApplyAdamWV2", "lr: %f", tilingData.get_lr());
  OP_LOGD("ApplyAdamWV2", "beta1: %f", tilingData.get_beta1());
  OP_LOGD("ApplyAdamWV2", "beta2: %f", tilingData.get_beta2());
  OP_LOGD("ApplyAdamWV2", "weightDecay: %f", tilingData.get_weightDecay());
  OP_LOGD("ApplyAdamWV2", "eps: %f", tilingData.get_eps());
  OP_LOGD("ApplyAdamWV2", "amsgrad: %ld", tilingData.get_amsgrad());
  OP_LOGD("ApplyAdamWV2", "maximize: %ld", tilingData.get_maximize());
  OP_LOGD("ApplyAdamWV2", "tilingKey: %ld", tilingData.get_tilingKey());
}

static inline bool IsInvalidType(const DataType dtype) {
  return dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT;
}

static inline bool IsDiffDtype(const std::vector<DataType>& dtypeLst) {
  for (uint64_t i = 1; i < dtypeLst.size(); ++i) {
    if (i == INDEX_IN_STEP) {
      continue;
    }
    if (dtypeLst[i] != dtypeLst[0]) {
      return true;
    }
  }
  return false;
}

static ge::graphStatus CheckInputDtype(const gert::TilingContext* context, ApplyAdamWV2TilingParam& tilingParam) {
  auto dtypePtr = context->GetInputDesc(INDEX_IN_VAR);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
  auto dtype = dtypePtr->GetDataType();
  OP_TILING_CHECK(IsInvalidType(dtype), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "input var dtype only support fp16, fp32, bf16 currently, please check."), return ge::GRAPH_FAILED);
  tilingParam.dtypeLst.push_back(dtype);

  dtypePtr = context->GetInputDesc(INDEX_IN_M);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
  dtype = dtypePtr->GetDataType();
  OP_TILING_CHECK(IsInvalidType(dtype), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "input m dtype only support fp16, fp32, bf16 currently, please check."), return ge::GRAPH_FAILED);
  tilingParam.dtypeLst.push_back(dtype);

  dtypePtr = context->GetInputDesc(INDEX_IN_V);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
  dtype = dtypePtr->GetDataType();
  OP_TILING_CHECK(IsInvalidType(dtype), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "input v dtype only support fp16, fp32, bf16 currently, please check."), return ge::GRAPH_FAILED);
  tilingParam.dtypeLst.push_back(dtype);

  dtypePtr = context->GetInputDesc(INDEX_IN_GRAD);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
  dtype = dtypePtr->GetDataType();
  OP_TILING_CHECK(IsInvalidType(dtype), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "input grad dtype only support fp16, fp32, bf16 currently, please check."), return ge::GRAPH_FAILED);
  tilingParam.dtypeLst.push_back(dtype);
  // grad的数据类型，用于判断cast转换时用哪种mode
  tilingParam.isBfloat16 = dtype == ge::DT_BF16 ? 1 : 0;

  dtypePtr = context->GetInputDesc(INDEX_IN_STEP);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
  dtype = dtypePtr->GetDataType();
  bool isInvalidType = dtype != ge::DT_FLOAT && dtype != ge::DT_INT64;
  OP_TILING_CHECK(isInvalidType, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "input step dtype only support fp32 and int64 currently, please check."), return ge::GRAPH_FAILED);
  tilingParam.dtypeLst.push_back(dtype);

  auto inputDesc = context->GetOptionalInputDesc(INDEX_IN_MAX_GRAD_NORM);
  OP_TILING_CHECK(tilingParam.amsgrad == 1 && inputDesc == nullptr,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "the input max_grad_norm is mandatory when the value of amsgrad is true."), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(inputDesc != nullptr && IsInvalidType(inputDesc->GetDataType()),
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "input max_grad_norm dtype only support fp16, fp32, bf16 currently, please check."), return ge::GRAPH_FAILED);
  if (inputDesc != nullptr) {
    tilingParam.dtypeLst.push_back(inputDesc->GetDataType());
  }

  return ge::GRAPH_SUCCESS;
}

static bool IsSameShape(const gert::Shape shape1, const gert::Shape shape2) {
  size_t inputShapeSize = shape1.GetDimNum();
  if (shape2.GetDimNum() != inputShapeSize) {
    return false;
  }
  for (size_t i = 0; i < inputShapeSize; ++i) {
    if (shape1.GetDim(i) != shape2.GetDim(i)) {
      return false;
    }
  }
  return true;
}

static ge::graphStatus CheckInputShape(const gert::TilingContext* context) {
  auto varShapePtr = context->GetInputShape(INDEX_IN_VAR);
  OPS_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
  auto varShape = varShapePtr->GetStorageShape();

  auto mShapePtr = context->GetInputShape(INDEX_IN_M);
  OPS_CHECK_NULL_WITH_CONTEXT(context, mShapePtr);
  auto mShape = mShapePtr->GetStorageShape();

  auto vShapePtr = context->GetInputShape(INDEX_IN_V);
  OPS_CHECK_NULL_WITH_CONTEXT(context, vShapePtr);
  auto vShape = vShapePtr->GetStorageShape();

  auto gradShapePtr = context->GetInputShape(INDEX_IN_GRAD);
  OPS_CHECK_NULL_WITH_CONTEXT(context, gradShapePtr);
  auto gradShape = gradShapePtr->GetStorageShape();

  auto stepShapePtr = context->GetInputShape(INDEX_IN_STEP);
  OPS_CHECK_NULL_WITH_CONTEXT(context, stepShapePtr);
  auto stepShape = stepShapePtr->GetStorageShape();

  auto maxGradNormShape = context->GetOptionalInputShape(INDEX_IN_MAX_GRAD_NORM);

  bool isDiffShape = !IsSameShape(varShape, mShape) || !IsSameShape(varShape, vShape) ||
    !IsSameShape(varShape, gradShape) || (maxGradNormShape != nullptr &&
    !IsSameShape(varShape, maxGradNormShape->GetStorageShape()));
  OP_TILING_CHECK(isDiffShape, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
    "var,m,v,max_grad_norm and grad should have same shape, please check."), return ge::GRAPH_FAILED);

  OP_TILING_CHECK(stepShape.GetDimNum() != 1 || stepShape.GetDim(0) != 1,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "step should have only one element, please check."),
    return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingAttr(const gert::TilingContext* context, ApplyAdamWV2TilingParam& tilingParam) {
  // get attrs of lr, beta1, beta2, weight_decay, eps, amsgrad and maximize
  auto* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  auto* attrLr = attrs->GetAttrPointer<float>(INDEX_ATTR_LR);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrLr);
  tilingParam.lr = static_cast<float>(*attrLr);

  auto* attrBeta1 = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrBeta1);
  tilingParam.beta1 = static_cast<float>(*attrBeta1);

  auto* attrBeta2 = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA2);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrBeta2);
  tilingParam.beta2 = static_cast<float>(*attrBeta2);

  auto* attrWeightDecay = attrs->GetAttrPointer<float>(INDEX_ATTR_WEIGHT_DECAY);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrWeightDecay);
  tilingParam.weightDecay = static_cast<float>(*attrWeightDecay);

  auto* attrEps = attrs->GetAttrPointer<float>(INDEX_ATTR_EPS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrEps);
  tilingParam.eps = static_cast<float>(*attrEps);

  auto* attrAmsgrad = attrs->GetAttrPointer<bool>(INDEX_ATTR_AMSGRAD);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrAmsgrad);
  auto amsgrad = *attrAmsgrad;
  int64_t amsgradInt = amsgrad ? 1 : 0;
  tilingParam.amsgrad = amsgradInt;

  auto* attrMaximize = attrs->GetAttrPointer<bool>(INDEX_ATTR_MAXIMIZE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrMaximize);
  auto maximize = *attrMaximize;
  int64_t maximizeInt = maximize ? 1 : 0;
  tilingParam.maximize = maximizeInt;

  return ge::GRAPH_SUCCESS;
}

static inline void GetTilingKey(ApplyAdamWV2TilingParam& tilingParam) {
  auto stepDtype = tilingParam.dtypeLst[INDEX_IN_STEP];
  if (IsDiffDtype(tilingParam.dtypeLst)) {
    auto gradDtype = tilingParam.dtypeLst[INDEX_IN_GRAD];
    if (gradDtype == ge::DT_FLOAT16 && stepDtype == ge::DT_FLOAT) {
      tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_FLOAT_KEY;
    } else if (gradDtype == ge::DT_FLOAT16 && stepDtype == ge::DT_INT64) {
      tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_INT64_KEY;
    } else if (gradDtype == ge::DT_BF16 && stepDtype == ge::DT_FLOAT) {
      tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_BF16_AND_STEP_FLOAT_KEY;
    } else if (gradDtype == ge::DT_BF16 && stepDtype == ge::DT_INT64) {
      tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_BF16_STEP_INT64_KEY;
    }
    tilingParam.isDiffDtype = true;
    return;
  }

  auto dtype = tilingParam.dtypeLst[0];
  if (dtype == ge::DT_BF16 && stepDtype == ge::DT_FLOAT) {
    tilingParam.tilingKey = DTYPE_BF16_AND_STEP_FLOAT_KEY;
  } else if (dtype == ge::DT_BF16 && stepDtype == ge::DT_INT64) {
    tilingParam.tilingKey = DTYPE_BF16_AND_STEP_INT64_KEY;
  } else if (dtype == ge::DT_FLOAT16 && stepDtype == ge::DT_FLOAT) {
    tilingParam.tilingKey = DTYPE_FP16_AND_STEP_FLOAT_KEY;
  } else if (dtype == ge::DT_FLOAT16 && stepDtype == ge::DT_INT64) {
    tilingParam.tilingKey = DTYPE_FP16_AND_STEP_INT64_KEY;
  } else if (dtype == ge::DT_FLOAT && stepDtype == ge::DT_FLOAT) {
    tilingParam.tilingKey = DTYPE_FP32_AND_STEP_FLOAT_KEY;
  } else if (dtype == ge::DT_FLOAT && stepDtype == ge::DT_INT64) {
    tilingParam.tilingKey = DTYPE_FP32_AND_STEP_INT64_KEY;
  }
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, ApplyAdamWV2TilingParam& tilingParam) {
  auto shapePtr = context->GetInputShape(INDEX_IN_VAR);
  OPS_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
  size_t totalDataNum = shapePtr->GetStorageShape().GetShapeSize();
  // 每个核单次可以处理的个数
  const size_t numPerLoop = 2432;
  // 所有核处理完所有数据总的循环系数
  int64_t loopNum = (totalDataNum + numPerLoop - 1) / numPerLoop;
  // 最后一个loop处理的个数
  int64_t numLastLoopActual = totalDataNum % numPerLoop;
  // 最后一个loop处理的个数32字节对齐
  int64_t numLastLoop = numLastLoopActual == 0 ? numPerLoop : numLastLoopActual;

  // 每个核要处理的多少次循环
  int64_t loopNumPerCore = loopNum / tilingParam.totalCoreNum;
  // 前多少个核需要多处理一次循环
  int64_t handleExtraLoopCoreNum = loopNum % tilingParam.totalCoreNum;
  // 实际使用的核数
  int64_t usedCoreNum = loopNumPerCore > 0 ? tilingParam.totalCoreNum : handleExtraLoopCoreNum;
  if (handleExtraLoopCoreNum == 0) {
    handleExtraLoopCoreNum = usedCoreNum;
    loopNumPerCore--;
  }
  tilingParam.numLastLoop = numLastLoop;
  tilingParam.loopNumPerCore = loopNumPerCore;
  tilingParam.numPerLoop = numPerLoop;
  tilingParam.handleExtraLoopCoreNum = handleExtraLoopCoreNum;
  tilingParam.usedCoreNum = usedCoreNum;
  return ge::GRAPH_SUCCESS;
}

static void GetTilingData(ApplyAdamWV2TilingData& tilingData, const ApplyAdamWV2TilingParam& tilingParam) {
  tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
  tilingData.set_handleExtraLoopCoreNum(tilingParam.handleExtraLoopCoreNum);
  tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
  tilingData.set_numPerLoop(tilingParam.numPerLoop);
  tilingData.set_loopNumPerCore(tilingParam.loopNumPerCore);
  tilingData.set_numLastLoop(tilingParam.numLastLoop);
  tilingData.set_isBfloat16(tilingParam.isBfloat16);
  tilingData.set_lr(tilingParam.lr);
  tilingData.set_beta1(tilingParam.beta1);
  tilingData.set_beta2(tilingParam.beta2);
  tilingData.set_weightDecay(tilingParam.weightDecay);
  tilingData.set_eps(tilingParam.eps);
  tilingData.set_amsgrad(tilingParam.amsgrad);
  tilingData.set_maximize(tilingParam.maximize);
  tilingData.set_tilingKey(tilingParam.tilingKey);
}

ge::graphStatus Tiling4ApplyAdamWV2(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "Tiling4ApplyAdamWV2 running begin");
  ApplyAdamWV2TilingParam tilingParam;
  auto compileInfo = reinterpret_cast<const ApplyAdamWV2CompileInfo*>(context->GetCompileInfo());
  tilingParam.totalCoreNum = compileInfo->totalCoreNum;
  tilingParam.ubSize = compileInfo->ubSize;

  OP_TILING_CHECK(GetTilingAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Tiling4ApplyAdamWV2 GetTilingAttr fail."),
    return ge::GRAPH_FAILED);

  OP_TILING_CHECK(CheckInputDtype(context, tilingParam) != ge::GRAPH_SUCCESS,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

  OP_TILING_CHECK(CheckInputShape(context) != ge::GRAPH_SUCCESS,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input shape check failed."), return ge::GRAPH_FAILED);

  GetTilingKey(tilingParam);

  OP_TILING_CHECK(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Tiling4ApplyAdamWV2 DoTiling fail."),
    return ge::GRAPH_FAILED);

  ApplyAdamWV2TilingData tilingData;
  GetTilingData(tilingData, tilingParam);
  OP_TILING_CHECK(ApplyAdamWV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ApplyAdamWV2SetTilingData set tiling data fail."),
    return ge::GRAPH_FAILED);
  context->SetBlockDim(tilingData.get_usedCoreNum());
  context->SetTilingKey(tilingData.get_tilingKey());
  size_t* workspaces = context->GetWorkspaceSizes(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, workspaces);
  workspaces[0] = WORKSPACE_SIZE;

  PrintTilingData(tilingData);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForApplyAdamWV2(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdamWV2)
.Tiling(Tiling4ApplyAdamWV2)
.TilingParse<ApplyAdamWV2CompileInfo>(TilingPrepareForApplyAdamWV2);

}  // namespace optiling
