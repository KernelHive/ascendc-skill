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
 * \file dequant_swiglu_quant_tiling.cc
 * \brief
 */
#include <chrono>
#include "register/op_impl_registry.h"
#include "dequant_swiglu_quant_tiling.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_templates_registry.h"
using namespace AscendC;
using namespace ge;

namespace optiling {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}
constexpr int64_t ATTR_ACTIVATE_LEFT_INDEX = 0;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 1;
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_INDEX = 1;
constexpr int64_t ACTIVATION_SCALE_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t QUANT_SCALE_INDEX = 4;
constexpr int64_t QUANT_OFFSET_INDEX = 5;
constexpr int64_t INPUT_GROUP_INDEX = 6;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr uint64_t WORKSPACE_SIZE = 32;
constexpr uint64_t TILING_KEY_HAS_GROUP = 0;
constexpr uint64_t TILING_KEY_NO_GROUP = 1;
constexpr int64_t UB_REVERSE = 1024;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t PERFORMANCE_CORE_NUM = 36;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_INT32};
static const std::map<std::string, int64_t> SUPPORT_QUANT_MODE = {{"dynamic", 1}};

ge::graphStatus DequantSwigluQuantDskTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = reinterpret_cast<const DequantSwigluQuantCompileInfo*>(context_->GetCompileInfo());
    OP_TILING_CHECK(compileInfoPtr == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(context_, "compile info is null"),
                    return ge::GRAPH_FAILED);
    coreNum_ = compileInfoPtr->coreNum;
    ubSize_ = compileInfoPtr->ubSize;
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNum();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
  }

  maxPreCore_ = static_cast<int64_t>(coreNum_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckDtype() {
  auto xPtr = context_->GetInputDesc(X_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
  auto xDtype = xPtr->GetDataType();
  OP_TILING_CHECK((SUPPORT_DTYPE.find(xDtype) == SUPPORT_DTYPE.end()),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "x dtype only support int32, please check."),
                  return ge::GRAPH_FAILED);
  if (hasGroupIndex_) {
    auto groupIndexPtr = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, groupIndexPtr);
    auto groupIndexDtype = groupIndexPtr->GetDataType();
    bool dtypeInValid = groupIndexDtype != ge::DT_INT32 && groupIndexDtype != ge::DT_INT64;
    OP_TILING_CHECK(dtypeInValid,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "group_index dtype only support int32 and int64, please check."),
                    return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckForModeDynamic() {
  auto weightScalePtr = context_->GetOptionalInputDesc(WEIGHT_SCALE_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, weightScalePtr);
  auto weightScaleDtype = weightScalePtr->GetDataType();
  bool dtypeInValid = weightScaleDtype != ge::DT_FLOAT;
  OP_TILING_CHECK(dtypeInValid,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                  "weight_scale dtype only support float32, please check."),
                  return ge::GRAPH_FAILED);

  auto activationScalePtr = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, activationScalePtr);
  auto activationScaleDtype = activationScalePtr->GetDataType();
  dtypeInValid = activationScaleDtype != ge::DT_FLOAT;
  OP_TILING_CHECK(dtypeInValid,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                  "activation_scale dtype only support float32, please check."),
                  return ge::GRAPH_FAILED);

  auto quantScalePtr = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, quantScalePtr);
  auto quantScaleDtype = quantScalePtr->GetDataType();
  dtypeInValid = quantScaleDtype != ge::DT_FLOAT && quantScaleDtype != ge::DT_FLOAT16;
  OP_TILING_CHECK(dtypeInValid,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                  "quant_scale dtype only support float32 or float16, please check."),
                  return ge::GRAPH_FAILED);

  auto activationScaleShapePtr = context_->GetOptionalInputShape(ACTIVATION_SCALE_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, activationScaleShapePtr);
  auto activationScaleShape = activationScaleShapePtr->GetStorageShape();
  int64_t activationScaleNum = activationScaleShape.GetShapeSize();
  bool shapeInValid = hasGroupIndex_ ? false : (activationScaleNum != inDimx_);
  OP_TILING_CHECK(
      shapeInValid,
      VECTOR_INNER_ERR_REPORT_TILIING(
          context_->GetNodeName(),
          "the size of activation_scale must be equal to the size of x divided by the last dim size, please check."),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetAttr() {
  auto* attrs = context_->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

  auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
  actRight_ = (attrActivateLeft == nullptr || *attrActivateLeft == false) ? 1 : 0;
  std::string quantmode = "dynamic";
  auto it = SUPPORT_QUANT_MODE.find(quantmode);
  OP_TILING_CHECK(it == SUPPORT_QUANT_MODE.end(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                  "attr quant_mode only support dynamic currently, please check."),
                  return ge::GRAPH_FAILED);
  quantMode_ = it->second;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetShapeAttrsInfo() {
  if (!IsDSKCase()) {
    return ge::GRAPH_SUCCESS;
  }
  auto shapeX = context_->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
  const gert::Shape& inputShapeX = shapeX->GetStorageShape();

  int64_t inputShapeXTotalNum = inputShapeX.GetShapeSize();
  int64_t inputShapeXRank = inputShapeX.GetDimNum();
  inDimy_ = inputShapeX.GetDim(inputShapeXRank - 1);
  inDimx_ = inputShapeXTotalNum / inDimy_;
  auto shapeY = context_->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, shapeY);
  const gert::Shape& outputShapeY = shapeY->GetStorageShape();
  outDimy_ = outputShapeY.GetDim(inputShapeXRank - 1);

  OP_TILING_CHECK(inDimy_ % (BLOCK_SIZE * SWI_FACTOR) != 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                  "only support lastdimSize being divided by 64, but is %ld", inDimy_),
                  return ge::GRAPH_FAILED);

  auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
  hasGroupIndex_ = shapeGroupIndex != nullptr;
  if (hasGroupIndex_) {
    const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
    groupNum_ = inputShapeGroupIndex.GetDimNum() == 0 ? 1 : inputShapeGroupIndex.GetDim(0);
  }

  OP_TILING_CHECK(GetAttr() != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

  auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INDEX);
  OP_TILING_CHECK(
      biasShapePtr != nullptr,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "bias only support None currently, please check."),
      return ge::GRAPH_FAILED);
  OP_TILING_CHECK(CheckDtype() != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "dtype check failed."),
                  return ge::GRAPH_FAILED);
  if (quantMode_ == QUANT_MODE_DYNAMIC) {
    OP_TILING_CHECK(CheckForModeDynamic() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "CheckForModeDynamic failed."),
                    return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

bool DequantSwigluQuantDskTiling::IsDSKCase() {
  auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
  if (shapeGroupIndex != nullptr) {
    return true;
  }
  auto quantScalePtr = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
  if (quantScalePtr != nullptr) {
    auto quantScaleDtype = quantScalePtr->GetDataType();
    if (quantScaleDtype == ge::DT_FLOAT16) {
      return true;
    }
  }
  return false;
}

bool DequantSwigluQuantDskTiling::IsCapable() {
  return IsDSKCase();
}

ge::graphStatus DequantSwigluQuantDskTiling::DoOpTiling() {
  auto inputShapeX = context_->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, inputShapeX);

  /* 切分计算逻辑
  x used mem: [UbFactorDimx, outDimy_ * 2] dtype: float
  activation_scale used mem: [UbFactorDimx, 8] dtype: float
  weight_scale used mem: [UbFactorDimx, outDimy_ * 2] dtype: float
  quant_scale used mem: [1, outDimy_] dtype: float
  y used mem: [UbFactorDimx, outDimy_] dtype: int8_t
  scale used mem: [UbFactorDimx,] dtype: float
  tmp used mem: [UbFactorDimx, outDimy_ * 2] dtype: float
  x, activation_scale enable db
  ub reverse 1024B
  */
  int64_t db = 2;
  // UbFactorDimx is 1,compute maxOutDimy
  int64_t numerator = ubSize_ - UB_REVERSE - BLOCK_SIZE - db * BLOCK_ELEM * sizeof(float) - sizeof(float);
  int64_t denominator =
      5 * sizeof(float) + db * SWI_FACTOR * sizeof(float) + sizeof(int8_t);  // 和dimy相关的buffer，计算分母
  int64_t maxOutDimy = static_cast<int64_t>(numerator / denominator);
  maxOutDimy = maxOutDimy / BLOCK_SIZE * BLOCK_SIZE;
  int64_t maxInDimy = static_cast<int64_t>(maxOutDimy * SWI_FACTOR);
  OP_LOGI(context_->GetNodeName(), "Get maxInDimy[%ld]", maxInDimy);
  OP_TILING_CHECK(inDimy_ > maxInDimy,
                  VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                  "only support lastdimSize <= %ld, but is %ld", maxInDimy, inDimy_),
                  return ge::GRAPH_FAILED);

  // compute ubFactorDimx
  numerator = ubSize_ - UB_REVERSE - outDimy_ * sizeof(float) - BLOCK_SIZE;  // 减去固定buffer
  denominator = db * (outDimy_ * SWI_FACTOR + BLOCK_ELEM) * sizeof(float) + outDimy_ * sizeof(int8_t) + sizeof(float) +
                outDimy_ * SWI_FACTOR * sizeof(float) +
                outDimy_ * SWI_FACTOR * sizeof(float);  // 和dimx相关的buffer，计算分母
  int64_t ubFactorDimx = static_cast<int64_t>(numerator / denominator);
  ubFactorDimx = std::min(ubFactorDimx, inDimx_);
  maxPreCore_ = (inDimx_ + ubFactorDimx - 1) / ubFactorDimx;
  maxPreCore_ = std::min(maxPreCore_, static_cast<int64_t>(PERFORMANCE_CORE_NUM));

  tilingKey_ = hasGroupIndex_ ? TILING_KEY_HAS_GROUP : TILING_KEY_NO_GROUP;
  tilingData_.set_inDimx(inDimx_);
  tilingData_.set_inDimy(inDimy_);
  tilingData_.set_outDimy(outDimy_);
  tilingData_.set_UbFactorDimx(ubFactorDimx);
  tilingData_.set_UbFactorDimy(outDimy_);
  tilingData_.set_usedCoreNum(maxPreCore_);
  tilingData_.set_maxCoreNum(maxPreCore_);
  tilingData_.set_inGroupNum(groupNum_);
  tilingData_.set_quantMode(quantMode_);
  tilingData_.set_actRight(actRight_);

  return ge::GRAPH_SUCCESS;
}

void DequantSwigluQuantDskTiling::DumpTilingInfo() {
  std::ostringstream info;
  info << "inDimx_: " << tilingData_.get_inDimx();
  info << ", inDimy_: " << tilingData_.get_inDimy();
  info << ", outDimy: " << tilingData_.get_outDimy();
  info << ", UbFactorDimx: " << tilingData_.get_UbFactorDimx();
  info << ", UbFactorDimy: " << tilingData_.get_UbFactorDimy();
  info << ", usedCoreNum: " << tilingData_.get_usedCoreNum();
  info << ", maxCoreNum: " << tilingData_.get_maxCoreNum();
  info << ", inGroupNum: " << tilingData_.get_inGroupNum();
  info << ", quantMode: " << tilingData_.get_quantMode();
  info << ", actRight: " << tilingData_.get_actRight();
  info << ", tilingKey: " << tilingKey_;
  OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus DequantSwigluQuantDskTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t DequantSwigluQuantDskTiling::GetTilingKey() const {
  return tilingKey_;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetWorkspaceSize() {
  workspaceSize_ = WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::PostTiling() {
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(maxPreCore_);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("DequantSwigluQuant", DequantSwigluQuantDskTiling, 0);

ge::graphStatus TilingForDequantSwigluQuant(gert::TilingContext* context) {
  return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForDequantSwigluQuant(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepare4DequantSwigluQuant enter.");
  auto compileInfo = GetCompileInfoPtr<DequantSwigluQuantCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
  OP_TILING_CHECK((compileInfo->coreNum <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Get core num failed, core num: %u",
                                                  static_cast<uint32_t>(compileInfo->coreNum)),
                  return ge::GRAPH_FAILED);

  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  compileInfo->ubSize = ubSize;
  OP_TILING_CHECK((compileInfo->ubSize <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Get ub size failed, ub size: %u",
                                                  static_cast<uint32_t>(compileInfo->ubSize)),
                  return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "TilingPrepare4DequantSwigluQuant exit.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DequantSwigluQuant)
    .Tiling(TilingForDequantSwigluQuant)
    .TilingParse<DequantSwigluQuantCompileInfo>(TilingPrepareForDequantSwigluQuant);

}  // namespace optiling