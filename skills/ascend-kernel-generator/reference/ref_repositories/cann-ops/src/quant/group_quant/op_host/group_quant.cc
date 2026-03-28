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
 * \file group_quant_tiling.cc
 * \brief
 */
#include "group_quant_tiling.h"

using namespace gert;

namespace optiling {
static const uint64_t TILING_OFFSET = 1000000000000UL;
static const size_t INPUT_INDEX_OF_X = 0;
static const size_t INPUT_INDEX_OF_SCALE = 1;
static const size_t INPUT_INDEX_OF_GROUP_INDEX = 2;
static const size_t INPUT_INDEX_OF_OFFSET = 3;
static const size_t OUTPUT_INDEX_OF_Y = 0;
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const size_t DIM_NUM_OF_X = 2;
static const size_t DIM_NUM_OF_SCALE = 2;
static const size_t DIM_NUM_OF_GROUP_INDEX = 1;
static const size_t DIM_NUM_OF_OFFSET = 1;
static const size_t DIM_NUM_OF_Y = 2;
static const size_t DIM_INDEX_0 = 0;
static const size_t DIM_INDEX_1 = 1;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_INT4 = 29;
static const size_t WORKSPACES_DEFAULT_SIZE_32B = 32;
static const int64_t EVEN_FACTOR = 2;

bool GroupQuantTiling::IsCapable() {
  return true;
}

ge::graphStatus GroupQuantTiling::GetPlatformInfo() {
  OP_LOGD(context_->GetNodeName(), "GetPlatformInfo begin.");

  auto compileInfo = reinterpret_cast<const GroupQuantCompileInfo*>(context_->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
  auto platformInfoPtr = context_->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    coreNumVar = compileInfo->coreNum;
    OP_LOGD(context_->GetNodeName(), "Get core num from compile info. Core num is %ld.", coreNumVar);
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNumVar = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context_->GetNodeName(), "Get core num from platform. Core num is %ld.", coreNumVar);
  }

  OP_LOGD(context_->GetNodeName(), "GetPlatformInfo end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::GetShapeAttrsInfo() {
  OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo begin.");

  // check for input x
  auto inputX = context_->GetInputShape(INPUT_INDEX_OF_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
  auto inputXDesc = context_->GetInputDesc(INPUT_INDEX_OF_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
  auto xDtype = inputXDesc->GetDataType();
  OP_TILING_CHECK((xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                      "x datatype only support FLOAT32, FLOAT16 or BFLOAT16"),
      return ge::GRAPH_FAILED);
  auto xShape = ops::EnsureNotScalar(inputX->GetStorageShape());
  OP_TILING_CHECK(xShape.GetDimNum() != DIM_NUM_OF_X,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "x shape length should be 2"),
      return ge::GRAPH_FAILED);
  dimS = xShape.GetDim(DIM_INDEX_0);
  dimH = xShape.GetDim(DIM_INDEX_1);

  // check for input scale
  auto inputScale = context_->GetInputShape(INPUT_INDEX_OF_SCALE);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, inputScale);
  auto inputScaleDesc = context_->GetInputDesc(INPUT_INDEX_OF_SCALE);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, inputScaleDesc);
  auto scaleDtype = inputScaleDesc->GetDataType();
  OP_TILING_CHECK((scaleDtype != ge::DT_FLOAT && scaleDtype != ge::DT_FLOAT16 && scaleDtype != ge::DT_BF16),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                      "scale datatype only support FLOAT32, FLOAT16 or BFLOAT16"),
      return ge::GRAPH_FAILED);
  auto scaleShape = ops::EnsureNotScalar(inputScale->GetStorageShape());
  OP_TILING_CHECK(scaleShape.GetDimNum() != DIM_NUM_OF_SCALE,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "scale shape length should be 2"),
                  return ge::GRAPH_FAILED);
  dimE = scaleShape.GetDim(DIM_INDEX_0);
  OP_TILING_CHECK(dimE < 1,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                      "no support for the first dim of scale shape is less than 1"),
      return ge::GRAPH_FAILED);
  OP_TILING_CHECK(scaleShape.GetDim(DIM_INDEX_1) != dimH,
      VECTOR_INNER_ERR_REPORT_TILIING(
          context_->GetNodeName(),
          "the second dim of scale shape should be same as the second dim of x shape"),
      return ge::GRAPH_FAILED);

  // check for input group_index
  auto inputGrpIdx = context_->GetInputShape(INPUT_INDEX_OF_GROUP_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGrpIdx);
  auto inputGrpIdxDesc = context_->GetInputDesc(INPUT_INDEX_OF_GROUP_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGrpIdxDesc);
  auto grpIdxDtype = inputGrpIdxDesc->GetDataType();
  OP_TILING_CHECK((grpIdxDtype != ge::DT_INT32 && grpIdxDtype != ge::DT_INT64),
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                      "group_index datatype only support INT32 or INT64"),
      return ge::GRAPH_FAILED);
  auto grpIdxShape = ops::EnsureNotScalar(inputGrpIdx->GetStorageShape());
  OP_TILING_CHECK(grpIdxShape.GetDimNum() != DIM_NUM_OF_GROUP_INDEX,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "group_index shape length should be 1"),
      return ge::GRAPH_FAILED);
  OP_TILING_CHECK(grpIdxShape.GetDim(DIM_INDEX_0) != dimE,
      VECTOR_INNER_ERR_REPORT_TILIING(
          context_->GetNodeName(),
          "The first dim of group_index shape should be same as the first dim of scale shape"),
      return ge::GRAPH_FAILED);

  // check for input offset which is an optional input
  auto inputOffset = context_->GetInputShape(INPUT_INDEX_OF_OFFSET);
  if (inputOffset == nullptr) {
    hasOffset = 0;
    OP_LOGD(context_->GetNodeName(), "Input offset is not exist.");
  } else {
    hasOffset = 1;
    auto inputOffsetDesc = context_->GetInputDesc(INPUT_INDEX_OF_OFFSET);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputOffsetDesc);
    auto offsetDtype = inputOffsetDesc->GetDataType();
    OP_TILING_CHECK(offsetDtype != scaleDtype,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "offset datatype should be same as scale datatype"),
        return ge::GRAPH_FAILED);
    auto offsetShape = ops::EnsureNotScalar(inputOffset->GetStorageShape());
    OP_TILING_CHECK(offsetShape.GetDimNum() != DIM_NUM_OF_OFFSET || offsetShape.GetDim(DIM_INDEX_0) != 1,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "offset should be scalar or vector with shape is [1, ]"),
        return ge::GRAPH_FAILED);
  }

  // check attr dst_type
  auto* attrs = context_->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const int32_t* pDstType = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
  if (pDstType != nullptr) {
    int32_t dstType = *pDstType;
    OP_TILING_CHECK(dstType != DTYPE_INT8 && dstType != DTYPE_INT4,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "dst_type should be DT_INT4 or DT_INT8"),
        return ge::GRAPH_FAILED);
  }

  // check for output y
  auto outputY = context_->GetOutputShape(OUTPUT_INDEX_OF_Y);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, outputY);
  auto outputYDesc = context_->GetOutputDesc(OUTPUT_INDEX_OF_Y);
  OPS_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
  auto yDtype = outputYDesc->GetDataType();
  OP_TILING_CHECK(yDtype != ge::DT_INT4 && yDtype != ge::DT_INT8,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "y datatype only support INT4 or INT8"),
      return ge::GRAPH_FAILED);
  auto yShape = ops::EnsureNotScalar(outputY->GetStorageShape());
  OP_TILING_CHECK(
      yShape.GetDimNum() != DIM_NUM_OF_Y || yShape.GetDim(DIM_INDEX_0) != dimS || yShape.GetDim(DIM_INDEX_1) != dimH,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "y shape should be same as x shape"),
      return ge::GRAPH_FAILED);
  if (yDtype == ge::DT_INT4) {
    OP_TILING_CHECK(dimH % EVEN_FACTOR != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "y datatype is int4, the second dim of shape should be an even number"),
        return ge::GRAPH_FAILED);
  }

  OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::DoOpTiling() {
  OP_LOGD(context_->GetNodeName(), "DoOpTiling begin.");

  if (dimS == 0) {
    // kernel support empty tensor
    needCoreNum = 1;
  } else if (dimS < coreNumVar) {
    needCoreNum = dimS;
  } else {
    needCoreNum = coreNumVar;
  }
  OP_TILING_CHECK(needCoreNum < 1,
      VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "need core num should be greater than 0"),
      return ge::GRAPH_FAILED);
  if (dimS % needCoreNum == 0) {
    preCoreNum = needCoreNum;
  } else {
    preCoreNum = dimS % needCoreNum;
  }
  xRowNumPreCore = ops::CeilDiv(dimS, needCoreNum);
  xRowNumPostCore = dimS / needCoreNum;

  tilingData.set_dimS(dimS);
  tilingData.set_dimE(dimE);
  tilingData.set_dimH(dimH);
  tilingData.set_hasOffset(hasOffset);
  tilingData.set_needCoreNum(needCoreNum);
  tilingData.set_preCoreNum(preCoreNum);
  tilingData.set_xRowNumPreCore(xRowNumPreCore);
  tilingData.set_xRowNumPostCore(xRowNumPostCore);

  OP_LOGD(context_->GetNodeName(), "DoOpTiling end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t GroupQuantTiling::GetTilingKey() const {
  uint64_t tilingKey = 0;
  return tilingKey % TILING_OFFSET;
}

ge::graphStatus GroupQuantTiling::GetWorkspaceSize() {
  OP_LOGD(context_->GetNodeName(), "GetWorkspaceSize begin.");

  workspaceSize_ = WORKSPACES_DEFAULT_SIZE_32B;

  OP_LOGD(context_->GetNodeName(), "GetWorkspaceSize end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::PostTiling() {
  OP_LOGD(context_->GetNodeName(), "PostTiling begin.");

  context_->SetBlockDim(tilingData.get_needCoreNum());
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;

  gert::TilingData* rawTilingData = context_->GetRawTilingData();
  OP_LOGE_IF(rawTilingData == nullptr, ge::GRAPH_FAILED, context_->GetNodeType(), "GetRawTilingData failed.");
  OP_TILING_CHECK(
      tilingData.GetDataSize() > rawTilingData->GetCapacity(),
      VECTOR_INNER_ERR_REPORT_TILIING(context_, "actual tiling data size > context tiling data size"),
      return ge::GRAPH_FAILED);
  tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
  rawTilingData->SetDataSize(tilingData.GetDataSize());

  OP_LOGD(context_->GetNodeName(), "PostTiling end.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GroupQuant(gert::TilingContext* context) {
  // 初始化算子Tiling类
  GroupQuantTiling tiling(context);
  // 执行算子tiling框架
  return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4GroupQuant(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepare4GroupQuant begin.");

  auto compileInfo = GetCompileInfoPtr<GroupQuantCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
  OP_TILING_CHECK((compileInfo->coreNum <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Get core num failed"),
                  return ge::GRAPH_FAILED);

  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  compileInfo->ubSizePlatForm = ubSizePlatForm;
  OP_TILING_CHECK((compileInfo->ubSizePlatForm <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Get ub size failed"),
                  return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "TilingPrepare4GroupQuant end.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupQuant)
    .Tiling(Tiling4GroupQuant)
    .TilingParse<GroupQuantCompileInfo>(TilingPrepare4GroupQuant);
}  // namespace optiling
