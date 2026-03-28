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
 * \file group_norm_swish_tiling.cpp
 * \brief
 */

#include "group_norm_swish_tiling.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_templates_registry.h"

namespace optiling {
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_GAMMA = 1;
static const uint64_t INPUT_IDX_BETA = 2;
static const uint64_t X_SHAPE_MIN_LEN = 2;
static const uint64_t ATTR_IDX_NUM_GROUPS = 0;
static const uint64_t ATTR_IDX_EPSILON = 2;
static const uint64_t ATTR_IDX_ACTIVATE_SWISH = 3;
static const uint64_t ATTR_IDX_SWISH_SCALE = 4;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t FLOAT32_BYTES = 4;
static const uint64_t FLOAT16_BYTES = 2;
static const int64_t X_TWO_BYTES_TILING_KEY = 100;
static const int64_t X_FOUR_BYTES_TILING_KEY = 200;
static const int64_t X_GAMMA_SAME_TILING_KEY = 10;
static const int64_t X_GAMMA_DIFF_TILING_KEY = 20;
static const int64_t SHAPE_HW1_TILING_KEY = 1;
static const int64_t SHAPE_SMALL_TILING_KEY = 2;
static const int64_t SHAPE_NORM_TILING_KEY = 3;
static const int64_t SHAPE_LARGE_TILING_KEY = 4;
static const int64_t ProcessSize = 8192;
static const int64_t numPerBlock = 8;
static const uint64_t RESERVED_WORKSPACE_SIZE_ATLAS_A2 = 16 * 1024 * 1024;
static const uint64_t RESERVED_WORKSPACE_SIZE_310P = 2 * 1024 * 1024;

inline static int64_t CeilDiv(int64_t value, int64_t factor) {
  if (factor == 0) {
    return value;
  } else if (value % factor == 0) {
    return value / factor;
  } else {
    return value / factor + 1;
  }
}

inline static int64_t CeilInt(int64_t value, int64_t factor) {
  return CeilDiv(value, factor) * factor;
}

void PrintTilingData(gert::TilingContext *context,
                     GroupNormSwishTilingData &tilingData) {
  auto nodeName = context->GetNodeName();
  OPS_LOG_D(nodeName, ">>>>>>>>>>>>>>> Start to print GroupNormSwish tiling "
                      "data <<<<<<<<<<<<<<<<");
  OPS_LOG_D(nodeName, "numGroups is %lu.", tilingData.get_numGroups());
  OPS_LOG_D(nodeName, "epsilon is %f.", tilingData.get_epsilon());
  OPS_LOG_D(nodeName, "activateSwish is %lu.", tilingData.get_activateSwish());
  OPS_LOG_D(nodeName, "swishScale is %f.", tilingData.get_swishScale());
  OPS_LOG_D(nodeName, "hwNum is %lu.", tilingData.get_hwNum());
  OPS_LOG_D(nodeName, "shapeC is %lu.", tilingData.get_shapeC());
  OPS_LOG_D(nodeName, "shapeCAlign is %lu.", tilingData.get_shapeCAlign());
  OPS_LOG_D(nodeName, "shapeD is %lu.", tilingData.get_shapeD());
  OPS_LOG_D(nodeName, "numPerGroup is %lu.", tilingData.get_numPerGroup());
  OPS_LOG_D(nodeName, "groupPerCore is %lu.", tilingData.get_groupPerCore());
  OPS_LOG_D(nodeName, "groupLastCore is %lu.", tilingData.get_groupLastCore());
  OPS_LOG_D(nodeName, "groupPerCoreAlign is %lu.",
            tilingData.get_groupPerCoreAlign());
  OPS_LOG_D(nodeName, "numPerLoop is %lu.", tilingData.get_numPerLoop());
  OPS_LOG_D(nodeName, "loopTimes is %lu.", tilingData.get_loopTimes());
  OPS_LOG_D(nodeName, "loopTimesAlign is %lu.",
            tilingData.get_loopTimesAlign());
  OPS_LOG_D(nodeName, "numTailLoop is %lu.", tilingData.get_numTailLoop());
  OPS_LOG_D(nodeName, "usedCoreNum is %u.", context->GetBlockDim());
  OPS_LOG_D(nodeName, "tilingKey is %lu.", context->GetTilingKey());
  OPS_LOG_D(
      nodeName,
      ">>>>>>>>>>>>>>> Print GroupNormSwish tiling data end <<<<<<<<<<<<<<<<");
}

static ge::graphStatus CheckInputParams(const gert::TilingContext *context) {
  // check x
  auto inputX = context->GetInputDesc(INPUT_IDX_X);
  OPS_LOG_E_IF_NULL(context, inputX, return ge::GRAPH_FAILED);
  auto xDtype = inputX->GetDataType();
  OPS_CHECK((xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT &&
             xDtype != ge::DT_BF16),
            OPS_REPORT_VECTOR_INNER_ERR(
                context->GetNodeName(),
                "xDtype should be FP16/BF16/FP32, please check."),
            return ge::GRAPH_FAILED);
  auto xShapePtr = context->GetInputShape(INPUT_IDX_X);
  OPS_LOG_E_IF_NULL(context, xShapePtr, return ge::GRAPH_FAILED);
  auto xShape = xShapePtr->GetStorageShape();
  uint64_t xDims = xShape.GetDimNum();
  OPS_CHECK((xDims < X_SHAPE_MIN_LEN),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                        "inputDims can't be smaller than 2."),
            return ge::GRAPH_FAILED);
  uint64_t channel = xShape.GetDim(DIM_1);
  // check gamma and beta
  auto gammaShapePtr = context->GetInputShape(INPUT_IDX_GAMMA);
  OPS_LOG_E_IF_NULL(context, gammaShapePtr, return ge::GRAPH_FAILED);
  auto gammaShape = gammaShapePtr->GetStorageShape();
  uint64_t gammaSizes = gammaShape.GetDim(DIM_0);
  OPS_CHECK(
      (gammaShape.GetDimNum() != 1 || gammaSizes != channel),
      OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                  "The shape of gamma must be"
                                  " the same as channel, currently is %lu.",
                                  gammaSizes),
      return ge::GRAPH_FAILED);
  auto betaShapePtr = context->GetInputShape(INPUT_IDX_BETA);
  OPS_LOG_E_IF_NULL(context, betaShapePtr, return ge::GRAPH_FAILED);
  auto betaShape = betaShapePtr->GetStorageShape();
  uint64_t betaSizes = betaShape.GetDim(DIM_0);
  OPS_CHECK(
      (betaShape.GetDimNum() != 1 || betaSizes != channel),
      OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                  "The shape of beta must be"
                                  " the same as channel, currently is %lu.",
                                  betaSizes),
      return ge::GRAPH_FAILED);
  auto gammaDtypePtr = context->GetInputDesc(INPUT_IDX_GAMMA);
  OPS_LOG_E_IF_NULL(context, gammaDtypePtr, return ge::GRAPH_FAILED);
  auto gammaDtype = gammaDtypePtr->GetDataType();
  auto betaDtypePtr = context->GetInputDesc(INPUT_IDX_BETA);
  OPS_LOG_E_IF_NULL(context, betaDtypePtr, return ge::GRAPH_FAILED);
  auto betaDtype = betaDtypePtr->GetDataType();
  OPS_CHECK((gammaDtype != betaDtype),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                        "The dtype of gamma and beta must"
                                        " be consistent."),
            return ge::GRAPH_FAILED);
  if (xDtype == ge::DT_FLOAT) {
    OPS_CHECK((gammaDtype != xDtype),
              OPS_REPORT_VECTOR_INNER_ERR(
                  context->GetNodeName(),
                  "The dtype of x is float32, gamma and beta must"
                  " be consistent."),
              return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context) {
  auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
  uint64_t channel = xShape.GetDim(DIM_1);
  // check num_groups
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  const int64_t *numGroups =
      attrs->GetAttrPointer<int64_t>(ATTR_IDX_NUM_GROUPS);
  OPS_CHECK((*numGroups <= 0),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeType(),
                                        "numGroups must be bigger than 0."),
            return ge::GRAPH_FAILED);
  OPS_CHECK((channel % *numGroups != 0),
            OPS_REPORT_VECTOR_INNER_ERR(
                context->GetNodeType(),
                "channel must be integer multiples of numGroups."),
            return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static void SetAttrParams(const gert::TilingContext *context,
                          GroupNormSwishTilingData &tilingData) {
  auto attrs = context->GetAttrs();
  const int64_t *numGroups =
      attrs->GetAttrPointer<int64_t>(ATTR_IDX_NUM_GROUPS);
  const float *epsilon = attrs->GetAttrPointer<float>(ATTR_IDX_EPSILON);
  const bool *activateSwish =
      attrs->GetAttrPointer<bool>(ATTR_IDX_ACTIVATE_SWISH);
  const float *swishScale = attrs->GetAttrPointer<float>(ATTR_IDX_SWISH_SCALE);
  tilingData.set_numGroups(*numGroups);
  tilingData.set_epsilon(*epsilon);
  tilingData.set_activateSwish(*activateSwish);
  tilingData.set_swishScale(*swishScale);
}

static void SetInputParams(const gert::TilingContext *context,
                           GroupNormSwishTilingData &tilingData) {
  auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
  uint64_t hwNum = 1;
  uint64_t xDims = xShape.GetDimNum();
  for (uint64_t i = 2; i < xDims; i++) {
    hwNum = hwNum * xShape.GetDim(i);
  }
  tilingData.set_shapeC(xShape.GetDim(DIM_1));
  tilingData.set_shapeD(tilingData.get_shapeC() / tilingData.get_numGroups());
  tilingData.set_hwNum(hwNum);
  tilingData.set_numPerGroup(tilingData.get_shapeD() * hwNum);
}

static void SetBlockTiling(gert::TilingContext *context,
                           GroupNormSwishTilingData &tilingData) {
  auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
  uint64_t shapeN = xShape.GetDim(DIM_0);
  int64_t totalGroup = shapeN * tilingData.get_numGroups();
  int64_t groupPerCore = CeilDiv(totalGroup, tilingData.get_totalCoreNum());
  int64_t groupPerCoreAlign = CeilInt(groupPerCore, numPerBlock);
  uint32_t usedCoreNum = CeilDiv(totalGroup, groupPerCore);
  int64_t groupLastCore = totalGroup - (usedCoreNum - 1) * groupPerCore;

  tilingData.set_groupPerCore(groupPerCore);
  tilingData.set_groupPerCoreAlign(groupPerCoreAlign);
  context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
  tilingData.set_groupLastCore(groupLastCore);
}

static void SetTilingKey(gert::TilingContext *context,
                         GroupNormSwishTilingData &tilingData) {
  auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
  uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
  auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
  uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
  int64_t shapeCAlign = CeilInt(tilingData.get_shapeC(), 32 / gammaDtypeSize);
  tilingData.set_shapeCAlign(shapeCAlign);
  int64_t tilingKey = 0;
  if (xDtypeSize == FLOAT16_BYTES) {
    tilingKey += X_TWO_BYTES_TILING_KEY;
  } else if (xDtypeSize == FLOAT32_BYTES) {
    tilingKey += X_FOUR_BYTES_TILING_KEY;
  }

  if (xDtypeSize == gammaDtypeSize) {
    tilingKey += X_GAMMA_SAME_TILING_KEY;
  } else {
    tilingKey += X_GAMMA_DIFF_TILING_KEY;
  }

  if (tilingData.get_hwNum() == 1) {
    tilingKey += SHAPE_HW1_TILING_KEY;
  } else {
    if (tilingData.get_numPerGroup() <= ProcessSize) {
      int64_t limit = ProcessSize * 2 / xDtypeSize - 8;
      if (tilingData.get_shapeCAlign() + tilingData.get_groupPerCoreAlign() <=
          limit) {
        tilingKey += SHAPE_SMALL_TILING_KEY;
      } else {
        tilingKey += SHAPE_LARGE_TILING_KEY;
      }
    } else {
      int64_t limit =
          ProcessSize * 2 / xDtypeSize - tilingData.get_loopTimesAlign() * 2;
      if (tilingData.get_shapeCAlign() + tilingData.get_groupPerCoreAlign() <=
          limit) {
        tilingKey += SHAPE_NORM_TILING_KEY;
      } else {
        tilingKey += SHAPE_LARGE_TILING_KEY;
      }
    }
  }
  context->SetTilingKey(static_cast<uint64_t>(tilingKey));
}

static void SetUbTiling(GroupNormSwishTilingData &tilingData) {
  int64_t numPerLoop = 8192;
  tilingData.set_numPerLoop(numPerLoop);
  tilingData.set_loopTimes(CeilDiv(tilingData.get_numPerGroup(), numPerLoop));
  tilingData.set_loopTimesAlign(
      CeilInt(tilingData.get_loopTimes(), numPerBlock));

  int numTailLoop = tilingData.get_numPerGroup() % numPerLoop;
  numTailLoop = numTailLoop == 0 ? numPerLoop : numTailLoop;
  tilingData.set_numTailLoop(numTailLoop);
}

ASCENDC_EXTERN_C ge::graphStatus
TilingForGroupNormSwish(gert::TilingContext *context) {
  OPS_LOG_D(context->GetNodeName(),
            "Start running TilingPrepare4GroupNormSwish.");
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  GroupNormSwishTilingData tilingData;
  tilingData.set_totalCoreNum(ascendcPlatform.GetCoreNumAiv());
  OPS_LOG_D(context->GetNodeName(), "Get total core num:%d",
            tilingData.get_totalCoreNum());
  OPS_CHECK((tilingData.get_totalCoreNum() <= 0),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                        "Failed to get core num."),
            return ge::GRAPH_FAILED);
  uint64_t ubSizePlatForm = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                 ubSizePlatForm);
  OPS_LOG_D(context->GetNodeName(), "Get total ub size:%lu",
            static_cast<int64_t>(ubSizePlatForm));
  OPS_LOG_D(context->GetNodeName(), "TilingPrepare4GroupNormSwish ends.");

  OPS_LOG_D(context->GetNodeName(), "Start running TilingForGroupNormSwish.");
  // check input && attrs params
  OPS_CHECK((CheckInputParams(context) != ge::GRAPH_SUCCESS),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                        "InputParams is invalid."),
            return ge::GRAPH_FAILED);
  OPS_CHECK((CheckAttrParams(context) != ge::GRAPH_SUCCESS),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                        "AttrParams is invalid."),
            return ge::GRAPH_FAILED);
  // set TilingData

  SetAttrParams(context, tilingData);
  SetInputParams(context, tilingData);
  SetBlockTiling(context, tilingData);
  SetUbTiling(tilingData);
  SetTilingKey(context, tilingData);

  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                          context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

  size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_ATLAS_A2;
  if (ascendcPlatform.GetSocVersion() ==
      platform_ascendc::SocVersion::ASCEND310P) {
    sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_310P;
  }
  size_t *workspaces = context->GetWorkspaceSizes(1);
  workspaces[0] = sysWorkspaceSize;

  PrintTilingData(context, tilingData);
  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus
TilingPrepare4GroupNormSwish(gert::TilingParseContext *context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNormSwish)
    .Tiling(TilingForGroupNormSwish)
    .TilingParse<GroupNormSwishCompileInfo>(TilingPrepare4GroupNormSwish);
} // namespace optiling