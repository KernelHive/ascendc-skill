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
 * \file dynamic_quant_update_scatter_tiling.cpp
 * \brief
 */
#include "dynamic_quant_update_scatter_tiling.h"
#include "op_util.h"
#include "shape_utils.h"

using namespace ge;
using namespace AscendC;

namespace optiling {  
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
    if ((ptr) == nullptr) {                                                                        \
      std::printf("nullptr error!");                                                               \
      return ge::GRAPH_FAILED;                                                                     \
    }
#define COMMON_OP_LOG_SUB(moduleId, level, OpInfo, fmt, ...)                            \
    OP_LOG_SUB(moduleId, OPPROTO_SUBMOD_NAME, level, " %s:%d OpName:[%s]" #fmt, __FUNCTION__, \
            __LINE__, GetCstr(OpInfo), ##__VA_ARGS__)

#define unlikely(x) __builtin_expect((x), 0)
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
    do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
        OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
        return return_value;                                                                                     \
    }                                                                                                          \
} while (0)

#define OP_TILING_CHECK(cond, log_func, expr)   \
    do {                                        \
      if (cond) {                               \
        log_func;                               \
        expr;                                   \
      }                                         \
} while (0)

}

namespace optiling {
constexpr size_t VAR_DIM = 4;
constexpr size_t ONE_INDICES = 1;
constexpr size_t TWO_INDICES = 2;
constexpr size_t INDEX_VAR = 0;
constexpr size_t INDEX_VAR_SCALE = 1;
constexpr size_t INDEX_INDICES = 2;
constexpr size_t INDEX_UPDATES = 3;
constexpr size_t INDEX_SMOOTH_SCALES = 4;
constexpr size_t TILING_MODE_AXIS_NEG_2 = 100;
constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_BATCH = 101;
constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_QUANT = 102;
constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_ELE_LARGE_QUANT = 103;
constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LITTLE_QUANT = 104;
constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LARGE_QUANT = 105;
constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_MOD_64 = 106;
constexpr size_t BYTES_ONE_BLOCK = 32;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t MULTICORE_THRESHOLD = 1024;
constexpr size_t DOUBLE = 2;
constexpr size_t UB_SIZE_ALL = 262144;
constexpr size_t RESERVED_BYTES = 65536;  // 910 fp32 need more ub to transpose
constexpr size_t DIM_NEG_2 = -2;
constexpr size_t DIM_NEG_1 = -1;
constexpr size_t FLOAT_NUM_ONE_RPT = 128;
constexpr size_t UB_SIZE_REV = 4096;
constexpr size_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr size_t MASK = 64;
constexpr size_t STRIDE_MAX_MASK = 32;
constexpr int64_t MAX_RPT_FLOAT_NUM = 255 * 64;

struct DynamicQuantUpdateScatterComputeParams {
  gert::Shape varOriginShape;
  gert::Shape varScalesOriginShape;
  gert::Shape indicesOriginShape;
  gert::Shape updateOriginShape;
  gert::Shape smoothScalesShape;
  int64_t varDtypeSize;
  int64_t varScalesDtypeSize;
  int64_t indexDtypeSize;
  int64_t actualCoreNum;
  int64_t indicesShapeRank;
  int64_t ubSize;
  int64_t indexUbSize;
  int64_t updateDtypeSize;
  int64_t caleUbSize;
  int64_t maxUbFreeSize;
  int64_t varElements;
  int64_t varScalesElements;
  int64_t indexElements;
  int64_t oldDims;
  int64_t smoothScalesElements;
  int64_t smoothScalesDtypeSize;
  int64_t smoothScaleOrgUbsize;
  int64_t smoothScaleF32Ubsize;
  int64_t updateF32UbSize;
  int64_t updateOriUbSize;
  int64_t varQuantUbSize;
  int64_t varMergedLastDimSize;
  int64_t varOrigLastDimSize;
  int64_t varScalesMergedLastDimSize;
  int64_t updatesElements;
};

struct TilingParams {
  int64_t tilingMode;
  int64_t coreNum;            // ceil_div(updates_shape[0] * updates_shape[1], eachCoreBsNum)
  int64_t eachCoreBsNum;      // ceil_div(updates_shape[0] * updates_shape[1], ori_core_num)
  int64_t lastCoreBsNum;      // updates_shape[0] * updates_shape[1] - eachCoreBsNum * (core_num - 1)
  int64_t updateAxisShape;    // updates_shape[axis]
  int64_t srcBsStride;        // updates_shape[2] * updates_shape[3]
  int64_t dstBsStride;        // data_shape[2] *  data_shape[3]
  int64_t indexElements;      // index_shape[0]
  int64_t numHead;            // data_shape[1]
  int64_t sizePerHead;        // not axis shape, data_shape[2] or data_shape[3]
  int64_t sizeSrcPerHead;     // not axis shape, data_shape[2] or data_shape[3]
  int64_t dataAxisShape;      // data_shape of axis
  int64_t numOneBlock;        // 32 / dtype_size
  int64_t innerLoopEle;       // nums of ele per inner loop
  int64_t innerLoopTimes;     // inner loop time
  int64_t innerLoopTail;      // nums of ele tail inner loop
  int64_t indicesShapeRank;   // rank of indices shape
  int64_t srcFirBsStride;     // updates_shape[1] * updates_shape[2] * updates_shape[3]
  int64_t dstFirSecBsStride;  // data_shape[1] * data_shape[2] *  data_shape[3]
  int64_t updateDim0;         // updates_shape[0]
  int64_t updateDim1;         // updates_shape[1]
  int64_t varElements;
  int64_t varScalesElements;
  int64_t updatesElements;
  int64_t quantReptNum;
  int64_t varOrigLastDimSize;
  int64_t innerLoopFullRpt;
  int64_t innerLoopTimesLastCore;
  int64_t innerLoopTailLastCore;
};

static int64_t NewAxis(int64_t axis, DynamicQuantUpdateScatterComputeParams& computeParams) {
  int64_t newAxis = axis < 0 ? (computeParams.oldDims + axis) : axis;
  if (0 < newAxis && newAxis < computeParams.oldDims - 1) {
    newAxis = static_cast<int64_t>(DIM_2);
  }
  return newAxis;
}

float GetUpdateUbRatio(const DynamicQuantUpdateScatterComputeParams& computeParams, bool hasSmoothScales,
                       bool isLittleQuant) {
  int64_t floatSize = ge::GetSizeByDataType(ge::DT_FLOAT);
  int64_t intSize = ge::GetSizeByDataType(ge::DT_INT32);
  int64_t totalPart = computeParams.varDtypeSize + computeParams.updateDtypeSize + computeParams.smoothScalesDtypeSize +
                      computeParams.varScalesDtypeSize;
  if (!isLittleQuant) {
    totalPart = totalPart + intSize;
    if (computeParams.updateDtypeSize != floatSize) {
      totalPart += floatSize;
    }

    if (hasSmoothScales && (computeParams.smoothScalesDtypeSize != floatSize) &&
        (computeParams.smoothScalesDtypeSize != 0)) {
      totalPart += floatSize;
    }
  }

  float ratio = computeParams.updateDtypeSize * 1.0 / totalPart;

  return ratio;
}

void CalcTilingDataForLargeEleLargeQuant(TilingParams& param,
                                         const DynamicQuantUpdateScatterComputeParams& computeParams) {
  // calc update can use ub
  float ratio = GetUpdateUbRatio(computeParams, true, false);

  int64_t maxInnerLoopEle = static_cast<int64_t>(computeParams.caleUbSize * ratio);

  param.innerLoopEle = maxInnerLoopEle / computeParams.updateDtypeSize / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT;
  param.innerLoopEle = (param.innerLoopEle > MAX_RPT_FLOAT_NUM)
                           ? MAX_RPT_FLOAT_NUM / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT
                           : param.innerLoopEle;
  param.innerLoopTimes = param.varOrigLastDimSize / param.innerLoopEle;
  param.innerLoopTail = param.varOrigLastDimSize % param.innerLoopEle;

  param.srcFirBsStride = computeParams.updateOriginShape.GetDim(1) * computeParams.updateOriginShape.GetDim(DIM_2) *
                         computeParams.updateOriginShape.GetDim(DIM_3);
  param.dstFirSecBsStride = computeParams.varOriginShape.GetDim(1) * computeParams.varOriginShape.GetDim(DIM_2) *
                            computeParams.varOriginShape.GetDim(DIM_3);
  param.updateDim0 = computeParams.updateOriginShape.GetDim(0);
  param.updateDim1 = computeParams.updateOriginShape.GetDim(1);

  return;
}

void CalcTilingDataForLargeBatchLargeQuant(TilingParams& param,
                                           const DynamicQuantUpdateScatterComputeParams& computeParams) {
  // calc update can use ub
  float ratio = GetUpdateUbRatio(computeParams, true, false);
  int64_t maxInnerLoopEle = static_cast<int64_t>(computeParams.caleUbSize * ratio);

  param.innerLoopEle = maxInnerLoopEle / computeParams.updateDtypeSize / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT;
  param.innerLoopEle = (param.innerLoopEle > MAX_RPT_FLOAT_NUM)
                           ? MAX_RPT_FLOAT_NUM / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT
                           : param.innerLoopEle;
  param.innerLoopTimes = param.varOrigLastDimSize / param.innerLoopEle;
  param.innerLoopTail = param.varOrigLastDimSize % param.innerLoopEle;
  param.updateAxisShape = computeParams.updateOriginShape.GetDim(DIM_2);

  param.srcFirBsStride = computeParams.updateOriginShape.GetDim(1) * computeParams.updateOriginShape.GetDim(DIM_2) *
                         computeParams.updateOriginShape.GetDim(DIM_3);
  param.dstFirSecBsStride = computeParams.varOriginShape.GetDim(1) * computeParams.varOriginShape.GetDim(DIM_2) *
                            computeParams.varOriginShape.GetDim(DIM_3);
  param.updateDim0 = computeParams.updateOriginShape.GetDim(0);
  param.updateDim1 = computeParams.updateOriginShape.GetDim(1);

  return;
}

void CalcTilingDataForLargeEleLittleQuant(TilingParams& param,
                                          const DynamicQuantUpdateScatterComputeParams& computeParams) {
  // calc update can use ub
  int64_t maxInnerLoopEle =
      computeParams.caleUbSize - computeParams.smoothScaleOrgUbsize - computeParams.smoothScaleF32Ubsize;

  float ratio = GetUpdateUbRatio(computeParams, false, false);

  maxInnerLoopEle = static_cast<int64_t>(maxInnerLoopEle * ratio);

  param.innerLoopEle = maxInnerLoopEle / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT / computeParams.updateDtypeSize /
                       computeParams.varOrigLastDimSize * computeParams.varOrigLastDimSize;
  param.innerLoopEle = (param.innerLoopEle > MAX_RPT_FLOAT_NUM)
                           ? MAX_RPT_FLOAT_NUM / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT /
                                 computeParams.varOrigLastDimSize * computeParams.varOrigLastDimSize
                           : param.innerLoopEle;

  param.innerLoopFullRpt = param.innerLoopEle / computeParams.updateOriginShape.GetDim(DIM_3);
  param.innerLoopTimes = param.eachCoreBsNum / param.innerLoopFullRpt;
  param.innerLoopTail = param.eachCoreBsNum % param.innerLoopFullRpt;
  param.innerLoopTimesLastCore = param.lastCoreBsNum / param.innerLoopFullRpt;
  param.innerLoopTailLastCore = param.lastCoreBsNum % param.innerLoopFullRpt;

  param.srcFirBsStride = computeParams.updateOriginShape.GetDim(1) * computeParams.updateOriginShape.GetDim(DIM_2) *
                         computeParams.updateOriginShape.GetDim(DIM_3);
  param.dstFirSecBsStride = computeParams.varOriginShape.GetDim(1) * computeParams.varOriginShape.GetDim(DIM_2) *
                            computeParams.varOriginShape.GetDim(DIM_3);
  param.updateDim0 = computeParams.updateOriginShape.GetDim(0);
  param.updateDim1 = computeParams.updateOriginShape.GetDim(1);

  return;
}

void CalcTilingDataForLargeBatchLittleQuant(TilingParams& param,
                                            const DynamicQuantUpdateScatterComputeParams& computeParams) {
  // calc update can use ub
  int64_t floatSize = ge::GetSizeByDataType(ge::DT_FLOAT);
  int64_t intSize = ge::GetSizeByDataType(ge::DT_INT32);
  int64_t maxInnerLoopEle = computeParams.caleUbSize - computeParams.smoothScaleOrgUbsize -
                            computeParams.smoothScaleF32Ubsize -
                            computeParams.smoothScalesElements * (floatSize + intSize);

  float ratio = GetUpdateUbRatio(computeParams, false, true);

  maxInnerLoopEle = static_cast<int64_t>(maxInnerLoopEle * ratio);

  param.innerLoopEle = maxInnerLoopEle / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT / computeParams.updateDtypeSize /
                       computeParams.varOrigLastDimSize * computeParams.varOrigLastDimSize;
  param.innerLoopEle = (param.innerLoopEle > MAX_RPT_FLOAT_NUM)
                           ? MAX_RPT_FLOAT_NUM / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT /
                                 computeParams.varOrigLastDimSize * computeParams.varOrigLastDimSize
                           : param.innerLoopEle;

  param.srcFirBsStride = computeParams.updateOriginShape.GetDim(1) * computeParams.updateOriginShape.GetDim(DIM_2) *
                         computeParams.updateOriginShape.GetDim(DIM_3);
  param.dstFirSecBsStride = computeParams.varOriginShape.GetDim(1) * computeParams.varOriginShape.GetDim(DIM_2) *
                            computeParams.varOriginShape.GetDim(DIM_3);
  param.updateDim0 = computeParams.updateOriginShape.GetDim(0);
  param.updateDim1 = computeParams.updateOriginShape.GetDim(1);

  return;
}

int64_t CalcQuantUbSize(const gert::TilingContext* context, const DynamicQuantUpdateScatterComputeParams& computeParams,
                        TilingParams& param, int64_t batchNum) {
  int64_t floatSize = ge::GetSizeByDataType(ge::DT_FLOAT);

  // calc smoothScale need ubsize
  int64_t smoothScaleOrgUbsize = computeParams.smoothScalesElements * computeParams.smoothScalesDtypeSize;
  int64_t smoothScaleF32Ubsize =
      (computeParams.smoothScalesDtypeSize == floatSize) ? 0 : computeParams.smoothScalesElements * floatSize;
  // calc updates need ubsize
  int64_t updateOriUbSize = param.srcBsStride * batchNum * computeParams.updateDtypeSize;
  int64_t updateF32UbSize = (computeParams.updateDtypeSize == floatSize) ? 0 : param.srcBsStride * batchNum * floatSize;
  // for reduce max or float16 temp calc ub
  int64_t tempF32 = param.srcBsStride * batchNum * floatSize;
  int64_t tempI32 = param.srcBsStride * batchNum * floatSize;

  // calc varout need ubsize
  int64_t varUbSize = param.srcBsStride * batchNum * computeParams.varDtypeSize;

  // calc varscales need ubsize
  int64_t varscalesUbSize = param.srcBsStride * batchNum * floatSize / computeParams.varOrigLastDimSize;

  return (smoothScaleOrgUbsize + smoothScaleF32Ubsize + updateOriUbSize + updateF32UbSize + tempF32 + tempI32 +
          varUbSize + varscalesUbSize);
}

void SetLargeEleLitteQuantTilingMode(TilingParams& param, const DynamicQuantUpdateScatterComputeParams& computeParams) {
  int64_t maxStide = MASK * STRIDE_MAX_MASK;
  if ((computeParams.varOrigLastDimSize % MASK == 0) && (computeParams.varOrigLastDimSize < maxStide)) {
    param.tilingMode = TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_MOD_64;
  } else {
    param.tilingMode = TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_QUANT;
  }
  return;
}

ge::graphStatus GetTilingNeg2(const gert::TilingContext* context, TilingParams& param,
                              const DynamicQuantUpdateScatterComputeParams& computeParams) {
  param.sizePerHead = computeParams.varOriginShape.GetDim(DIM_3);
  param.dataAxisShape = computeParams.varOriginShape.GetDim(DIM_2);
  param.sizeSrcPerHead = computeParams.updateOriginShape.GetDim(DIM_3);
  param.updateAxisShape = computeParams.updateOriginShape.GetDim(DIM_2);

  if (CalcQuantUbSize(context, computeParams, param, 1) > static_cast<int64_t>(computeParams.maxUbFreeSize)) {
    // need to check updateOriginShape[0] * updateOriginShape[1] and updateOriginShape[2],  for choose different tiling
    // current use updateOriginShape[2] to cut core
    int64_t maxInnerLoopEle = computeParams.caleUbSize * GetUpdateUbRatio(computeParams, true, false) /
                              BYTES_ONE_BLOCK * BYTES_ONE_BLOCK / computeParams.updateDtypeSize;

    if (computeParams.updateOriginShape.GetDim(0) * computeParams.updateOriginShape.GetDim(1) <
        computeParams.updateOriginShape.GetDim(DIM_2)) {
      int64_t updateOriginShapeDim2 = computeParams.updateOriginShape.GetDim(DIM_2);
      param.eachCoreBsNum = ops::CeilDiv(updateOriginShapeDim2, computeParams.actualCoreNum);
      param.coreNum = ops::CeilDiv(updateOriginShapeDim2, param.eachCoreBsNum);
      param.lastCoreBsNum = updateOriginShapeDim2 - param.eachCoreBsNum * (param.coreNum - 1);

      if (maxInnerLoopEle > computeParams.updateOriginShape.GetDim(DIM_3)) {
        SetLargeEleLitteQuantTilingMode(param, computeParams);
        CalcTilingDataForLargeEleLittleQuant(param, computeParams);
      } else {
        param.tilingMode = TILING_MODE_AXIS_NEG_2_LARGE_ELE_LARGE_QUANT;
        CalcTilingDataForLargeEleLargeQuant(param, computeParams);
      }
    } else {
      if (maxInnerLoopEle > computeParams.varOrigLastDimSize) {
        param.tilingMode = TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LITTLE_QUANT;
        CalcTilingDataForLargeBatchLittleQuant(param, computeParams);
      } else {
        param.tilingMode = TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LARGE_QUANT;
        CalcTilingDataForLargeBatchLargeQuant(param, computeParams);
      }
    }
  } else if (CalcQuantUbSize(context, computeParams, param, param.eachCoreBsNum) >
             static_cast<int64_t>(computeParams.maxUbFreeSize)) {
    param.tilingMode = TILING_MODE_AXIS_NEG_2_LARGE_BATCH;
  } else {
    param.tilingMode = TILING_MODE_AXIS_NEG_2;
  }

  OP_LOGD(context->GetNodeName(), "tiling mode %ld", param.tilingMode);

  return ge::GRAPH_SUCCESS;
}

static void UpdateTilingParam(DynamicQuantUpdateScatterComputeParams& computeParams, TilingParams& param) {
  param.eachCoreBsNum = ops::CeilDiv(computeParams.updateOriginShape.GetDim(0) * computeParams.updateOriginShape.GetDim(1),
                                computeParams.actualCoreNum);
  param.coreNum = ops::CeilDiv(computeParams.updateOriginShape.GetDim(0) * computeParams.updateOriginShape.GetDim(1),
                          param.eachCoreBsNum);
  param.lastCoreBsNum = computeParams.updateOriginShape.GetDim(0) * computeParams.updateOriginShape.GetDim(1) -
                        param.eachCoreBsNum * (param.coreNum - 1);
  param.srcBsStride = computeParams.updateOriginShape.GetDim(DIM_2) * computeParams.updateOriginShape.GetDim(DIM_3);
  param.indexElements = computeParams.indexElements;
  param.varElements = computeParams.varElements;
  param.varScalesElements = computeParams.varScalesElements;
  param.updatesElements = computeParams.updatesElements;
  computeParams.indexUbSize = param.indexElements * computeParams.indexDtypeSize;
  computeParams.smoothScaleOrgUbsize = computeParams.smoothScalesElements * computeParams.smoothScalesDtypeSize;
  computeParams.smoothScaleF32Ubsize = (computeParams.smoothScalesDtypeSize == ge::GetSizeByDataType(ge::DT_FLOAT))
                                           ? 0
                                           : computeParams.smoothScalesElements * ge::GetSizeByDataType(ge::DT_FLOAT);

  param.quantReptNum = computeParams.updateOriginShape.GetDim(DIM_3) / computeParams.varOrigLastDimSize;
  param.varOrigLastDimSize = computeParams.varOrigLastDimSize;

  return;
}

ge::graphStatus GetTilingParam(gert::TilingContext* context, DynamicQuantUpdateScatterComputeParams& computeParams,
                               TilingParams& param) {
  // # tiling policy
  auto axis = context->GetAttrs()->GetAttrPointer<int64_t>(1);
  int64_t tmpAxis = *axis;
  tmpAxis = NewAxis(*axis, computeParams);
  param.indicesShapeRank = computeParams.indicesShapeRank;
  // Use single core to prevent multicore trampling
  if (tmpAxis == static_cast<int64_t>(DIM_3) &&
      computeParams.updateOriginShape.GetDim(DIM_3) * static_cast<int64_t>(DOUBLE) <
          computeParams.varOriginShape.GetDim(DIM_3) &&
      param.indicesShapeRank == TWO_INDICES) {
    computeParams.actualCoreNum = 1;
  }

  OP_TILING_CHECK(computeParams.actualCoreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is zero."),
                  return ge::GRAPH_FAILED);

  computeParams.varMergedLastDimSize = computeParams.updateOriginShape[DIM_3];
  computeParams.varScalesMergedLastDimSize = computeParams.varMergedLastDimSize / computeParams.varOrigLastDimSize;

  UpdateTilingParam(computeParams, param);
  computeParams.caleUbSize = computeParams.ubSize - computeParams.indexUbSize - UB_SIZE_REV;
  computeParams.maxUbFreeSize = computeParams.caleUbSize;

  OP_TILING_CHECK(computeParams.ubSize < static_cast<int64_t>(computeParams.indexUbSize + UB_SIZE_REV),
                  OP_LOGE(context->GetNodeName(), "index_size is larger to ub."), return ge::GRAPH_FAILED);

  param.dstBsStride = computeParams.varOriginShape.GetDim(DIM_2) * computeParams.varOriginShape.GetDim(DIM_3);
  param.numHead = computeParams.varOriginShape.GetDim(1);
  if (tmpAxis == static_cast<int64_t>(DIM_2) || tmpAxis == static_cast<int64_t>(DIM_NEG_2)) {
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != GetTilingNeg2(context, param, computeParams),
                    OP_LOGE(context->GetNodeName(), "some case not support."), return ge::GRAPH_FAILED);
  } else {
    OP_LOGE(context->GetNodeName(), "DynamicQuantUpdateScatter not support, axis is -1");
    return ge::GRAPH_FAILED;
  }
  param.numOneBlock = BYTES_ONE_BLOCK / computeParams.varDtypeSize;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareTilingParams(const gert::TilingContext* context,
                                    DynamicQuantUpdateScatterComputeParams& computeParams) {
  // get coreNum and ubSize
  auto compileInfo = reinterpret_cast<const DynamicQuantUpdateScatterCompileInfo*>(context->GetCompileInfo());
  computeParams.ubSize = compileInfo->ubSize;
  computeParams.actualCoreNum = compileInfo->coreNum;

  // get input_shape
  auto dataShape = context->GetInputShape(INDEX_VAR);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dataShape);
  auto varScalesShape = context->GetInputShape(INDEX_VAR_SCALE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, varScalesShape);
  auto indicesShape = context->GetInputShape(INDEX_INDICES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
  auto updatesShape = context->GetInputShape(INDEX_UPDATES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, updatesShape);

  computeParams.varOriginShape = ops::EnsureNotScalar(dataShape->GetOriginShape());
  computeParams.varScalesOriginShape = ops::EnsureNotScalar(varScalesShape->GetOriginShape());
  computeParams.indicesOriginShape = ops::EnsureNotScalar(indicesShape->GetOriginShape());
  computeParams.updateOriginShape = ops::EnsureNotScalar(updatesShape->GetOriginShape());
  computeParams.varElements = computeParams.varOriginShape.GetShapeSize();
  computeParams.varScalesElements = computeParams.varScalesOriginShape.GetShapeSize();
  computeParams.indexElements = computeParams.indicesOriginShape.GetShapeSize();
  computeParams.updatesElements = computeParams.updateOriginShape.GetShapeSize();

  computeParams.varOrigLastDimSize = computeParams.updateOriginShape[computeParams.updateOriginShape.GetDimNum() - 1];
  auto smoothScalesShape = context->GetOptionalInputShape(INDEX_SMOOTH_SCALES);
  if (smoothScalesShape == nullptr) {
    computeParams.smoothScalesElements = 0;
  } else {
    computeParams.smoothScalesShape = ops::EnsureNotScalar(smoothScalesShape->GetOriginShape());
    computeParams.smoothScalesElements = computeParams.smoothScalesShape.GetShapeSize();
  }

  // get varDtypeSize and indexDtypeSize
  auto dataDesc = context->GetInputDesc(INDEX_VAR);
  computeParams.varDtypeSize = ge::GetSizeByDataType(dataDesc->GetDataType());

  auto varScalesDesc = context->GetInputDesc(INDEX_VAR_SCALE);
  computeParams.varScalesDtypeSize = ge::GetSizeByDataType(varScalesDesc->GetDataType());

  auto indicesDesc = context->GetInputDesc(INDEX_INDICES);
  computeParams.indexDtypeSize = ge::GetSizeByDataType(indicesDesc->GetDataType());

  auto updateDesc = context->GetInputDesc(INDEX_UPDATES);
  computeParams.updateDtypeSize = ge::GetSizeByDataType(updateDesc->GetDataType());

  OP_TILING_CHECK((computeParams.varDtypeSize == 0) || (computeParams.indexDtypeSize == 0) ||
                      (computeParams.varScalesDtypeSize == 0) || (computeParams.updateDtypeSize == 0),
                  OP_LOGE(context->GetNodeName(), "some dtpye size is Zero "), return ge::GRAPH_FAILED);

  auto smoothScalesDesc = context->GetOptionalInputDesc(INDEX_SMOOTH_SCALES);
  if (smoothScalesDesc != nullptr) {
    const ge::DataType smoothScalesDtype = smoothScalesDesc->GetDataType();
    computeParams.smoothScalesDtypeSize = ge::GetSizeByDataType(smoothScalesDtype);
  } else {
    computeParams.smoothScalesDtypeSize = 0;
  }

  computeParams.indicesShapeRank = computeParams.indicesOriginShape.GetDimNum();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus VerifyQuantParam(const gert::TilingContext* context,
                                 DynamicQuantUpdateScatterComputeParams& computeParams) {
  if (computeParams.smoothScalesElements == 0) {
    return ge::GRAPH_SUCCESS;
  }

  int64_t dimNum = computeParams.updateOriginShape.GetDimNum();
  if (computeParams.smoothScalesElements != computeParams.updateOriginShape[dimNum - 1]) {
    return ge::GRAPH_FAILED;
  }

  OP_TILING_CHECK(computeParams.smoothScalesElements != computeParams.updateOriginShape[dimNum - 1],
                  OP_LOGE(context->GetNodeName(), "smoothScalesElements is %ld, updateOriginShape[-1] is %ld",
                          computeParams.smoothScalesElements, computeParams.updateOriginShape[dimNum - 1]),
                  return ge::GRAPH_FAILED);

  auto updatesDesc = context->GetInputDesc(INDEX_UPDATES);
  const ge::DataType updatesDtype = updatesDesc->GetDataType();
  auto smoothScalesDesc = context->GetOptionalInputDesc(INDEX_SMOOTH_SCALES);
  if (smoothScalesDesc != nullptr) {
    const ge::DataType smoothScalesDtype = smoothScalesDesc->GetDataType();
    OP_TILING_CHECK(smoothScalesDtype != updatesDtype,
                    OP_LOGE(context->GetNodeName(), "updatesDtype is not the same as smoothScalesDtype"),
                    return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus VerifyNullTenosr(const gert::TilingContext* context,
                                 DynamicQuantUpdateScatterComputeParams& computeParams) {
  OP_TILING_CHECK(computeParams.varOriginShape.GetDimNum() != computeParams.updateOriginShape.GetDimNum(),
                  OP_LOGE(context->GetNodeName(), "date should be same dims with update."), return ge::GRAPH_FAILED);

  OP_TILING_CHECK(computeParams.varOriginShape.GetDimNum() * computeParams.indicesShapeRank == 0,
                  OP_LOGE(context->GetNodeName(), "date or indice shouldn't be null."), return ge::GRAPH_FAILED);

  const char* reduceAxisPtr = context->GetAttrs()->GetAttrPointer<char>(0);
  std::string reduceAxis(reduceAxisPtr);

  OP_TILING_CHECK(reduceAxis != "update" && reduceAxis != "none" && reduceAxis != "",
                  OP_LOGE(context->GetNodeName(), "reduce is %s not supported.", reduceAxis.c_str()),
                  return ge::GRAPH_FAILED);

  int64_t dataNum = computeParams.varOriginShape.GetShapeSize();
  int64_t indicesNum = computeParams.indicesOriginShape.GetShapeSize();
  int64_t updateNum = computeParams.updateOriginShape.GetShapeSize();
  OP_TILING_CHECK(dataNum == 0 || indicesNum == 0 || updateNum == 0,
                  OP_LOGE(context->GetNodeName(), "date %ld or indice %ld or updata %ld shape shouldn't be zero.",
                          dataNum, indicesNum, updateNum),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MergeDims(const gert::TilingContext* context, DynamicQuantUpdateScatterComputeParams& computeParams) {
  const auto axis = context->GetAttrs()->GetAttrPointer<int64_t>(1);

  int64_t tmpAbsAxis = int64_t(*axis);
  tmpAbsAxis = tmpAbsAxis < 0 ? (computeParams.varOriginShape.GetDimNum() + tmpAbsAxis) : tmpAbsAxis;

  int64_t oldDims = computeParams.varOriginShape.GetDimNum();
  computeParams.oldDims = oldDims;

  OP_TILING_CHECK(tmpAbsAxis < 0 || tmpAbsAxis >= oldDims,
                  OP_LOGE(context->GetNodeName(), "axis should be less than data dims."), return ge::GRAPH_FAILED);

  size_t absAxis = size_t(tmpAbsAxis);
  OP_TILING_CHECK(absAxis == computeParams.varOriginShape.GetDimNum() - 1 || absAxis == 0,
                  OP_LOGD(context->GetNodeName(), "axis be first or last dims,connot merge dims."),
                  return ge::GRAPH_SUCCESS);
  gert::Shape dataNewShape;
  gert::Shape updataNewShape;

  dataNewShape.SetDimNum(0);
  updataNewShape.SetDimNum(0);

  dataNewShape.AppendDim(computeParams.varOriginShape[0]);
  updataNewShape.AppendDim(computeParams.updateOriginShape[0]);

  size_t dataSecondDims = 1;
  size_t updataSecondDims = 1;
  for (size_t i = 1; i < absAxis; i++) {
    dataSecondDims *= computeParams.varOriginShape[i];
    updataSecondDims *= computeParams.updateOriginShape[i];
  }
  dataNewShape.AppendDim(dataSecondDims);
  updataNewShape.AppendDim(updataSecondDims);

  dataNewShape.AppendDim(computeParams.varOriginShape[absAxis]);
  updataNewShape.AppendDim(computeParams.updateOriginShape[absAxis]);

  size_t dataFourthDims = 1;
  size_t updataFourthDims = 1;
  for (size_t i = absAxis + 1; i < computeParams.varOriginShape.GetDimNum(); i++) {
    dataFourthDims *= computeParams.varOriginShape[i];
    updataFourthDims *= computeParams.updateOriginShape[i];
  }

  dataNewShape.AppendDim(dataFourthDims);
  updataNewShape.AppendDim(updataFourthDims);

  computeParams.varOriginShape = dataNewShape;
  computeParams.updateOriginShape = updataNewShape;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus VerifyTilingParams(const gert::TilingContext* context,
                                   DynamicQuantUpdateScatterComputeParams& computeParams) {
  OP_TILING_CHECK(computeParams.updateOriginShape[0] != computeParams.indicesOriginShape[0],
                  OP_LOGE(context->GetNodeName(), "updateOriginShape[0] should be same with indicesOriginShape[0]."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(computeParams.updateOriginShape[0] > computeParams.varOriginShape[0],
                  OP_LOGE(context->GetNodeName(), "updateOriginShape[0] should be less than varOriginShape[0]."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(computeParams.updateOriginShape[1] != computeParams.varOriginShape[1],
                  OP_LOGE(context->GetNodeName(), "updateOriginShape[1] should be same with varOriginShape[1]."),
                  return ge::GRAPH_FAILED);
  if (computeParams.indicesShapeRank == TWO_INDICES) {
    OP_TILING_CHECK(computeParams.indicesOriginShape[1] != 2,
                    OP_LOGE(context->GetNodeName(), "when discrete, indicesOriginShape[1] should be 2."),
                    return ge::GRAPH_FAILED);
  }

  int64_t eleOneBlock = BYTES_ONE_BLOCK / computeParams.varDtypeSize;

  const auto axis = context->GetAttrs()->GetAttrPointer<int64_t>(1);
  int64_t tmpAxis = *axis;
  tmpAxis = NewAxis(*axis, computeParams);
  if (tmpAxis == static_cast<int64_t>(DIM_2) || tmpAxis == static_cast<int64_t>(DIM_NEG_2)) {
    OP_TILING_CHECK(computeParams.varOriginShape[DIM_3] % eleOneBlock != 0,
                    OP_LOGE(context->GetNodeName(), "varOriginShape[3] should be 32B align."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(computeParams.updateOriginShape[DIM_3] != computeParams.varOriginShape[DIM_3],
                    OP_LOGE(context->GetNodeName(), "updateOriginShape[3] should be same with varOriginShape[3]."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(computeParams.updateOriginShape[DIM_3] % eleOneBlock != 0,
                    OP_LOGE(context->GetNodeName(), "updateOriginShape[3] should be 32B align."),
                    return ge::GRAPH_FAILED);
  } else if (tmpAxis == static_cast<int64_t>(DIM_3) || tmpAxis == static_cast<int64_t>(DIM_NEG_1)) {
    OP_TILING_CHECK(computeParams.varOriginShape[DIM_3] % eleOneBlock != 0,
                    OP_LOGE(context->GetNodeName(), "varOriginShape[3] should be 32B align."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(computeParams.varOriginShape[DIM_2] % eleOneBlock != 0,
                    OP_LOGE(context->GetNodeName(), "varOriginShape[2] should be 32B align when axis is -1."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(computeParams.updateOriginShape[DIM_2] != computeParams.varOriginShape[DIM_2],
                    OP_LOGE(context->GetNodeName(), "updateOriginShape[2] should be same with varOriginShape[2]."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(computeParams.updateOriginShape[DIM_2] % eleOneBlock != 0,
                    OP_LOGE(context->GetNodeName(), "updateOriginShape[2] should be 32B align."),
                    return ge::GRAPH_FAILED);
  } else {
    OP_LOGE(context->GetNodeName(), "axis only support -1 or -2!");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static void PrintDebugInfo(const gert::TilingContext* context, const TilingParams& param) {
  OP_LOGD(context->GetNodeName(),
          "[DynamicQuantUpdateScatter]tilingMode:%ld, coreNum:%ld, eachCoreBsNum:%ld, lastCoreBsNum:%ld, "
          "updateAxisShape:%ld, srcBsStride:%ld, dstBsStride:%ld, "
          "indexElements:%ld, numHead:%ld, sizePerHead:%ld, dataAxisShape:%ld, numOneBlock:%ld, "
          "innerLoopEle:%ld, innerLoopTimes:%ld, innerLoopTail:%ld, indices_shape_rank:%ld, varOrigLastDimSize:%ld,",
          param.tilingMode, param.coreNum, param.eachCoreBsNum, param.lastCoreBsNum, param.updateAxisShape,
          param.srcBsStride, param.dstBsStride, param.indexElements, param.numHead, param.sizePerHead,
          param.dataAxisShape, param.numOneBlock, param.innerLoopEle, param.innerLoopTimes, param.innerLoopTail,
          param.indicesShapeRank, param.varOrigLastDimSize);
  OP_LOGD(context->GetNodeName(),
          "[DynamicQuantUpdateScatter]srcFirBsStride:%ld, dstFirSecBsStride:%ld, updateDimFirst:%ld, "
          "varScalesElements:%ld, varElements:%ld, innerLoopFullRpt:%ld, innerLoopTimesLastCore:%ld, "
          "innerLoopTailLastCore:%ld quantRNum:%ld",
          param.srcFirBsStride, param.dstFirSecBsStride, param.updateDim0, param.varScalesElements, param.varElements,
          param.innerLoopFullRpt, param.innerLoopTimesLastCore, param.innerLoopTailLastCore, param.quantReptNum);
}

static void SaveTilingDate(gert::TilingContext* context, const TilingParams& param) {
  DynamicQuantUpdateScatterTilingData dynamicQuantUpdateScatterTiling;
  dynamicQuantUpdateScatterTiling.set_coreNum(param.coreNum);
  dynamicQuantUpdateScatterTiling.set_eachCoreBsNum(param.eachCoreBsNum);
  dynamicQuantUpdateScatterTiling.set_lastCoreBsNum(param.lastCoreBsNum);
  dynamicQuantUpdateScatterTiling.set_updateAxisShape(param.updateAxisShape);
  dynamicQuantUpdateScatterTiling.set_srcBsStride(param.srcBsStride);
  dynamicQuantUpdateScatterTiling.set_dstBsStride(param.dstBsStride);
  dynamicQuantUpdateScatterTiling.set_indexElements(param.indexElements);
  dynamicQuantUpdateScatterTiling.set_numHead(param.numHead);
  dynamicQuantUpdateScatterTiling.set_sizePerHead(param.sizePerHead);
  dynamicQuantUpdateScatterTiling.set_dataAxisShape(param.dataAxisShape);
  dynamicQuantUpdateScatterTiling.set_numOneBlock(param.numOneBlock);
  dynamicQuantUpdateScatterTiling.set_innerLoopEle(param.innerLoopEle);
  dynamicQuantUpdateScatterTiling.set_innerLoopTimes(param.innerLoopTimes);
  dynamicQuantUpdateScatterTiling.set_innerLoopTail(param.innerLoopTail);
  dynamicQuantUpdateScatterTiling.set_indicesShapeRank(param.indicesShapeRank);
  dynamicQuantUpdateScatterTiling.set_srcFirBsStride(param.srcFirBsStride);
  dynamicQuantUpdateScatterTiling.set_dstFirSecBsStride(param.dstFirSecBsStride);
  dynamicQuantUpdateScatterTiling.set_updateDim0(param.updateDim0);
  dynamicQuantUpdateScatterTiling.set_updateDim1(param.updateDim1);
  dynamicQuantUpdateScatterTiling.set_varElements(param.varElements);
  dynamicQuantUpdateScatterTiling.set_varScalesElements(param.varScalesElements);
  dynamicQuantUpdateScatterTiling.set_updatesElements(param.updatesElements);
  dynamicQuantUpdateScatterTiling.set_quantReptNum(param.quantReptNum);
  dynamicQuantUpdateScatterTiling.set_varOrigLastDimSize(param.varOrigLastDimSize);
  dynamicQuantUpdateScatterTiling.set_sizeSrcPerHead(param.sizeSrcPerHead);
  dynamicQuantUpdateScatterTiling.set_innerLoopFullRpt(param.innerLoopFullRpt);
  dynamicQuantUpdateScatterTiling.set_innerLoopTimesLastCore(param.innerLoopTimesLastCore);
  dynamicQuantUpdateScatterTiling.set_innerLoopTailLastCore(param.innerLoopTailLastCore);

  size_t* workSpaces = context->GetWorkspaceSizes(1);
  workSpaces[0] = WORK_SPACE_SIZE;
  dynamicQuantUpdateScatterTiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                               context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(dynamicQuantUpdateScatterTiling.GetDataSize());
  context->SetBlockDim(param.coreNum);
  context->SetTilingKey(param.tilingMode);

  return;
}

static ge::graphStatus Tiling4DynamicQuantUpdateScatterCache(gert::TilingContext* context) {
  DynamicQuantUpdateScatterComputeParams computeParams;
  TilingParams param;
  (void)memset_s(&param, sizeof(param), 0, sizeof(param));
  (void)memset_s(&computeParams, sizeof(computeParams), 0, sizeof(computeParams));

  OP_TILING_CHECK(PrepareTilingParams(context, computeParams) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "PrepareTilingParams failed!"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(VerifyNullTenosr(context, computeParams) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "VerifyNullTenosr failed!"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(VerifyQuantParam(context, computeParams) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "VerifyQuantParam failed!"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(MergeDims(context, computeParams) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "MergeDims failed!"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(VerifyTilingParams(context, computeParams) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "VerifyTilingParams failed!"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(GetTilingParam(context, computeParams, param) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "GetTilingParam failed!"), return ge::GRAPH_FAILED);

  PrintDebugInfo(context, param);
  SaveTilingDate(context, param);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForDynamicQuantUpdateScatter(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingForDynamicQuantUpdateScatter in");

  auto dataShape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dataShape);
  auto indicesShape = context->GetInputShape(INDEX_INDICES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

  const auto& indicesOriginShape = ops::EnsureNotScalar(indicesShape->GetOriginShape());

  if (indicesOriginShape.GetDimNum() == ONE_INDICES || indicesOriginShape.GetDimNum() == TWO_INDICES) {
    return Tiling4DynamicQuantUpdateScatterCache(context);
  }

  OP_LOGE(context->GetNodeName(), "DynamicQuantUpdateScatter only support dim of indices_shape is 1 or 2");

  return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepareForDynamicQuantUpdateScatter(gert::TilingParseContext* context) {
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

  auto compileInfoPtr = context->GetCompiledInfo<DynamicQuantUpdateScatterCompileInfo>();
  OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
  OP_TILING_CHECK((compileInfoPtr->coreNum <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                  "TilingPrepareForDynamicQuantUpdateScatter get core num failed."),
                  return ge::GRAPH_FAILED);

  uint64_t ubSize = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  compileInfoPtr->ubSize = (int64_t)ubSize;

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicQuantUpdateScatter)
    .Tiling(TilingForDynamicQuantUpdateScatter)
    .TilingParse<DynamicQuantUpdateScatterCompileInfo>(TilingPrepareForDynamicQuantUpdateScatter);
}  // namespace optiling
