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
 * \file ge_glu_v2.cpp
 * \brief tiling
 */
#include <chrono>
#include "tiling/tiling_api.h"
#include "ge_glu_v2_tiling.h"
#include "register/op_impl_registry.h"

#define OPPROTO_SUBMOD_NAME "OP_COMMON"

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ret;                                                                                  \
  }
#define OP_TILING_CHECK(cond, log_func, expr)  \
  do {                                         \
    if (cond) {                                \
    std::printf(log_func);                     \
    expr;                                      \
    }                                          \
  } while (0)
}  // namespace ops

namespace optiling {
#define COMMON_OP_LOG_SUB(moduleId, level, OpInfo, fmt, ...)                            \
  OP_LOG_SUB(moduleId, OPPROTO_SUBMOD_NAME, level, " %s:%d OpName:[%s]" #fmt, __FUNCTION__, \
          __LINE__, GetCstr(OpInfo), ##__VA_ARGS__)

#define unlikely(x) __builtin_expect((x), 0)
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
}

namespace optiling {
static const uint32_t INPUT_IDX = 0;
static const uint32_t DIM_0 = 0;
static const uint32_t DIM_1 = 1;
static const uint32_t DIM_2 = 2;
static const uint32_t DIM_3 = 3;
static const uint32_t ATTR_DIM_INDEX = 0;
static const uint32_t ATTR_APPROXIMATE_INDEX = 1;
static const uint32_t ATTR_ACTIVATE_LEFT_INDEX = 2;
static const uint32_t FP16_DTYPE_BYTES = 2;
static const uint32_t BF16_DTYPE_BYTES = 2;
static const uint32_t FP32_DTYPE_BYTES = 4;
static const uint32_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
static const uint32_t SPLIT_FACTOR = 2;
static const uint32_t SPLIT_ERROR_STATUS = 10000;
static const int64_t APPROXIMATE_USING_TANH = 1;
static const int64_t APPROXIMATE_USING_ERF = 0;
static const int64_t BYTES_ONE_BLOCK = 32;

static const int64_t FP16_BLOCK_SIZE = 16;
static const int64_t BFP16_BLOCK_SIZE = 16;
static const int64_t FP32_BLOCK_SIZE = 8;

constexpr uint32_t BATCH_MODE = 1;

inline static ge::graphStatus SetTilingDataForGeGluV2(gert::TilingContext* context, GeGluV2TilingData& tilingData) {
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor) {
  int64_t valueNum = 0;
  if (factor == 0) {
    return value;
  }
  if (value % factor == 0) {
    valueNum = value / factor;
  } else {
    valueNum = value / factor + 1;
  }
  return valueNum;
}

static void GetTilingDataSmall(GeGluV2TilingData& tilingData, TilingParam& tilingParam, bool isVreduce) {
  int64_t splitSizeAlign {tilingParam.ny};
  int64_t blockSize = tilingParam.blockSize;
  int64_t group {1};
  if (isVreduce || tilingParam.ny % blockSize == 0) {
    group = tilingParam.bufSize / tilingParam.ny;
  } else {
    splitSizeAlign = (tilingParam.ny + blockSize - 1) / blockSize * blockSize;
    group = tilingParam.bufSize / splitSizeAlign;
  }

  int64_t numPerCore = CeilDiv(tilingParam.x, tilingParam.coreNum);
  int64_t realCoreNum = CeilDiv(tilingParam.x, numPerCore);

  tilingData.set_splitSize(tilingParam.ny);
  tilingData.set_group(group);
  tilingData.set_numPerCore(numPerCore);
  tilingData.set_loopNum(numPerCore / group);
  tilingData.set_nLastTailGroup(numPerCore % group);
  tilingData.set_realCoreNum(realCoreNum);

  if (realCoreNum != 0) {
    if (tilingParam.x % realCoreNum != 0) {
      int64_t tailTotalNum = tilingParam.x - numPerCore * (realCoreNum - 1);
      tilingData.set_tailLoopNum(tailTotalNum / group);
      tilingData.set_lastTailGroup(tailTotalNum % group);
    } else {
      tilingData.set_tailLoopNum(0);
      tilingData.set_lastTailGroup(0);
    }
  }
}

static void GetTilingDataBig(GeGluV2TilingData& tilingData, TilingParam& tilingParam) {
  int64_t group = tilingParam.ny / tilingParam.bufSize;
  int64_t tailNum = tilingParam.ny - group * tilingParam.bufSize;

  tilingData.set_realCoreNum(tilingParam.coreNum);
  tilingData.set_splitSize(tilingParam.bufSize);
  tilingData.set_group(group);
  tilingData.set_tailLoopNum(tailNum);
  if (tailNum == 0) {
    tilingData.set_loopNum(tilingParam.x * group);
  } else {
    tilingData.set_loopNum(tilingParam.x * (group + 1));
  }
}

static void GetFp16TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam) {
  static const int64_t FP16_BUFFER_SIZE_LIMIT = 6144;
  tilingParam.blockSize = FP16_BLOCK_SIZE;

  if (tilingParam.ny == 1) {
    static const int64_t FP16_BUFFER_SIZE_VREDUCE = 6144;
    tilingParam.bufSize = FP16_BUFFER_SIZE_VREDUCE;
    GetTilingDataSmall(tilingData, tilingParam, true);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_102) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_112);
    tilingData.set_tilingKey(tilingkey);
  } else if (tilingParam.ny <= FP16_BUFFER_SIZE_LIMIT) {
    static const int64_t FP16_BUFFER_SIZE_SMALL = 6144;
    tilingParam.bufSize = FP16_BUFFER_SIZE_SMALL;
    GetTilingDataSmall(tilingData, tilingParam, false);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_101) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_111);
    tilingData.set_tilingKey(tilingkey);
  } else {
    static const int64_t FP16_BUFFER_SIZE_BIG = 6144;
    tilingParam.bufSize = FP16_BUFFER_SIZE_BIG;
    GetTilingDataBig(tilingData, tilingParam);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_103) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_113);
    tilingData.set_tilingKey(tilingkey);
  }

  tilingData.set_blockSize(FP16_BLOCK_SIZE);
}

static void GetBf16TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam) {
  static const int64_t BFP16_BUFFER_SIZE_LIMIT = 6144;
  tilingParam.blockSize = BFP16_BLOCK_SIZE;

  if (tilingParam.ny == 1) {
    static const int64_t BFP16_BUFFER_SIZE_VREDUCE = 6144;
    tilingParam.bufSize = BFP16_BUFFER_SIZE_VREDUCE;
    GetTilingDataSmall(tilingData, tilingParam, true);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_202) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_212);
    tilingData.set_tilingKey(tilingkey);
  } else if (tilingParam.ny <= BFP16_BUFFER_SIZE_LIMIT) {
    static const int64_t BF16_BUFFER_SIZE_SMALL = 6144;
    tilingParam.bufSize = BF16_BUFFER_SIZE_SMALL;
    GetTilingDataSmall(tilingData, tilingParam, false);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_201) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_211);
    tilingData.set_tilingKey(tilingkey);
  } else {
    static const int64_t BFP16_BUFFER_SIZE_BIG = 6144;
    tilingParam.bufSize = BFP16_BUFFER_SIZE_BIG;
    GetTilingDataBig(tilingData, tilingParam);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_203) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_213);
    tilingData.set_tilingKey(tilingkey);
  }

  tilingData.set_blockSize(BFP16_BLOCK_SIZE);
}

static void GetFp32TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam) {
  static const int64_t FP32_BUFFER_SIZE_LIMIT = 6144;
  tilingParam.blockSize = FP32_BLOCK_SIZE;

  if (tilingParam.ny == 1) {
    static const int64_t FP32_BUFFER_SIZE_VREDUCE_TANH = 6144;
    static const int64_t FP32_BUFFER_SIZE_VREDUCE_ERF = 4912;
    // 根据ub大小、存活空间个数计算，32B对齐
    tilingParam.bufSize = tilingParam.approximate == 1 ? FP32_BUFFER_SIZE_VREDUCE_TANH : 
                                                         FP32_BUFFER_SIZE_VREDUCE_ERF;
    GetTilingDataSmall(tilingData, tilingParam, true);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_302) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_312);
    tilingData.set_tilingKey(tilingkey);
  } else if (tilingParam.ny <= FP32_BUFFER_SIZE_LIMIT) {
    static const int64_t FP32_BUFFER_SIZE_SMALL = 6144;
    tilingParam.bufSize = FP32_BUFFER_SIZE_SMALL;
    GetTilingDataSmall(tilingData, tilingParam, false);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_301) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_311);
    tilingData.set_tilingKey(tilingkey);
  } else {
    static const int64_t FP32_BUFFER_SIZE_BIG = 6144;
    tilingParam.bufSize = FP32_BUFFER_SIZE_BIG;
    GetTilingDataBig(tilingData, tilingParam);
    int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_303) :
                                                       static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_313);
    tilingData.set_tilingKey(tilingkey);
  }

  tilingData.set_blockSize(FP32_BLOCK_SIZE);
}

static void GetFp16TilingDataAscend310P(GeGluV2TilingData& tilingData, TilingParam& tilingParam) {
  static const int64_t FP16_BUFFER_SIZE_LIMIT = 8192;
  tilingParam.blockSize = FP16_BLOCK_SIZE;

  if (tilingParam.ny <= FP16_BUFFER_SIZE_LIMIT) {
    static const int64_t FP16_BUFFER_SIZE_SMALL = 8192;
    tilingParam.bufSize = FP16_BUFFER_SIZE_SMALL;
    GetTilingDataSmall(tilingData, tilingParam, false);
    tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_101));
  } else {
    static const int64_t FP16_BUFFER_SIZE_BIG = 8192;
    tilingParam.bufSize = FP16_BUFFER_SIZE_BIG;
    GetTilingDataBig(tilingData, tilingParam);
    tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_103));
  }

  tilingData.set_blockSize(FP16_BLOCK_SIZE);
}

static void GetFp32TilingDataAscend310P(GeGluV2TilingData& tilingData, TilingParam& tilingParam) {
  static const int64_t FP32_BUFFER_SIZE_LIMIT = 8192;
  tilingParam.blockSize = FP32_BLOCK_SIZE;

  if (tilingParam.ny <= FP32_BUFFER_SIZE_LIMIT) {
    static const int64_t FP32_BUFFER_SIZE_SMALL = 8192;
    tilingParam.bufSize = FP32_BUFFER_SIZE_SMALL;
    GetTilingDataSmall(tilingData, tilingParam, false);
    tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_301));
  } else {
    static const int64_t FP32_BUFFER_SIZE_BIG = 6144;
    tilingParam.bufSize = FP32_BUFFER_SIZE_BIG;
    GetTilingDataBig(tilingData, tilingParam);
    tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_303));
  }

  tilingData.set_blockSize(FP32_BLOCK_SIZE);
}

static ge::graphStatus CheckInputParams(gert::TilingContext* context) {
  auto input = context->GetInputTensor(INPUT_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input);

  auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
  int32_t typeSize = ge::GetSizeByDataType(dtype);

  if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT) {
    return ge::GRAPH_FAILED;
  }

  if (typeSize <= 0) {
    return ge::GRAPH_FAILED;
  }

  // How to check split dim is -1.
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GeGluV2(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

static size_t GetAttrSplitDim(const gert::TilingContext* context) {
  auto input = context->GetInputTensor(INPUT_IDX);
  auto inputShape = input->GetStorageShape();
  size_t inputShapeSize = static_cast<int64_t>(inputShape.GetDimNum());

  auto* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
  auto* attrDim = attrs->GetAttrPointer<int64_t>(ATTR_DIM_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrDim, false);
  auto splitDim = static_cast<int64_t>(*attrDim);
  OP_LOGD(context->GetNodeName(),"splitDim is %ld, inputShapeSize is %zu", splitDim, inputShapeSize);

  size_t splitDimU = splitDim < 0 ? splitDim + inputShapeSize : splitDim;
  if (splitDimU >= inputShapeSize) {
    return SPLIT_ERROR_STATUS;
  }

  if (inputShape.GetDim(splitDimU) % SPLIT_FACTOR != 0) {
    return SPLIT_ERROR_STATUS;
  }

  return splitDimU;
}

static ge::graphStatus GetTilingAttr(const gert::TilingContext* context, TilingParam& tilingParam) {
  auto* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);

  auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrActivateLeft, false);
  auto activateLeft = *attrActivateLeft;
  int64_t activateLeftInt = activateLeft ? 1 : 0;
  tilingParam.activateLeft = activateLeftInt;

  auto* attrApproximate = attrs->GetAttrPointer<int64_t>(ATTR_APPROXIMATE_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrApproximate, false);
  auto approximate = static_cast<int64_t>(*attrApproximate);

  if (approximate != 0 && approximate != 1) {
    return ge::GRAPH_FAILED;
  }

  tilingParam.approximate = approximate;

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTillingParam(const gert::TilingContext* context, TilingParam& tilingParam) {
    // get attrs of dim and activateLeft
  size_t splitDim = GetAttrSplitDim(context);
  if (splitDim == SPLIT_ERROR_STATUS) {
    return ge::GRAPH_FAILED;
  }

  auto inputShape = context->GetInputTensor(INPUT_IDX)->GetStorageShape();
  // fuse dims
  int64_t x{1}, y{1}, n{1};
  for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
    if (i < splitDim) {
      x *= inputShape.GetDim(i);
    } else if (i > splitDim) {
      y *= inputShape.GetDim(i);
    } else {
      n = inputShape.GetDim(i) / SPLIT_FACTOR;
    }
  }
  int64_t ny = n * y;

  auto platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  auto totalCoreNum = ascendcPlatform.GetCoreNumAiv();
  OP_LOGD(context->GetNodeName(), "Tiling totalCoreNum: %d", totalCoreNum);
  if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
    totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
  }
  OP_LOGD(context->GetNodeName(), "Tiling totalCoreNum: %d", totalCoreNum);
  if (totalCoreNum <= 0) {
    return ge::GRAPH_FAILED;
  }

  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  if (ubSizePlatForm <= 0) {
    return ge::GRAPH_FAILED;
  }

  auto isAscend310P = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P;

  tilingParam.x = x;
  tilingParam.ny = ny;
  tilingParam.coreNum = totalCoreNum;
  tilingParam.isAscend310P = isAscend310P;
  if (GetTilingAttr(context, tilingParam) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

static void GetTillingData(ge::DataType dtype, TilingParam& tilingParam, GeGluV2TilingData& tilingData) {
  if (tilingParam.isAscend310P) {
    if (dtype == ge::DT_FLOAT16) {
      GetFp16TilingDataAscend310P(tilingData, tilingParam);
    } else {
      GetFp32TilingDataAscend310P(tilingData, tilingParam);
    }
  } else {
    if (dtype == ge::DT_FLOAT16) {
      GetFp16TilingData(tilingData, tilingParam);
    } else if (dtype == ge::DT_BF16) {
      GetBf16TilingData(tilingData, tilingParam);
    } else {
      GetFp32TilingData(tilingData, tilingParam);
    }
  }
}

static ge::graphStatus Tiling4GeGluV2(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "Tiling4GeGluV2 enter.");
  context->SetScheduleMode(BATCH_MODE);
  if (CheckInputParams(context) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  TilingParam tilingParam;
  if (GetTillingParam(context, tilingParam) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
  GeGluV2TilingData tilingData;

  GetTillingData(dtype, tilingParam, tilingData);

  tilingData.set_ny(tilingParam.ny);
  tilingData.set_activateLeft(tilingParam.activateLeft);
  tilingData.set_approximate(tilingParam.approximate);
  if (SetTilingDataForGeGluV2(context, tilingData) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  context->SetBlockDim(tilingData.get_realCoreNum());
  context->SetTilingKey(tilingData.get_tilingKey());
  size_t* workspaces = context->GetWorkspaceSizes(1);
  workspaces[0] = WORK_SPACE_SIZE + tilingParam.coreNum * BYTES_ONE_BLOCK;

  OP_LOGD(context->GetNodeName(),
          "tilingData is splitSize:%ld, group:%ld, realCoreNum:%ld, numPerCore:%ld, loopNum:%ld, \
           tailLoopNum:%ld,nLastTailGroup:%ld, lastTailGroup:%ld, tilingKey:%ld, blockSize:%ld, \
           activateLeft: %ld, ny: %ld, approximate: %ld ",
          tilingData.get_splitSize(), tilingData.get_group(), tilingData.get_realCoreNum(),
          tilingData.get_numPerCore(), tilingData.get_loopNum(), tilingData.get_tailLoopNum(),
          tilingData.get_nLastTailGroup(), tilingData.get_lastTailGroup(), tilingData.get_tilingKey(),
          tilingData.get_blockSize(), tilingData.get_activateLeft(), tilingData.get_ny(),
          tilingData.get_approximate());

  OP_LOGD(context->GetNodeName(), "Tiling4GeGluV2 exit.");
  return ge::GRAPH_SUCCESS;
}

struct GeGluV2CompileInfo {};

IMPL_OP_OPTILING(GeGluV2).Tiling(Tiling4GeGluV2).TilingParse<GeGluV2CompileInfo>(TilingPrepare4GeGluV2);
}  // namespace optiling
