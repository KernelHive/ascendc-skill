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
 * \file cross_entropy_loss_grad_tiling.cc
 * \brief
 */

#include "cross_entropy_loss_grad_tiling.h"

using namespace std;
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
constexpr uint64_t INPUT_GRAD_LOSS_IDX = 0;
constexpr uint64_t INPUT_LOG_PROB_IDX = 1;
constexpr uint64_t INPUT_TARGET_IDX = 2;
constexpr uint64_t INPUT_WEIGHT_IDX = 3;
constexpr uint64_t REDUCTION_ATTR_IDX = 0;
constexpr uint64_t IGNORE_INDEX_ATTR_IDX = 1;
constexpr uint64_t LABEL_SMOOTHING_ATTR_IDX = 2;

constexpr uint64_t INT8_DTYPE_SIZE = 1;
constexpr uint64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint64_t FP32_INT32_DTYPE_SIZE = 4;
constexpr uint64_t INT64_DTYPE_SIZE = 8;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t ALIGN_SIZE = 512;
constexpr uint64_t FP16_BF16_SPLIT = 8;
constexpr uint64_t FP32_SPLIT = 12;

constexpr uint64_t TILING_KEY_BFLOAT16 = 1;
constexpr uint64_t TILING_KEY_FLOAT16 = 2;
constexpr uint64_t TILING_KEY_FLOAT32 = 3;
constexpr uint64_t TILING_KEY_NONE = 0;
constexpr uint64_t TILING_KEY_MEAN = 1;
constexpr uint64_t TILING_KEY_SUM = 2;

uint64_t maxUbSize;
uint64_t coreNum;

CrossEntropyLossGradTilingData ceLossGradTiling;

int64_t GetDivRem(const int64_t value1, const int64_t value2) {
  if (value2 == 0) {
    return value1;
  }
  return value1 % value2;
}

int64_t GetCeilInt(const int64_t value1, const int64_t value2) {
  if (value2 == 0) {
    return value2;
  }
  return (value1 + value2 - 1) / value2;
}

int64_t GetDiv(const int64_t value1, const int64_t value2) {
  if (value2 == 0) {
    return value2;
  }
  return value1 / value2;
}

int64_t GetAlign(const int64_t value1, const int64_t value2) {
  if (value2 == 0) {
    return value2;
  }
  return value1 / value2 * value2;
}

int64_t GetCeilAlign(const int64_t value1, const int64_t value2) {
  if (value2 == 0) {
    return value2;
  }
  return (value1 + value2 - 1) / value2 * value2;
}

static void PrintInfo(gert::TilingContext* context) {
  auto nodeName = context->GetNodeName();
  OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print CrossEntropyLossGrad tiling data <<<<<<<<<<<<<<<<");
  OP_LOGD(nodeName, ">>> reduction: %ld", ceLossGradTiling.get_reduction());
  OP_LOGD(nodeName, ">>> ignoreIndex: %ld", ceLossGradTiling.get_ignoreIndex());
  OP_LOGD(nodeName, ">>> labelSmoothing: %f", ceLossGradTiling.get_labelSmoothing());
  OP_LOGD(nodeName, ">>> rowVal: %ld", ceLossGradTiling.get_rowVal());
  OP_LOGD(nodeName, ">>> colVal: %ld", ceLossGradTiling.get_colVal());
  OP_LOGD(nodeName, ">>> frontCoreNum: %ld", ceLossGradTiling.get_frontCoreNum());
  OP_LOGD(nodeName, ">>> tailCoreNum: %ld", ceLossGradTiling.get_tailCoreNum());
  OP_LOGD(nodeName, ">>> usedCoreNum: %ld", ceLossGradTiling.get_usedCoreNum());
  OP_LOGD(nodeName, ">>> frontRowNum: %ld", ceLossGradTiling.get_frontRowNum());
  OP_LOGD(nodeName, ">>> tailRowNum: %ld", ceLossGradTiling.get_tailRowNum());
  OP_LOGD(nodeName, ">>> alignColLoopNum: %ld", ceLossGradTiling.get_alignColLoopNum());
  OP_LOGD(nodeName, ">>> colLoop: %ld", ceLossGradTiling.get_colLoop());
  OP_LOGD(nodeName, ">>> colLoopNumTail: %ld", ceLossGradTiling.get_colLoopNumTail());
  OP_LOGD(nodeName, ">>> targetSize: %ld", ceLossGradTiling.get_targetSize());
  OP_LOGD(nodeName, ">>> targetCastSize: %ld", ceLossGradTiling.get_targetCastSize());
  OP_LOGD(nodeName, ">>> gradLossSize: %ld", ceLossGradTiling.get_gradLossSize());
  OP_LOGD(nodeName, ">>> gradLossFp32Size: %ld", ceLossGradTiling.get_gradLossFp32Size());
  OP_LOGD(nodeName, ">>> ignoreSize: %ld", ceLossGradTiling.get_ignoreSize());
  OP_LOGD(nodeName, ">>> maskSize: %ld", ceLossGradTiling.get_maskSize());
  OP_LOGD(nodeName, ">>> targetWeightSize: %ld", ceLossGradTiling.get_targetWeightSize());
  OP_LOGD(nodeName, ">>> tBuf2Size: %ld", ceLossGradTiling.get_tBuf2Size());
  OP_LOGD(nodeName, ">>> tBuf3Size: %ld", ceLossGradTiling.get_tBuf3Size());
  OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print CrossEntropyLossGrad tiling data end <<<<<<<<<<<<<<<<");
}

static void InitSplitInfo(gert::TilingContext* context, CrossEntropyLossGradTilingData& tiling) {
  auto platformInfo = context->GetPlatformInfo();
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
  coreNum = ascendcPlatform.GetCoreNumAiv();

  // 分核
  uint64_t rowVal = ceLossGradTiling.get_rowVal();
  uint64_t frontCoreNum = GetDivRem(rowVal, coreNum) != 0 ? GetDivRem(rowVal, coreNum) : coreNum;
  uint64_t tailCoreNum = rowVal <= coreNum ? 0 : coreNum - frontCoreNum;
  uint64_t blockDim = frontCoreNum + tailCoreNum;
  uint64_t frontRowNum = GetCeilInt(rowVal, coreNum);
  uint64_t tailRowNum = GetDiv(rowVal, coreNum);
  ceLossGradTiling.set_frontCoreNum(frontCoreNum);
  ceLossGradTiling.set_tailCoreNum(tailCoreNum);
  ceLossGradTiling.set_usedCoreNum(blockDim);
  ceLossGradTiling.set_frontRowNum(frontRowNum);
  ceLossGradTiling.set_tailRowNum(tailRowNum);
  context->SetBlockDim(blockDim);
}

static ge::graphStatus CELossGradCoreSplitInfo(gert::TilingContext* context, CrossEntropyLossGradTilingData& ceLossGradTiling,
                                               uint64_t dtypeSize, uint64_t reductionKey) {
  uint64_t colVal = ceLossGradTiling.get_colVal();
  uint64_t frontRowNum = ceLossGradTiling.get_frontRowNum();
  uint64_t colLoopNum = 0;  // ub一次可以处理最大数据量
  uint64_t alignColLoopNum = 0;
  uint64_t targetSize = GetCeilAlign(frontRowNum * INT64_DTYPE_SIZE, BLOCK_SIZE);
  uint64_t targetCastSize = GetCeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
  uint64_t gradLossSize = 0;
  uint64_t gradLossFp32Size = 0;
  uint64_t ignoreSize = GetCeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
  uint64_t maskSize = GetCeilAlign(frontRowNum * INT8_DTYPE_SIZE, BLOCK_SIZE);
  uint64_t targetWeightSize = 0;
  uint64_t tBuf2Size = 0;
  uint64_t tBuf3Size = 0;

  auto weightTensor = context->GetOptionalInputTensor(INPUT_WEIGHT_IDX);
  if (weightTensor == nullptr) {  // weight为None
    OP_LOGD(context->GetNodeName(), "[CrossEntropyLossGrad] In this case weight is None.");
  } else {  // weight非None
    OP_LOGD(context->GetNodeName(), "[CrossEntropyLossGrad] In this case weight is not None.");
    targetWeightSize = GetCeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
  }

  if (dtypeSize == FP32_INT32_DTYPE_SIZE) {
    if (reductionKey == TILING_KEY_NONE) {
      gradLossSize = GetCeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
      colLoopNum = GetDiv((maxUbSize - targetSize - targetCastSize - gradLossSize - ignoreSize - maskSize - targetWeightSize), FP32_SPLIT);
    } else if (reductionKey == TILING_KEY_MEAN) {
      tBuf2Size = BLOCK_SIZE;
      tBuf3Size = GetCeilAlign(max(coreNum, frontRowNum) * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
      colLoopNum = GetDiv((maxUbSize - targetSize - targetCastSize - gradLossSize - gradLossFp32Size - ignoreSize - maskSize - targetWeightSize - tBuf2Size - tBuf3Size), FP32_SPLIT);
    } else if (reductionKey == TILING_KEY_SUM) {
      colLoopNum = GetDiv((maxUbSize - targetSize - targetCastSize - ignoreSize - maskSize - targetWeightSize), FP32_SPLIT);
    }
    if (colLoopNum < colVal) {
      alignColLoopNum = GetAlign(colLoopNum, (ALIGN_SIZE / FP32_INT32_DTYPE_SIZE));
    } else {
      alignColLoopNum = GetAlign(colVal, (ALIGN_SIZE / FP32_INT32_DTYPE_SIZE));
    }
  } else if (dtypeSize == FP16_BF16_DTYPE_SIZE) {
    if (reductionKey == TILING_KEY_NONE) {
      gradLossSize = GetCeilAlign(frontRowNum * FP16_BF16_DTYPE_SIZE, BLOCK_SIZE);
      gradLossFp32Size = GetCeilAlign(frontRowNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
      colLoopNum = GetDiv((maxUbSize - targetSize - targetCastSize - gradLossSize - gradLossFp32Size - ignoreSize - maskSize - targetWeightSize), FP16_BF16_SPLIT);
    } else if (reductionKey == TILING_KEY_MEAN) {
      tBuf2Size = BLOCK_SIZE;
      tBuf3Size = GetCeilAlign(max(coreNum, frontRowNum) * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
      colLoopNum = GetDiv((maxUbSize - targetSize - targetCastSize - gradLossSize - gradLossFp32Size - ignoreSize - maskSize - targetWeightSize - tBuf2Size - tBuf3Size), FP16_BF16_SPLIT);
    } else if (reductionKey == TILING_KEY_SUM) {
      colLoopNum = GetDiv((maxUbSize - targetSize - targetCastSize - ignoreSize - maskSize - targetWeightSize), FP16_BF16_SPLIT);
    }
    if (colLoopNum < colVal) {
      // 512B对齐 ub一次可以处理最大对齐数据量
      alignColLoopNum = GetAlign(colLoopNum, (ALIGN_SIZE / FP16_BF16_DTYPE_SIZE));
    } else {
      alignColLoopNum = GetAlign(colVal, (ALIGN_SIZE / FP16_BF16_DTYPE_SIZE));
    }
  }

  uint64_t colLoop = GetDiv(colVal, alignColLoopNum);
  uint64_t colLoopNumTail = GetDivRem(colVal, alignColLoopNum);
  colLoopNumTail = alignColLoopNum != 0 ? colLoopNumTail : colVal;
  ceLossGradTiling.set_alignColLoopNum(alignColLoopNum);
  ceLossGradTiling.set_colLoop(colLoop);
  ceLossGradTiling.set_colLoopNumTail(colLoopNumTail);
  ceLossGradTiling.set_targetSize(targetSize);
  ceLossGradTiling.set_targetCastSize(targetCastSize);
  ceLossGradTiling.set_gradLossSize(gradLossSize);
  ceLossGradTiling.set_gradLossFp32Size(gradLossFp32Size);
  ceLossGradTiling.set_ignoreSize(ignoreSize);
  ceLossGradTiling.set_maskSize(maskSize);
  ceLossGradTiling.set_targetWeightSize(targetWeightSize);
  ceLossGradTiling.set_tBuf2Size(tBuf2Size);
  ceLossGradTiling.set_tBuf3Size(tBuf3Size);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingInput(gert::TilingContext* context, uint64_t& weightKey) {
  const gert::StorageShape* gradLossShape = context->GetInputShape(INPUT_GRAD_LOSS_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, gradLossShape);
  const gert::StorageShape* logProbShape = context->GetInputShape(INPUT_LOG_PROB_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, logProbShape);
  const gert::StorageShape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, targetShape);

  OP_TILING_CHECK((targetShape->GetStorageShape().GetDim(0) != logProbShape->GetStorageShape().GetDim(0)),
                   VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                   "The dim 0 of logProb should be equal to the size of target."),
                   return ge::GRAPH_FAILED);

  auto weightTensor = context->GetOptionalInputTensor(INPUT_WEIGHT_IDX);
  if (weightTensor != nullptr) {
    auto weightShape = weightTensor->GetStorageShape();
    OP_TILING_CHECK((weightShape.GetDim(0) != logProbShape->GetStorageShape().GetDim(1)),
                     VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                     "The dim 1 of logProb should be equal to the size of weight."),
                     return ge::GRAPH_FAILED);
  }
  weightKey = (weightTensor == nullptr) ? 0 : 1;

  uint64_t rowVal = 1;
  uint64_t colVal = 1;
  rowVal = logProbShape->GetStorageShape().GetDim(0);
  colVal = logProbShape->GetStorageShape().GetDim(1);
  ceLossGradTiling.set_rowVal(rowVal);
  ceLossGradTiling.set_colVal(colVal);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingAttr(gert::TilingContext* context, uint64_t& reductionKey) {
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  auto reduction = attrs->GetAttrPointer<char>(REDUCTION_ATTR_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, reduction);
  if (strcmp(reduction, "none") == 0) {
    OP_LOGD(context->GetNodeName(), "Attr reduction is none.");
    reductionKey = TILING_KEY_NONE;
  } else if (strcmp(reduction, "mean") == 0) {
    OP_LOGD(context->GetNodeName(), "Attr reduction is mean.");
    reductionKey = TILING_KEY_MEAN;
  } else if (strcmp(reduction, "sum") == 0) {
    OP_LOGD(context->GetNodeName(), "Attr reduction is sum.");
    reductionKey = TILING_KEY_SUM;
  }
  ceLossGradTiling.set_reduction(reductionKey);
  auto ignoreIndex = attrs->GetAttrPointer<int64_t>(IGNORE_INDEX_ATTR_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, ignoreIndex);
  ceLossGradTiling.set_ignoreIndex(static_cast<int64_t>(*ignoreIndex));
  auto labelSmoothing = attrs->GetAttrPointer<float>(LABEL_SMOOTHING_ATTR_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, labelSmoothing);
  ceLossGradTiling.set_labelSmoothing(static_cast<float>(*labelSmoothing));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4CrossEntropyLossGrad(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "CrossEntropyLossGradTiling tiling start");
  uint64_t weightKey = 0;
  OP_TILING_CHECK(GetTilingInput(context, weightKey) != ge::GRAPH_SUCCESS,
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetTilingInput failed."),
      return ge::GRAPH_FAILED);

  uint64_t reductionKey = 0;
  OP_TILING_CHECK(GetTilingAttr(context, reductionKey) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetTilingAttr failed."),
                  return ge::GRAPH_FAILED);

  uint64_t dtypeSize = 0;
  uint64_t dtypeKey = 0;
  auto dataType = context->GetInputDesc(INPUT_LOG_PROB_IDX)->GetDataType();
  if (dataType == ge::DT_BF16) {
    dtypeSize = FP16_BF16_DTYPE_SIZE;
    dtypeKey = TILING_KEY_BFLOAT16;
  } else if (dataType == ge::DT_FLOAT16) {
    dtypeSize = FP16_BF16_DTYPE_SIZE;
    dtypeKey = TILING_KEY_FLOAT16;
  } else if (dataType == ge::DT_FLOAT) {
    dtypeSize = FP32_INT32_DTYPE_SIZE;
    dtypeKey = TILING_KEY_FLOAT32;
  }

  uint64_t tilingKey = dtypeKey * 10 + weightKey;
  OP_LOGD(context->GetNodeName(), ">>> [CrossEntropyLossGradTiling] tilingKey: %ld", tilingKey);
  context->SetTilingKey(tilingKey);

  // 核间切分
  InitSplitInfo(context, ceLossGradTiling);
  OP_LOGD(context->GetNodeName(), "CrossEntropyLossGradTiling InitSplitInfo finished");
  // 核内切分
  CELossGradCoreSplitInfo(context, ceLossGradTiling, dtypeSize, reductionKey);
  OP_LOGD(context->GetNodeName(), "CrossEntropyLossGradTiling CELossGradCoreSplitInfo finished");

  ceLossGradTiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(ceLossGradTiling.GetDataSize());
  size_t usrSize = GetCeilAlign(coreNum * FP32_INT32_DTYPE_SIZE, BLOCK_SIZE);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = usrSize + sysWorkspaceSize;

  PrintInfo(context);
  OP_LOGD(context->GetNodeName(), "CrossEntropyLossGradTiling tiling end");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CrossEntropyLossGrad(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

struct CrossEntropyLossGradCompileInfo {};

IMPL_OP_OPTILING(CrossEntropyLossGrad)
  .Tiling(Tiling4CrossEntropyLossGrad)
  .TilingParse<CrossEntropyLossGradCompileInfo>(TilingPrepare4CrossEntropyLossGrad);

}  // namespace optiling