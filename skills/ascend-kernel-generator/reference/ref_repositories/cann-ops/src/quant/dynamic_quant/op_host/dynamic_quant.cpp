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
 * \file dynamic_quant.cpp
 * \brief
 */
#include "dynamic_quant_tiling.h"

#include <algorithm>
#include <map>

#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                        \
  if ((ptr) == nullptr) {                                                                         \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();  \
    std::printf(name, "is nullptr!");                                                             \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                            \
    return ret;                                                                                   \
  }
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                  \
  if ((ptr) == nullptr) {                                                                          \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();   \
    std::printf(name, "is nullptr!");                                                              \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                             \
    return ge::GRAPH_FAILED;                                                                       \
  }
}

namespace optiling {

constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT = 2;
constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT_V2 = 3;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t SMOOTH_INDEX = 1;
constexpr uint32_t GROUP_INDEX = 2;
constexpr uint32_t DST_TYPE_ATTR_INDEX = 0;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t SCALE_INDEX = 1;
constexpr uint32_t OFFSET_INDEX = 2;
constexpr uint32_t ONE = 1;
constexpr uint32_t FP16_DB_SMOOTH_UB_SIZE = 19;
constexpr uint32_t FP16_DB_UB_SIZE = 13;
constexpr uint32_t FP16_SMOOTH_UB_SIZE = 17;
constexpr uint32_t FP16_UB_SIZE = 11;

constexpr uint32_t MOE_UB_SIZE = 17;

constexpr uint32_t BF16_DB_SMOOTH_UB_SIZE = 19;
constexpr uint32_t BF16_DB_UB_SIZE = 13;
constexpr uint32_t BF16_SMOOTH_UB_SIZE = 17;
constexpr uint32_t BF16_UB_SIZE = 11;

constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t COMPARE_INT = 255;
constexpr uint32_t B_ASCEND_HW_NUMCORES = 40;
constexpr uint32_t P_ASCEND_HUW_NUMCORE = 8;
constexpr uint32_t NUM_SIXTEEN = 16;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t ALIGEN_EIGHT = 8;
constexpr uint32_t SEVEN = 7;
constexpr uint32_t INVERSE_FIFTEEN = ~15;
constexpr uint32_t RESERVED_LENGTH = 1 * 1024;
constexpr uint32_t UB_ALIG_NUM = 32;
constexpr uint32_t FLOAT_NUM_ONE_RPT = 128;
constexpr uint32_t MAX_EXPERT_NUM = 1024;
constexpr uint32_t MOE_SMOOTH_NUM = 2;
constexpr int64_t TILING_KEY_BF16 = 0;
constexpr int64_t TILING_KEY_HALF = 1;
constexpr int64_t TILING_KEY_DB_BF16 = 2;
constexpr int64_t TILING_KEY_DB_HALF = 3;
constexpr int64_t TILING_KEY_LARGE_SHAPE = 6;
constexpr int64_t TILING_KEY_MOE = 7;
constexpr int64_t TILING_KEY_MOE_LARGE_SHAPE = 8;
constexpr int64_t EVEN_FACTOR = 2;

static std::map<const ge::DataType, const uint32_t> g_dTypeLen = {{ge::DT_INT32, 4}, {ge::DT_INT64, 8}};

template <uint32_t base, typename T = uint32_t>
T AlignUp(T a) {
  return (a + base - 1) / base * base;
}

class DynamicQuantTiling {
 public:
  DynamicQuantTiling() {
  }
  ~DynamicQuantTiling() {
  }
  DynamicQuantTilingData tilingData;
  ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

 private:
  void SetTilingKey(gert::TilingContext* context, ge::DataType dataType, bool useDb);
  ge::graphStatus CheckInputDtype(gert::TilingContext* context);
  ge::graphStatus CheckOutputDtype(gert::TilingContext* context);
  ge::graphStatus CheckOpInpuShape(gert::TilingContext* context);
  ge::graphStatus CheckOpOutputShape(gert::TilingContext* context);
  ge::graphStatus CheckAttrs(gert::TilingContext* context);
  ge::graphStatus CheckOpShape(gert::TilingContext* context);
  ge::graphStatus CheckOpDim(const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape1Dim,
                             uint32_t shape2Dim);
  ge::graphStatus CheckOpParams(gert::TilingContext* context);
  void ResetLargeTilingParams();
  void SetTilingData(gert::TilingContext* context, ge::DataType xDtype);
  void CalculateMaxUbSizePerRow(gert::TilingContext* context, ge::DataType xDtype);
  ge::graphStatus GetCompileInfo(gert::TilingContext* context);

 private:
  uint32_t vectorCoreNum{0};
  uint64_t ubSize{0};

  uint32_t coreNum;
  uint32_t rowLen;
  uint32_t headCoreNum;
  uint32_t rowPerHeadCore;
  uint32_t rowPerTailCore;
  uint32_t multiRowNumHeadCore;
  uint32_t multiRowNumTailCore;

  uint32_t rowNum;
  uint32_t rowNumPerMinTask;
  uint32_t scaleNumPerMinTask;
  uint32_t rowNumPerTask;
  uint32_t taskNum;
  uint32_t wholeTaskNum;
  uint32_t ubPerRow;
  uint32_t ubPerRowNew;

  uint32_t innerLoopEle = 0;
  uint32_t innerLoopTimes = 0;
  uint32_t innerLoopTail = 0;

  uint32_t groupNum = 0;
  uint32_t groupDtypeSize = 0;
  bool hasSmooth = false;

  int32_t yDtype;
};

void DynamicQuantTiling::SetTilingKey(gert::TilingContext* context, ge::DataType dataType, bool useDb) {
  if (groupNum > 0) {
    if (innerLoopTimes > 0) {
      context->SetTilingKey(TILING_KEY_MOE_LARGE_SHAPE);
    } else {
      context->SetTilingKey(TILING_KEY_MOE);
    }
  } else {
    if (innerLoopTimes > 0) {
      context->SetTilingKey(TILING_KEY_LARGE_SHAPE);
      return;
    }
    if (useDb) {
      if (dataType == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_DB_BF16);
        return;
      }
      context->SetTilingKey(TILING_KEY_DB_HALF);
      return;
    }

    if (dataType == ge::DT_BF16) {
      context->SetTilingKey(TILING_KEY_BF16);
      return;
    }
    context->SetTilingKey(TILING_KEY_HALF);
  }
}

ge::graphStatus DynamicQuantTiling::CheckOpDim(const gert::StorageShape* shape1, const gert::StorageShape* shape2,
                                               uint32_t shape1Dim, uint32_t shape2Dim) {
  if (shape1Dim != shape2Dim) {
    return ge::GRAPH_FAILED;
  }

  for (uint32_t i = 0; i < shape1Dim; i++) {
    if (shape1->GetStorageShape().GetDim(i) != shape2->GetStorageShape().GetDim(i)) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpInpuShape(gert::TilingContext* context) {
  auto xShape = context->GetInputShape(X_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
  size_t xDimNum = xShape->GetStorageShape().GetDimNum();
  if (xDimNum == 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "x shape dimension is less than 2!");
    return ge::GRAPH_FAILED;
  }
  int64_t xDimLast = xShape->GetStorageShape().GetDim(xDimNum - 1);

  if (yDtype == ge::DT_INT4) {
    OP_TILING_CHECK(
        (xDimLast % EVEN_FACTOR),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "if y datatype is int4, the last dim(%ld) of x should be an even number", xDimLast),
        return ge::GRAPH_FAILED);
  }

  auto smoothShape = context->GetOptionalInputShape(SMOOTH_INDEX);
  if (smoothShape != nullptr) {
    auto groupShape = context->GetOptionalInputShape(GROUP_INDEX);
    if (groupShape != nullptr) {
      groupNum = groupShape->GetStorageShape().GetDim(groupShape->GetStorageShape().GetDimNum() - 1);
      OP_TILING_CHECK((groupNum > MAX_EXPERT_NUM),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "moe expert nums exceed maximum, the maximum is 1024."),
                      return ge::GRAPH_FAILED);
    }

    size_t smoothDimNum = smoothShape->GetStorageShape().GetDimNum();
    // 针对moe场景下的校验
    if (groupNum >= 1) {
      OP_TILING_CHECK((smoothDimNum != MOE_SMOOTH_NUM),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "In moe scenario, smooth_scales dim num must be two."),
                      return ge::GRAPH_FAILED);
      int64_t smoothDimFirst = smoothShape->GetStorageShape().GetDim(0);
      if (groupNum != smoothDimFirst) {
        OP_LOGE(context->GetNodeName(),
                "moe expert and smooth_scales first dim is not equal! expert nums is :%d, smooth_scales is:%ld",
                groupNum, smoothDimFirst);
        return ge::GRAPH_FAILED;
      }
    } else {
      if (smoothDimNum != 1) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "smooth_scales dim num not one");
        return ge::GRAPH_FAILED;
      }
    }
    int64_t smoothDimLast = smoothShape->GetStorageShape().GetDim(smoothDimNum - 1);
    if (xDimLast != smoothDimLast) {
      OP_LOGE(context->GetNodeName(), "x and smooth_scales last dim is not equal! x is :%ld, smooth_scales is:%ld",
              xDimLast, smoothDimLast);
      return ge::GRAPH_FAILED;
    }
    hasSmooth = true;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpOutputShape(gert::TilingContext* context) {
  auto xShape = context->GetInputShape(X_INDEX);
  size_t xDimNum = xShape->GetStorageShape().GetDimNum();
  auto yShape = context->GetOutputShape(Y_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
  size_t yDimNum = yShape->GetStorageShape().GetDimNum();
  OP_TILING_CHECK((CheckOpDim(xShape, yShape, xDimNum, yDimNum) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "x and y shape is inconsistency."),
                  return ge::GRAPH_FAILED);

  auto scaleShape = context->GetOutputShape(SCALE_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
  size_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
  OP_TILING_CHECK((CheckOpDim(xShape, scaleShape, xDimNum - 1, scaleDimNum) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      context->GetNodeName(), "scale shape is wrong, must be same as before x exclude the last dim."),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpShape(gert::TilingContext* context) {
  OP_TILING_CHECK((CheckOpInpuShape(context) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input shape check failed!"),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK((CheckOpOutputShape(context) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "output shape check failed!"),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOutputDtype(gert::TilingContext* context) {
  auto yDesc = context->GetOutputDesc(Y_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, yDesc);
  yDtype = yDesc->GetDataType();
  if (yDtype != ge::DataType::DT_INT8 && yDtype != ge::DataType::DT_INT4) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "y dtype is support int8 or int4!");
    return ge::GRAPH_FAILED;
  }

  auto scaleDesc = context->GetOutputDesc(SCALE_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
  auto scaleDtype = scaleDesc->GetDataType();
  if (scaleDtype != ge::DataType::DT_FLOAT) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "scale dtype is only support fp32!");
    return ge::GRAPH_FAILED;
  }

  if (context->GetComputeNodeOutputNum() == OUTPUT_NUM_DYNAMIC_QUANT_V2) {
    auto offsetDesc = context->GetOutputDesc(OFFSET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, offsetDesc);
    auto offsetDtype = scaleDesc->GetDataType();
    if (offsetDtype != ge::DataType::DT_FLOAT) {
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "offset dtype is only support fp32!");
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckInputDtype(gert::TilingContext* context) {
  auto xDesc = context->GetInputDesc(X_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
  auto xDtype = xDesc->GetDataType();
  if (xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "x dtype is only support fp16 or bf16.");
    return ge::GRAPH_FAILED;
  }

  auto smoothDesc = context->GetOptionalInputDesc(SMOOTH_INDEX);
  if (smoothDesc != nullptr) {
    auto smoothDtype = smoothDesc->GetDataType();
    if (smoothDtype != ge::DataType::DT_FLOAT16 && smoothDtype != ge::DataType::DT_BF16) {
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "smooth dtype is only support fp16 or bf16.");
      return ge::GRAPH_FAILED;
    }
    if (xDtype != smoothDtype) {
      OP_LOGE(context->GetNodeName(), "x and smooth_scales dtype is not equal.");
      return ge::GRAPH_FAILED;
    }

    auto groupDesc = context->GetOptionalInputDesc(GROUP_INDEX);
    if (groupDesc != nullptr) {
      auto groupDtype = groupDesc->GetDataType();
      if (groupDtype != ge::DataType::DT_INT32) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "group dtype is only support int32.");
        return ge::GRAPH_FAILED;
      }
      groupDtypeSize = g_dTypeLen[groupDtype];
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckAttrs(gert::TilingContext* context) {
  auto* attrs = context->GetAttrs();
  if (attrs != nullptr) {
    const int32_t* dstTypePtr = attrs->GetAttrPointer<int32_t>(DST_TYPE_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, dstTypePtr, ge::GRAPH_FAILED);
    int32_t dstType = *dstTypePtr;
    if (dstType != yDtype) {
      OP_LOGE(context->GetNodeName(), "dst type:%d not equal output dtype:%d", dstType, yDtype);
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpParams(gert::TilingContext* context) {
  OP_TILING_CHECK((CheckInputDtype(context) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "x or smooth_scales dtype is invalid"),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK((CheckOutputDtype(context) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "op output dtype is invalid"),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK((CheckAttrs(context) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "op attrs is invalid."),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK((CheckOpShape(context) != ge::GRAPH_SUCCESS),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input or output shape is invalid"),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

void DynamicQuantTiling::ResetLargeTilingParams() {
  innerLoopEle = 0;
  innerLoopTimes = 0;
  innerLoopTail = 0;
  groupNum = 0;
  hasSmooth = false;
}

void DynamicQuantTiling::SetTilingData(gert::TilingContext* context, ge::DataType xDtype) {
  uint64_t maxUseUbSize = ubSize - RESERVED_LENGTH;
  uint32_t ubAvail = maxUseUbSize / ubPerRowNew;
  bool useDb = true;
  if (ubAvail == 0) {
    // 这里需要除个单行最大需要的空间
    innerLoopEle = maxUseUbSize / BF16_DB_SMOOTH_UB_SIZE / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT;
    innerLoopTimes = rowLen / innerLoopEle;
    innerLoopTail = rowLen % innerLoopEle;
    ubAvail = ONE;
  } else if (groupNum == 0) {
    if (ubPerRow < maxUseUbSize) {
      useDb = true;
      ubAvail = std::max(ubAvail, ONE);
    } else {
      useDb = false;
      ubAvail = ONE;
    }
  } else {
    useDb = false;
    ubAvail = std::max(ubAvail, ONE);
  }
  SetTilingKey(context, xDtype, useDb);
  tilingData.set_coreNum(coreNum);
  tilingData.set_rowLen(rowLen);
  tilingData.set_headCoreNum(headCoreNum);
  tilingData.set_rowPerHeadCore(rowPerHeadCore);
  tilingData.set_rowPerTailCore(rowPerTailCore);
  tilingData.set_multiRowNumHeadCore(std::min({COMPARE_INT, ubAvail, rowPerHeadCore}));
  tilingData.set_multiRowNumTailCore(std::min({COMPARE_INT, ubAvail, rowPerTailCore}));
  tilingData.set_innerLoopEle(innerLoopEle);
  tilingData.set_innerLoopTimes(innerLoopTimes);
  tilingData.set_innerLoopTail(innerLoopTail);
  tilingData.set_groupNum(groupNum);
  tilingData.set_hasSmooth(hasSmooth ? 1 : 0);
}

/**
 * Calculate the maximum ub space required for each row
 * @param context: ge::TilingContext
 */
void DynamicQuantTiling::CalculateMaxUbSizePerRow(gert::TilingContext* context, ge::DataType xDtype) {
  // Align RowLen
  uint32_t alignedRowLen = AlignUp<NUM_SIXTEEN>(rowLen);
  uint32_t alignedGroupNumLen = 0;
  if (groupNum > 0) {
    alignedGroupNumLen = AlignUp<ALIGEN_EIGHT>(groupNum);
    ubPerRowNew = alignedGroupNumLen * groupDtypeSize + MOE_UB_SIZE * alignedRowLen;
    tilingData.set_alignGroupNum(alignedGroupNumLen);
  } else {
    if (xDtype == ge::DT_BF16) {
      if (hasSmooth) {
        ubPerRow = BF16_DB_SMOOTH_UB_SIZE * alignedRowLen;
        ubPerRowNew = BF16_SMOOTH_UB_SIZE * alignedRowLen;
      } else {
        ubPerRow = BF16_DB_UB_SIZE * alignedRowLen;
        ubPerRowNew = BF16_UB_SIZE * alignedRowLen;
      }
    } else {
      if (hasSmooth) {
        ubPerRow = FP16_DB_SMOOTH_UB_SIZE * alignedRowLen;
        ubPerRowNew = FP16_SMOOTH_UB_SIZE * alignedRowLen;
      } else {
        ubPerRow = FP16_DB_UB_SIZE * alignedRowLen;
        ubPerRowNew = FP16_UB_SIZE * alignedRowLen;
      }
    }
  }
}

ge::graphStatus DynamicQuantTiling::GetCompileInfo(gert::TilingContext* context) {
  auto compileInfo = reinterpret_cast<const DynamicQuantCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, compileInfo, ge::GRAPH_FAILED);
  vectorCoreNum = compileInfo->vectorCoreNum;
  ubSize = compileInfo->ubSize;
  OP_TILING_CHECK((vectorCoreNum <= 0 || ubSize <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      context->GetNodeName(), "RunFusionKernelTiling GetCompileInfo Failed, coreNum:%d, ubSize:%ld.",
                      vectorCoreNum, ubSize),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::RunFusionKernelTiling(gert::TilingContext* context) {
  ResetLargeTilingParams();
  OP_TILING_CHECK(
      (GetCompileInfo(context) != ge::GRAPH_SUCCESS),
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "RunFusionKernelTiling GetCompileInfo failed."),
      return ge::GRAPH_FAILED);
  OP_TILING_CHECK(
      (CheckOpParams(context) != ge::GRAPH_SUCCESS),
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "RunFusionKernelTiling CheckOpParams failed."),
      return ge::GRAPH_FAILED);
  const gert::StorageShape* xShape = context->GetInputShape(X_INDEX);

  rowLen = xShape->GetStorageShape().GetDim(xShape->GetStorageShape().GetDimNum() - 1);
  size_t dimNum = xShape->GetStorageShape().GetDimNum() - 1;
  uint64_t tempHeadCoreNum = 1;
  for (size_t i = 0; i < dimNum; i++) {
    tempHeadCoreNum *= xShape->GetStorageShape().GetDim(i);
  }
  rowNum = tempHeadCoreNum;
  // For 910B
  rowNumPerMinTask = 1;
  scaleNumPerMinTask = 1;
  // For 310P
  if (vectorCoreNum < B_ASCEND_HW_NUMCORES) {
    rowNumPerMinTask = (UB_ALIG_NUM + rowLen - 1) / rowLen;
    scaleNumPerMinTask = P_ASCEND_HUW_NUMCORE;
  }
  rowNumPerTask = std::max(rowNumPerMinTask, scaleNumPerMinTask);
  wholeTaskNum = rowNum / rowNumPerTask;

  coreNum = std::max(std::min(vectorCoreNum, wholeTaskNum), ONE);
  headCoreNum = rowNum % coreNum;

  rowPerHeadCore = (rowNum + coreNum - 1) / coreNum;
  rowPerTailCore = rowNum / coreNum;

  auto x = context->GetInputDesc(X_INDEX);
  CalculateMaxUbSizePerRow(context, x->GetDataType());
  SetTilingData(context, x->GetDataType());

  size_t* workSpaces = context->GetWorkspaceSizes(1);
  workSpaces[0] = SYS_WORKSPACE_SIZE;
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  context->SetBlockDim(coreNum);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTilingContext(gert::TilingContext* context) {
  if (context == nullptr || (context->GetRawTilingData() == nullptr) ||
      (context->GetRawTilingData()->GetData() == nullptr) || (context->GetWorkspaceSizes(1) == nullptr) ||
      context->GetWorkspaceNum() <= 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForDynamicQuant(gert::TilingContext* context) {
  if (CheckTilingContext(context) == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }
  static thread_local DynamicQuantTiling tiling;
  auto ret = tiling.RunFusionKernelTiling(context);
  return ret;
}

IMPL_OP_OPTILING(DynamicQuant)
    .Tiling(TilingForDynamicQuant);
IMPL_OP_OPTILING(DynamicQuantV2)
    .Tiling(TilingForDynamicQuant);
}  // namespace optiling
