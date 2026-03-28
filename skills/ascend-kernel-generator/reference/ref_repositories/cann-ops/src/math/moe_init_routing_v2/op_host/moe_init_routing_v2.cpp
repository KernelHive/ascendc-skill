/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */ 
 
#include "moe_init_routing_v2_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>
    
using namespace std;
namespace optiling {
const static int64_t TILING_KEY_DROPLESS_SORT_ONE_CORE = 10001;
const static int64_t TILING_KEY_DROPLESS_SORT_MULTI_CORE = 10002;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE = 10011;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE = 10012;
const static int64_t TILING_KEY_HIGH_PERFORMANCE = 20000;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static size_t DIM_THREE = 3;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t INDEX_INPUT_X = 0;
const static int64_t INDEX_INPUT_EXPERT_IDX = 1;
const static int64_t ATTR_ACTIVE_ROWS = 0;
const static int64_t ATTR_EXPERT_CAPACITY = 1;
const static int64_t ATTR_EXPERT_NUM = 2;
const static int64_t ATTR_DROP_PAD_MODE = 3;
const static int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
const static int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
const static int64_t ATTR_START_EXPERTID = 6;
const static int64_t ATTR_END_EXPERTID = 7;
const static int64_t ATTR_DEVICE_ID = 8;
const static int64_t OUTOUT_EXPANDED_X = 0;
const static int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
const static int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
const static int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t EXPERT_TOKENS_COUNT = 2;

#define CHECK_FAIL(context, cond, ...)                \
  do {                                                \
    if (cond) {                                       \
      printf(context->GetNodeName(), ##__VA_ARGS__); \
      return ge::GRAPH_FAILED;                        \
    }                                                 \
  } while (0)

#define CHECK_NULL(context, ptr, ...)                                                              \
  do {                                                                                             \
    if ((ptr) == nullptr) {                                                                        \
      const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
      printf(name, "%s is nullptr!", ##__VA_ARGS__);                               \
      printf("EZ9999", "op[%s], %s is nullptr!", name, ##__VA_ARGS__);                  \
      return ge::GRAPH_FAILED;                                                                     \
    }                                                                                              \
  } while (0)

inline static int64_t CeilLog4(int64_t x) {
  return std::ceil(std::log(x) / std::log(NUM_FOUR));
}

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y) {
  if (y == 0) {
    return 0;
  }
  return x <= y ? x : x % y;
}

template <typename T>
typename std::enable_if <std::is_signed<T>::value, T>::type CeilDiv(T x, T y) {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
  }

  return x;
}

void MoeInitRoutingV2TilingBase::Reset() {
  opName = nullptr;
  return;
}

ge::graphStatus MoeInitRoutingV2TilingBase::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();

  if (platformInfo == nullptr)
  {
    return ge::GRAPH_FAILED;
  }
  
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  aivNum = ascendcPlatform.GetCoreNumAiv();
  aicoreParams_.blockDim = aivNum;
  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  aicoreParams_.ubSize = ubSizePlatForm;
  moeInitRoutingTilingData.set_coreNum(aivNum);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::CheckTokenCount(int64_t num, const char* tag) {
  auto expertTokensShapePtr = context_->GetOutputShape(num);
  CHECK_NULL(context_, expertTokensShapePtr, tag);

  auto expertTokensDesc = context_->GetOutputDesc(num);
  CHECK_NULL(context_, expertTokensDesc, tag);
  auto dt = expertTokensDesc->GetDataType();
  CHECK_FAIL(context_, dt != ge::DT_INT32, "The data type of %s should be int32.", tag);

  const gert::Shape expertTokensShape = expertTokensShapePtr->GetStorageShape();
  size_t expertTokensDimNum = expertTokensShape.GetDimNum();
  CHECK_FAIL(context_, expertTokensDimNum != DIM_ONE, "The dim number of %s should be 1.", tag);
  CHECK_FAIL(context_, expertTokensShape.GetDim(0) != expertNum, "The first dim of %s should be %ld.", tag, expertNum);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::CheckOutShape() {
  // 获取输出shape
  auto expandedXShapePtr = context_->GetOutputShape(OUTOUT_EXPANDED_X);
  CHECK_NULL(context_, expandedXShapePtr, "expandedX");
  const gert::Shape expandedXShape = expandedXShapePtr->GetStorageShape();

  auto expandedRowIdxShapePtr = context_->GetOutputShape(OUTOUT_EXPANDED_ROW_IDX);
  CHECK_NULL(context_, expandedRowIdxShapePtr, "expandedRowIdx");
  const gert::Shape expandedRowIdxShape = expandedRowIdxShapePtr->GetStorageShape();

  size_t expandedXDimNum = expandedXShape.GetDimNum();
  if (dropPadMode > 0) {
    CHECK_FAIL(context_, expandedXDimNum != DIM_THREE, "The dim number of expandedX should be 3.");
    CHECK_FAIL(context_, expandedXShape.GetDim(0) != expertNum, "The first dim of expandedX should be %ld.", expertNum);
    CHECK_FAIL(context_, expandedXShape.GetDim(1) != expertCapacity, "The second dim of expandedX should be %ld.",
               expertCapacity);
    CHECK_FAIL(context_, expandedXShape.GetDim(NUM_TWO) != moeInitRoutingTilingData.get_cols(),
               "The third dim of expandedX should be %ld.", moeInitRoutingTilingData.get_cols());
  } else {
    CHECK_FAIL(context_, expandedXDimNum != DIM_TWO, "The dim number of expandedX should be 2.");
    int64_t firstDim = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();
    firstDim = activateNum == 0 ? firstDim : std::min(firstDim, activateNum);
    CHECK_FAIL(context_, expandedXShape.GetDim(0) != firstDim, "The first dim of expandedX should be %ld.", firstDim);
    CHECK_FAIL(context_, expandedXShape.GetDim(1) != moeInitRoutingTilingData.get_cols(),
               "The second dim of expandedX should be %ld.", moeInitRoutingTilingData.get_cols());
  }

  size_t expandedRowIdxDimNum = expandedRowIdxShape.GetDimNum();
  CHECK_FAIL(context_, expandedRowIdxDimNum != DIM_ONE, "The dim number of expandedRowIdx should be 1.");
  CHECK_FAIL(context_, expandedRowIdxShape.GetDim(0) != totalLength, "The first dim of expandedRowIdx should be %ld.",
             totalLength);

  if (dropPadMode == 0 && expertTokensCountOrCumsumFlag != 0) {
    if (CheckTokenCount(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM, "expertTokensCountOrCumsum") == ge::GRAPH_FAILED) {
      return ge::GRAPH_FAILED;
    }
  }

  if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
    if (CheckTokenCount(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY, "expertTokensBeforeCapacity") == ge::GRAPH_FAILED) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::GetShapeAttrsInfo() {
  opName = context_->GetNodeName();


  // 获取输入shape
  auto xShapePtr = context_->GetInputShape(INDEX_INPUT_X);
  CHECK_NULL(context_, xShapePtr, "x");
  const gert::Shape xShape = xShapePtr->GetStorageShape();
  auto expertIdxShapePtr = context_->GetInputShape(INDEX_INPUT_EXPERT_IDX);
  CHECK_NULL(context_, expertIdxShapePtr, "expertIdx");
  const gert::Shape expertIdxShape = expertIdxShapePtr->GetStorageShape();

  auto attrs = context_->GetAttrs();
  CHECK_NULL(context_, attrs, "Attrs");
  const int64_t* activateNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_ROWS);
  if (activateNumPtr != nullptr) {
    activateNum = *activateNumPtr;
  }
  const int64_t* expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY);
  if (expertCapacityPtr != nullptr) {
    expertCapacity = *expertCapacityPtr;
  }
  const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
  if (expertNumPtr != nullptr) {
    expertNum = *expertNumPtr;
  }
  const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
  if (dropPadModePtr != nullptr) {
    dropPadMode = *dropPadModePtr;
  }
  const int64_t* expertTokensCountOrCumsumFlagPtr =
      attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG);
  if (expertTokensCountOrCumsumFlagPtr != nullptr) {
    expertTokensCountOrCumsumFlag = *expertTokensCountOrCumsumFlagPtr;
  }
  const bool* expertTokensBeforeCapacityFlagPtr = attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG);
  if (expertTokensBeforeCapacityFlagPtr != nullptr) {
    expertTokensBeforeCapacityFlag = *expertTokensBeforeCapacityFlagPtr;
  }

  const int64_t* start_expertIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_START_EXPERTID);
  if (start_expertIdPtr != nullptr) {
    start_expertId = *start_expertIdPtr;
  }

  const int64_t* end_expertIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_END_EXPERTID);
  if (end_expertIdPtr != nullptr) {
    end_expertId = *end_expertIdPtr;
  }

    const int64_t* device_idPtr = attrs->GetAttrPointer<int64_t>(ATTR_DEVICE_ID);
  if (device_idPtr != nullptr) {
    device_id = *device_idPtr;
  }

  // 参数校验
  size_t xDimNnum = xShape.GetDimNum();
  size_t expertIdxDimNum = expertIdxShape.GetDimNum();
  CHECK_FAIL(context_, xDimNnum != DIM_TWO || expertIdxDimNum != DIM_TWO,
             "The dim number of x and expertIdx should be 2.");
  CHECK_FAIL(context_, xShape.GetDim(0) != expertIdxShape.GetDim(0),
             "The first dim of x and expertIdx should be equal.");
  CHECK_FAIL(context_, expertIdxShape.GetDim(1) < 0, "The second dim of expertIdx cannot be less than 0.");
  CHECK_FAIL(context_, activateNum < 0, "The activeNum cannot be less than 0.");
  CHECK_FAIL(context_, expertCapacity < 0, "The expertCapacity cannot be less than 0.");
  CHECK_FAIL(context_, expertNum < 0, "The expertNum cannot be less than 0.");
  CHECK_FAIL(context_, dropPadMode < 0 || dropPadMode > 1, "The dropPadMode should be 0 or 1.");
  CHECK_FAIL(context_, dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1),
             "The expertCapacity and expertNum should be greater than 0 when dropPadMode is 1");
  CHECK_FAIL(context_, expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > EXPERT_TOKENS_COUNT,
             "The expertTokensCountOrCumsumFlag should be 0, 1 or 2.");
  CHECK_FAIL(context_, expertTokensCountOrCumsumFlag > 0 && expertNum <= 0,
             "The expertNum should be greater than 0 when expertTokensCountOrCumsumFlag is greater than 0");
  CHECK_FAIL(context_, dropPadMode > 0 && expertCapacity > xShape.GetDim(0), "The first dim of x cannot be less than expertCapacity");
  if (dropPadMode == 1) {
    // droppad场景下不输出expertTokensCountOrCumsum
    expertTokensCountOrCumsumFlag = 0;
  } else {
    // dropless场景下不输出expertTokensBeforeCapacity
    expertTokensBeforeCapacityFlag = false;
  }
  moeInitRoutingTilingData.set_cols(xShape.GetDim(1));
  moeInitRoutingTilingData.set_n(expertIdxShape.GetDim(0));
  moeInitRoutingTilingData.set_k(expertIdxShape.GetDim(1));
  moeInitRoutingTilingData.set_expertCapacity(expertCapacity);
  moeInitRoutingTilingData.set_expertNum(expertNum);

  moeInitRoutingTilingData.set_start_expertId(start_expertId);
  moeInitRoutingTilingData.set_end_expertId(end_expertId);
  moeInitRoutingTilingData.set_device_id(device_id);

  moeInitRoutingTilingData.set_dropPadMode(dropPadMode);
  moeInitRoutingTilingData.set_expertTokensCountOrCumsumFlag(expertTokensCountOrCumsumFlag);
  moeInitRoutingTilingData.set_expertTokensBeforeCapacityFlag(expertTokensBeforeCapacityFlag);
  totalLength = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();

  auto ret = CheckOutShape();
  inuptXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
  return ret;
}

void MoeInitRoutingV2TilingBase::ShowTilingData() {
//   OP_LOGI(opName,
//           "moeInitRoutingTilingData is coreNum:%ld, n:%ld, cols:%ld, k:%ld, expertCapacity:%ld, expertNum:%ld, "
//           "dropPadMode:%ld, expertTokensCountOrCumsumFlag:%ld, expertTokensBeforeCapacityFlag:%ld",
//           moeInitRoutingTilingData.get_coreNum(), moeInitRoutingTilingData.get_n(), moeInitRoutingTilingData.get_cols(),
//           moeInitRoutingTilingData.get_k(), moeInitRoutingTilingData.get_expertCapacity(),
//           moeInitRoutingTilingData.get_expertNum(), moeInitRoutingTilingData.get_dropPadMode(),
//           moeInitRoutingTilingData.get_expertTokensCountOrCumsumFlag(),
//           moeInitRoutingTilingData.get_expertTokensBeforeCapacityFlag());
//   OP_LOGI(opName,
//           "MoeV2VBSComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, "
//           "perCorePerLoopElements:%ld, "
//           "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
//           "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld",
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_needCoreNum(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreElements(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLoops(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_perCorePerLoopElements(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLastLoopElements(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreElements(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLoops(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCorePerLoopElements(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLastLoopElements(),
//           moeInitRoutingTilingData.vbsComputeParamsOp.get_oneLoopMaxElements());
//   OP_LOGI(opName, "VMSMiddleComputeTilingData is needCoreNum:%ld",
//           moeInitRoutingTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
//   OP_LOGI(opName, "SortOutComputeTilingData is oneLoopMaxElements:%ld",
//           moeInitRoutingTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
//   OP_LOGI(opName,
//           "SrcToDstComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
//           "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_needCoreNum(),
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_activateRows(),
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreRows(),
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCorePerLoopRows(),
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreLastLoopRows(),
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreRows(),
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCorePerLoopRows(),
//           moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreLastLoopRows());
//   OP_LOGI(opName,
//           "SrcToDstComputeCapacityTilingData is needCoreNum:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
//           "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
//           moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_needCoreNum(),
//           moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCoreRows(),
//           moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCorePerLoopRows(),
//           moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCoreLastLoopRows(),
//           moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCoreRows(),
//           moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCorePerLoopRows(),
//           moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCoreLastLoopRows());
//   OP_LOGI(opName,
//           "GatherOutComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
//           "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_needCoreNum(),
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_activateRows(),
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreRows(),
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCorePerLoopRows(),
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreLastLoopRows(),
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreRows(),
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCorePerLoopRows(),
//           moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreLastLoopRows());
}

ge::graphStatus MoeInitRoutingV2TilingBase::DoOpTiling() {
  // NUM_TWO sort value and indices
  // NUM_FOUR sort need space
  // SORT32_ALIGN_ELEMENT 32Bytes aligned
  sortLoopMaxElement =
      (aicoreParams_.ubSize) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
  isFullLoad = IsFullLoad();
  Tiling4VBSCompute();
  Tiling4VMSMiddleCompute();
  Tiling4SortOutCompute();
  Tiling4SrcToDstCompute();
  Tiling4SrcToDstCapacityCompute();
  Tiling4GatherOutCompute();
  ShowTilingData();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingV2TilingBase::GetTilingKey() const {
  if (isFullLoad) {
    return TILING_KEY_HIGH_PERFORMANCE;
  }
  if (dropPadMode == 0) {
    if (totalLength <= sortLoopMaxElement) {  // 排序只用到一个核排序
      return TILING_KEY_DROPLESS_SORT_ONE_CORE;
    } else {
      return TILING_KEY_DROPLESS_SORT_MULTI_CORE;
    }
  } else {
    if (totalLength <= sortLoopMaxElement) {
      return TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE;
    } else {
      return TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE;
    }
  }
  return tilingKey_;
}

ge::graphStatus MoeInitRoutingV2TilingBase::GetWorkspaceSize() {
  // 计算workspace大小
  size_t sortWorkspaceSize = totalLength * sizeof(float) * NUM_TWO * NUM_THREE;  // 排序需要的空间
  size_t scatterWorkspaceSize = totalLength * sizeof(int32_t) * NUM_TWO;
  size_t expertTokenFlagSize = aivNum * 2 * sizeof(int32_t);
  workspaceSize_ = sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::PostTiling() {
  context_->SetBlockDim(aivNum);
  size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = workspaceSize_;
  moeInitRoutingTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                        context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(moeInitRoutingTilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

void MoeInitRoutingV2TilingBase::Tiling4VBSOneCoreCompute(MoeV2VBSComputeTilingData* tilingData) {
  tilingData->set_needCoreNum(1);
  tilingData->set_perCoreElements(totalLength);
  tilingData->set_perCoreLoops(1);
  tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
  tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
  tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
  tilingData->set_lastCoreLoops(1);
  tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
  tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeInitRoutingV2TilingBase::Tiling4VBSMultiCoreCompute(MoeV2VBSComputeTilingData* tilingData) {
  int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);  // 向上取整
  needCoreNum = std::pow(4, CeilLog4(needCoreNum));                // 用到多核时，核数最多是4^x
  needCoreNum = std::min(needCoreNum, aivNum);                     // 不能超过物理核数
  if (needCoreNum == 0) {
    return;
  }
  int64_t perCoreElements = totalLength / needCoreNum;  // 每个核处理的元素数
  int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
  int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
  int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
  if (lastCoreElement > alineCeilPerCoreElements) {
    perCoreElements = alineCeilPerCoreElements;
    needCoreNum = CeilDiv(totalLength, perCoreElements);
  } else {
    perCoreElements = alineFloorPerCoreElements;
  }

  tilingData->set_needCoreNum(needCoreNum);
  do {
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_perCoreLoops(CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement));  // 每个核处理的loop数
    tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                            (tilingData->get_perCoreLoops() - 1) *
                                                tilingData->get_perCorePerLoopElements());

    tilingData->set_lastCoreElements(totalLength -
                                     (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
    int64_t lastCorePerLoopElements =
        CeilDiv(CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()), SORT32_ALIGN_ELEMENT) *
        SORT32_ALIGN_ELEMENT;
    tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                             (tilingData->get_lastCoreLoops() - 1) *
                                                 tilingData->get_lastCorePerLoopElements());
    perCoreElements -= SORT32_ALIGN_ELEMENT;
  } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
//   OP_TILING_CHECK(
//       tilingData->get_lastCoreLastLoopElements() <= 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "vbs tiling failed"), ;);
    if (tilingData->get_lastCoreLastLoopElements() <= 0)
    {
       ;
    }
    
}

void MoeInitRoutingV2TilingBase::Tiling4VBSCompute() {
  auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
  tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
  if (totalLength <= sortLoopMaxElement) {  // 只用到一个核
    Tiling4VBSOneCoreCompute(tilingData);
    return;
  }
  Tiling4VBSMultiCoreCompute(tilingData);
}

void MoeInitRoutingV2TilingBase::Tiling4VMSMiddleCompute() {
  auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
  auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
  if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) {  // 队列数小于一次vms则没有中间归并
    tilingData->set_needCoreNum(0);                               // 需要的核数
    return;
  }
  int64_t needCoreNum = CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
  tilingData->set_needCoreNum(needCoreNum);  // 需要的核数
}

void MoeInitRoutingV2TilingBase::Tiling4SortOutCompute() {
  auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
  tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCompute() {
  auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

  int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                           (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0) {
    tilingData->set_needCoreNum(0);
    return;
  }
  int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
  tilingData->set_needCoreNum(needCoreNum);
  int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);

  tilingData->set_perCoreRows(perCoreRows);

  if (perLoopMaxRows >= tilingData->get_perCoreRows()) {  // 一个loop结束
    tilingData->set_perCorePerLoopRows(tilingData->get_perCoreRows());
    tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows());
  } else {
    tilingData->set_perCorePerLoopRows(perLoopMaxRows);
    tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows() -
                                        (CeilDiv(tilingData->get_perCoreRows(), perLoopMaxRows) - 1) * perLoopMaxRows);
  }

  tilingData->set_lastCoreRows(lastCoreNum);
  if (perLoopMaxRows >= tilingData->get_lastCoreRows()) {
    tilingData->set_lastCorePerLoopRows(tilingData->get_lastCoreRows());
    tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows());
  } else {
    tilingData->set_lastCorePerLoopRows(perLoopMaxRows);
    tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows() -
                                         (CeilDiv(tilingData->get_lastCoreRows(), perLoopMaxRows) - 1) *
                                             perLoopMaxRows);
  }
}

void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute() {
  auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;

  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0) {
    tilingData->set_needCoreNum(0);
    return;
  }
  int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
  tilingData->set_needCoreNum(needCoreNum);
  int64_t cols = moeInitRoutingTilingData.get_cols();
  tilingData->set_perCoreRows(perCoreRows);
  int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
  tilingData->set_lastCoreRows(lastCoreRows);

  int64_t rowSize =
      (perCoreRows * sizeof(int32_t) * 2 + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
  int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

  if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
    tilingData->set_perCorePerLoopRows(perCoreRows);
    tilingData->set_perCoreLastLoopRows(perCoreRows);
    tilingData->set_lastCorePerLoopRows(lastCoreRows);
    tilingData->set_lastCoreLastLoopRows(lastCoreRows);
    tilingData->set_perCoreLoops(1);
    tilingData->set_lastCoreLoops(1);
    tilingData->set_perLoopCols(cols);
    tilingData->set_lastLoopCols(cols);
    tilingData->set_colLoops(1);
  } else {
    int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
    int64_t baseMaxColsSize = (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    if (cols < MAX_COLS_ONE_LOOP) {
      basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                           static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    } else if (perCoreRows < basePerLoopMaxRows) {
      baseMaxCols =
          (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize_ / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    }
    tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
    tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
    tilingData->set_colLoops((cols + baseMaxCols - 1) / baseMaxCols);

    tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLoops((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

    tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLoops((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
  }
}

void MoeInitRoutingV2TilingBase::Tiling4GatherOutCompute() {
  auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
  tilingData->set_activateRows(totalLength);
  if (dropPadMode == 0 && activateNum > 0) {
    tilingData->set_activateRows(std::min(activateNum, totalLength));
  }
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0 || moeInitRoutingTilingData.get_cols() <= 0) {
    tilingData->set_needCoreNum(0);
    return;
  }
  tilingData->set_needCoreNum(CeilDiv(totalLength, perCoreRows));
  int64_t cols = moeInitRoutingTilingData.get_cols();
  tilingData->set_perCoreRows(perCoreRows);
  int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
  tilingData->set_lastCoreRows(lastCoreRows);

  int64_t rowSize = (perCoreRows * sizeof(int32_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
  int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

  if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO) {
    tilingData->set_perCorePerLoopRows(perCoreRows);
    tilingData->set_perCoreLastLoopRows(perCoreRows);
    tilingData->set_lastCorePerLoopRows(lastCoreRows);
    tilingData->set_lastCoreLastLoopRows(lastCoreRows);
    tilingData->set_perCoreLoops(1);
    tilingData->set_lastCoreLoops(1);
    tilingData->set_perLoopCols(cols);
    tilingData->set_lastLoopCols(cols);
    tilingData->set_colLoops(1);
  } else {
    int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
    int64_t baseMaxColsSize = (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - baseMaxColsSize) /
                                 static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    if (cols < MAX_COLS_ONE_LOOP) {
      basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - colSize) /
                           static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    } else if (perCoreRows < basePerLoopMaxRows) {
      baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - rowSize) / inuptXDtypeSize_ /
                    ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    }
    tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
    tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
    tilingData->set_colLoops((cols + baseMaxCols - 1) / baseMaxCols);

    tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLoops((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

    tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLoops((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
  }
}

bool MoeInitRoutingV2TilingBase::IsFullLoad() {
  if (totalLength > sortLoopMaxElement || moeInitRoutingTilingData.get_cols() > MAX_COLS_ONE_LOOP ||
      this->dropPadMode == 1) {
    return false;
  }
  int64_t sortSpace =
      CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * ONE_CORE_SORT_BUFFER;
  int64_t otherSpace =
      CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * NUM_THREE;
  int64_t expertSpace = CeilDiv(this->expertNum * int64_t(sizeof(int32_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
  int64_t perCoreXRows = moeInitRoutingTilingData.get_n() / aivNum;
  int64_t remainder = moeInitRoutingTilingData.get_n() % aivNum;
  // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
  perCoreXRows = remainder <= 1 ? perCoreXRows + 1 : perCoreXRows + NUM_TWO;
  int64_t gatherSpace =
      CeilDiv(moeInitRoutingTilingData.get_cols() * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * perCoreXRows;
  int64_t remainUbAfterSort = aicoreParams_.ubSize - sortSpace - otherSpace - expertSpace - gatherSpace;
  return remainUbAfterSort > 0;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  MoeInitRoutingV2TilingBase tiling(context);
  return tiling.DoTiling();
}
}


namespace ge {

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    return ge::GRAPH_FAILED;                                                                     \
}

static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr int64_t OTHER_SHAPE = -1;
static constexpr int64_t INDEX_INPUT_X = 0;
static constexpr int64_t INDEX_INPUT_EXPERT_IDX = 1;
static constexpr int64_t OUTOUT_EXPANDED_X = 0;
static constexpr int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
static constexpr int64_t ATTR_ACTIVE_ROWS = 0;
static constexpr int64_t ATTR_EXPERT_CAPACITY = 1;
static constexpr int64_t ATTR_EXPERT_NUM = 2;
static constexpr int64_t ATTR_DROP_PAD_MODE = 3;
static constexpr int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
static constexpr int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
static constexpr int64_t EXPERT_TOKENS_COUNT = 2;


static bool isSameDim(int64_t dim1, int64_t dim2) {
  if (dim1 == OTHER_SHAPE || dim2 == OTHER_SHAPE) {
    return true;
  }
  return dim1 == dim2;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext* context, const gert::Shape* xShape,
                                       const gert::Shape* expertIdxShape) {
  int64_t x_n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
  int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
  if (x_n < OTHER_SHAPE || cols < OTHER_SHAPE) {
    // OP_LOGE(context->GetNodeName(), "Invalid x shape, shape is %s.", ge::Shape2String(*xShape).c_str());
    return ge::GRAPH_FAILED;
  }

  int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(0);
  int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
  if (expert_idx_n < OTHER_SHAPE || expert_idx_k < OTHER_SHAPE) {
    // OP_LOGE(context->GetNodeName(), "Invalid expertIdx shape, shape is %s.", ge::Shape2String(*expertIdxShape).c_str());
    return ge::GRAPH_FAILED;
  }

  if (!isSameDim(x_n, expert_idx_n)) {
    // OP_LOGE(context->GetNodeName(), "The first dim of x(%ld) and expertIdx(%ld) should be equal.", x_n, expert_idx_n);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParm(gert::InferShapeContext* context, const gert::Shape* xShape,
                                 const gert::Shape* expertIdxShape, const int64_t activeNum,
                                 const int64_t expertCapacity, const int64_t expertNum, const int64_t dropPadMode,
                                 const int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag) {
  if (xShape->GetDimNum() == 1U) {
    if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
    //   OP_LOGE(context->GetNodeName(), "The dynamic dim of x should be -2, current shape is %s.",
    //           ge::Shape2String(*xShape).c_str());
      return ge::GRAPH_FAILED;
    }
  } else if (xShape->GetDimNum() != DIM_TWO) {
    // OP_LOGE(context->GetNodeName(), "The dim of x should be 2 or dynamic, current shape is %s.",
    //         ge::Shape2String(*xShape).c_str());
    return ge::GRAPH_FAILED;
  }

  if (expertIdxShape->GetDimNum() == 1U) {
    if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
    //   OP_LOGE(context->GetNodeName(), "The dynamic dim of expertIdx should be -2, current shape is %s.",
    //           ge::Shape2String(*expertIdxShape).c_str());
      return ge::GRAPH_FAILED;
    }
  } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
    // OP_LOGE(context->GetNodeName(), "The dim of expertIdx should be 2 or dynamic, current shape is %s.",
    //         ge::Shape2String(*expertIdxShape).c_str());
    return ge::GRAPH_FAILED;
  }
  if (activeNum < 0) {
    // OP_LOGE(context->GetNodeName(), "activeNum cannot be less than 0.");
    return ge::GRAPH_FAILED;
  }
  if (expertCapacity < 0) {
    // OP_LOGE(context->GetNodeName(), "The expertCapacity cannot be less than 0.");
    return ge::GRAPH_FAILED;
  }
  if (expertNum < 0) {
    // OP_LOGE(context->GetNodeName(), "The expertNum should cannot be less than 0.");
    return ge::GRAPH_FAILED;
  }
  if (dropPadMode < 0 || dropPadMode > 1) {
    // OP_LOGE(context->GetNodeName(), "The dropPadMode should be 0 or 1.");
    return ge::GRAPH_FAILED;
  }
  if (dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1)) {
    // OP_LOGE(context->GetNodeName(), "The expertCapacity and expertNum should be greater 0 when dropPadMode is 1");
    return ge::GRAPH_FAILED;
  }
  if (expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > EXPERT_TOKENS_COUNT) {
    // OP_LOGE(context->GetNodeName(), "The expertTokensCountOrCumsumFlag should be 0, 1 or 2.");
    return ge::GRAPH_FAILED;
  }
  if (expertTokensCountOrCumsumFlag > 0 && expertNum <= 0) {
    // OP_LOGE(context->GetNodeName(),
            // "The expertNum should be greater than 0 when expertTokensCountOrCumsumFlag is greater than 0");
    return ge::GRAPH_FAILED;
  }
  if (dropPadMode > 0 && xShape->GetDim(0) > 0 && expertCapacity > xShape->GetDim(0)) {
    // OP_LOGE(context->GetNodeName(),
            // "The first dim of x cannot be less than expertCapacity when dropPadMode is 1");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static void InferOutputShape(const gert::Shape* xShape, const gert::Shape* expertIdxShape, gert::Shape* expandedXShape,
                             gert::Shape* expandedRowIdx, gert::Shape* expertTokensBeforeCapacityShape,
                             gert::Shape* expertTokensCountOrCumsumShape, const int64_t activeNum,
                             const int64_t expertNum, const int64_t expertCapacity, const int64_t dropPadMode,
                             bool expertTokensBeforeCapacityFlag, const int64_t expertTokensCountOrCumsumFlag) {
  int64_t n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
  int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
  int64_t k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
  int64_t outActiveNum = OTHER_SHAPE;
  int64_t expandedRowIdxNum = OTHER_SHAPE;

  if (n > 0 && k > 0) {
    expandedRowIdxNum = n * k;
    outActiveNum = activeNum > 0 ? std::min(n * k, activeNum) : n * k;
  }

  if (dropPadMode > 0) {
    expandedXShape->SetDimNum(DIM_THREE);
    expandedXShape->SetDim(0U, expertNum);
    expandedXShape->SetDim(1U, expertCapacity);
    expandedXShape->SetDim(2U, cols < 0 ? OTHER_SHAPE : cols);
  } else {
    expandedXShape->SetDimNum(DIM_TWO);
    expandedXShape->SetDim(0U, outActiveNum);
    expandedXShape->SetDim(1U, cols < 0 ? OTHER_SHAPE : cols);
  }

  expandedRowIdx->SetDimNum(DIM_ONE);
  expandedRowIdx->SetDim(0U, expandedRowIdxNum);

  if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
    expertTokensBeforeCapacityShape->SetDimNum(DIM_ONE);
    expertTokensBeforeCapacityShape->SetDim(0U, expertNum);
  }

  if (dropPadMode == 0 && expertTokensCountOrCumsumFlag > 0) {
    expertTokensCountOrCumsumShape->SetDimNum(DIM_ONE);
    expertTokensCountOrCumsumShape->SetDim(0U, expertNum);
  }
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
//   OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRountingV2Infershape.");
  // 获取attr
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_ROWS);
  const int64_t activeNum = (activeNumPtr == nullptr) ? 0 : *activeNumPtr;
  const int64_t* expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY);
  const int64_t expertCapacity = (expertCapacityPtr == nullptr) ? 0 : *expertCapacityPtr;
  const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
  const int64_t expertNum = (expertNumPtr == nullptr) ? 0 : *expertNumPtr;
  const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
  const int64_t dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;
  const int64_t* expertTokensCountOrCumsumFlagPtr =
      attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG);
  const int64_t expertTokensCountOrCumsumFlag =
      (expertTokensCountOrCumsumFlagPtr == nullptr) ? 0 : *expertTokensCountOrCumsumFlagPtr;
  const bool* expertTokensBeforeCapacityFlagPtr = attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG);
  const bool expertTokensBeforeCapacityFlag =
      (expertTokensBeforeCapacityFlagPtr == nullptr) ? false : *expertTokensBeforeCapacityFlagPtr;

  // 获取输入shape
  const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
  const gert::Shape* expertIdxShape = context->GetInputShape(INDEX_INPUT_EXPERT_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);
  gert::Shape* expandedXShape = context->GetOutputShape(OUTOUT_EXPANDED_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
  gert::Shape* expandedRowIdx = context->GetOutputShape(OUTOUT_EXPANDED_ROW_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdx);
  
  gert::Shape* expertTokensCountOrCumsumShape = context->GetOutputShape(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM);
  if (dropPadMode == 0 && expertTokensCountOrCumsumFlag > 0) {
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokensCountOrCumsumShape);
  }
  gert::Shape* expertTokensBeforeCapacityShape = context->GetOutputShape(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY);
  if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokensBeforeCapacityShape);
  }

  if (CheckInputShape(context, xShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (CheckParm(context, xShape, expertIdxShape, activeNum, expertCapacity, expertNum, dropPadMode,
                expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  InferOutputShape(xShape, expertIdxShape, expandedXShape, expandedRowIdx, expertTokensBeforeCapacityShape, 
                   expertTokensCountOrCumsumShape, activeNum, expertNum, expertCapacity, dropPadMode,
                   expertTokensBeforeCapacityFlag, expertTokensCountOrCumsumFlag);

//   OP_LOGD(context->GetNodeName(), "End to do MoeInitRountingV2Infershape.");
    return GRAPH_SUCCESS;
}
}


namespace ops {
class MoeInitRoutingV2 : public OpDef {
public:
    explicit MoeInitRoutingV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("expert_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("expanded_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("expanded_row_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("expert_tokens_count_or_cumsum")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("expert_tokens_before_capacity")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("active_num").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_capacity").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_num").AttrType(OPTIONAL).Int(0);
        this->Attr("drop_pad_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_tokens_count_or_cumsum_flag").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_tokens_before_capacity_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("start_expertId").AttrType(OPTIONAL).Int(0);
        this->Attr("end_expertId").AttrType(OPTIONAL).Int(0);
        this->Attr("device_id").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(MoeInitRoutingV2);
}
