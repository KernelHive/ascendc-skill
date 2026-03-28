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
 * \file upsample_bicubic2d_aa.cpp
 * \brief
 */
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "upsample_bicubic2d_aa_tiling.h"

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
  if ((ptr) == nullptr) {                         \
    std::printf("nullptr error!");                \
    return ge::GRAPH_FAILED;                      \
  }
}

namespace optiling {

constexpr uint32_t BEST_PERFORMANCE_SIZE_1 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_2 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_3 = 48;
constexpr uint32_t BEST_PERFORMANCE_SIZE_4 = 64;

constexpr uint32_t BEST_PERFORMANCE_SCALE_1 = 50;
constexpr uint32_t BEST_PERFORMANCE_SCALE_2 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_3 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_4 = 5;
constexpr float BICUBIC_SUPPORT_SIZE = 2.0;
constexpr uint32_t BUFFER_LEN = 2;
constexpr uint32_t N_INDEX = 0;
constexpr uint32_t C_INDEX = 1;
constexpr uint32_t H_INDEX = 2;
constexpr uint32_t W_INDEX = 3;
constexpr uint32_t DIM_NUM = 4;

constexpr uint32_t NUM_TWO = 2;

constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BF16 = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;

class UpsampleBicubic2dAATiling {
public:
    explicit UpsampleBicubic2dAATiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

 private:
  void SetScale();
  void SetSliceSize();
  inline float ComputeScaleValue(int64_t inputSize, int64_t outputSize, bool alignCornersFlag, const float* scale);
  void GetWorkSpace(uint32_t needCoreNum);
  void GetOutputShape();
  uint8_t GetDataTypeSize() const;
  uint64_t GetTilingKeyVal() const;
  uint32_t GetNeedCoreNumWidth(uint32_t coreNumPlatform);
  uint32_t GetNeedCoreNumHeight(uint32_t coreNumPlatform);
  void FillTilingData();
  void GetTCubeTilingW();
  void GetTCubeTilingH();
  bool CheckShapes() const;

  template <typename T1, typename T2>
  inline T1 CeilA2B(T1 a, T2 b);

  template <typename T1>
  inline int32_t Ceil(T1 x) const;

 private:
  int64_t sliceSize = 0;
  UpsampleBicubic2dAATilingData tilingData;
  gert::TilingContext* tilingContext = nullptr;
  ge::DataType dataType = ge::DT_UNDEFINED;
  uint16_t dataTypeSize = 0;
  gert::Shape inputShape;
  const bool* alignCorners = nullptr;
  const float* scaleH = nullptr;
  const float* scaleW = nullptr;
  float realScaleH = 0.0;
  float realScaleW = 0.0;
  const gert::ContinuousVector* outputSizeVevtor = nullptr;
  int32_t sliceStartListW[MAX_CORE_CONT] = {0};
  int32_t sliceEndListW[MAX_CORE_CONT] = {0};
  int32_t tailSliceStartListW[MAX_CORE_CONT] = {0};
  int32_t tailsliceEndListW[MAX_CORE_CONT] = {0};
  int32_t tailRowStartListW[MAX_CORE_CONT] = {0};
  int32_t tailRowEndListW[MAX_CORE_CONT] = {0};
  int32_t sliceStartListH[MAX_CORE_CONT] = {0};
  int32_t sliceEndListH[MAX_CORE_CONT] = {0};
  int32_t tailSliceStartListH[MAX_CORE_CONT] = {0};
  int32_t tailSliceEndListH[MAX_CORE_CONT] = {0};
  int32_t tailBatchStartListH[MAX_CORE_CONT] = {0};
  int32_t tailBatchEndListH[MAX_CORE_CONT] = {0};

  uint32_t needCoreNumW = 0;
  uint32_t needCoreNumH = 0;

  int32_t outputShapes[4] = {0};
  int32_t inputShapes[4] = {0};

  TCubeTiling matmulTilingW;
  TCubeTiling matmulTilinH;
  int32_t singleCoreKW = 0;
  int32_t singleCoreKH = 0;
};


void UpsampleBicubic2dAATiling::SetScale() {
  const int64_t* outputSizeArray = reinterpret_cast<const int64_t*>(outputSizeVevtor->GetData());

  int64_t outputHeight = outputSizeArray[0];
  int64_t outputWidth = outputSizeArray[1];
  bool alignCornersFlag = *alignCorners;
  realScaleH = ComputeScaleValue(inputShape.GetDim(H_INDEX), outputHeight, alignCornersFlag, scaleH);
  realScaleW = ComputeScaleValue(inputShape.GetDim(W_INDEX), outputWidth, alignCornersFlag, scaleW);

  tilingData.set_scaleH(realScaleH);
  tilingData.set_scaleW(realScaleW);

  float supportW = (realScaleW >= 1.0) ? BICUBIC_SUPPORT_SIZE * realScaleW : BICUBIC_SUPPORT_SIZE;
  float supportH = (realScaleH >= 1.0) ? BICUBIC_SUPPORT_SIZE * realScaleH : BICUBIC_SUPPORT_SIZE;

  tilingData.set_supportW(supportW);
  tilingData.set_supportH(supportH);

  int16_t maxInterpSizeW = Ceil(supportW) * 2 + 1;
  int16_t maxInterpSizeH = Ceil(supportH) * 2 + 1;

  tilingData.set_maxInterpSizeW(maxInterpSizeW);
  tilingData.set_maxInterpSizeH(maxInterpSizeH);

  float invscaleW = 1.0;
  if (realScaleW > 1.0) {
    invscaleW = static_cast<float>(1.0 / realScaleW);
  }
  float invscaleH = 1.0;
  if (realScaleH > 1.0) {
    invscaleH = static_cast<float>(1.0 / realScaleH);
  }

  tilingData.set_invscaleW(invscaleW);
  tilingData.set_invscaleH(invscaleH);
}

inline float UpsampleBicubic2dAATiling::ComputeScaleValue(int64_t inputSize, int64_t outputSize, bool alignCornersFlag,
                                                           const float* scale) {
  if (alignCornersFlag) {
    if (outputSize > 1) {
      return static_cast<float>(inputSize - 1) / (outputSize - 1);
    } else {
      return static_cast<float>(0);
    }
  } else {
    if (inputSize == outputSize) {
      return 1.0;
    }
    return (scale != nullptr && *scale > 0) ? *scale
                                            : (static_cast<float>(inputSize) / outputSize);
  }
}

ge::graphStatus UpsampleBicubic2dAATiling::RunBigKernelTiling() {
  const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  outputSizeVevtor = attrs->GetAttrPointer<gert::ContinuousVector>(0);
  alignCorners = attrs->GetAttrPointer<bool>(1);
  scaleH = attrs->GetAttrPointer<float>(H_INDEX);
  scaleW = attrs->GetAttrPointer<float>(W_INDEX);

  auto tempInputDesc = tilingContext->GetInputDesc(0);
  OP_TILING_CHECK(tempInputDesc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "InputDesc == nullptr"),
                  return ge::GRAPH_FAILED);
  dataType = tempInputDesc->GetDataType();
  dataTypeSize = GetDataTypeSize();

  auto srcShape = tilingContext->GetInputShape(0);
  OP_TILING_CHECK(srcShape == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "InputShape == nullptr"),
                  return ge::GRAPH_FAILED);

  inputShape = srcShape->GetOriginShape();
  if (CheckShapes() == false) {
    return ge::GRAPH_FAILED;
  }

  tilingContext->SetTilingKey(GetTilingKeyVal());

  GetOutputShape();

  SetScale();

  SetSliceSize();

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
  uint32_t coreNumPlatForm = ascendcPlatform.GetCoreNumAic();

  needCoreNumW = GetNeedCoreNumWidth(coreNumPlatForm);
  needCoreNumH = GetNeedCoreNumHeight(coreNumPlatForm);

  uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
  GetWorkSpace(needCoreNum);

  tilingContext->SetBlockDim(needCoreNum);

  GetTCubeTilingW();
  GetTCubeTilingH();

  FillTilingData();
  return ge::GRAPH_SUCCESS;
}

void UpsampleBicubic2dAATiling::SetSliceSize() {
  auto maxScale = realScaleH > realScaleW ? realScaleH : realScaleW;
  if (maxScale <= BEST_PERFORMANCE_SCALE_4) {
    sliceSize = BEST_PERFORMANCE_SIZE_4;
  } else if (maxScale <= BEST_PERFORMANCE_SCALE_3) {
    sliceSize = BEST_PERFORMANCE_SIZE_3;
  } else if (maxScale <= BEST_PERFORMANCE_SCALE_2) {
    sliceSize = BEST_PERFORMANCE_SIZE_2;
  } else {
    sliceSize = BEST_PERFORMANCE_SIZE_1;
  }
  if (sliceSize > outputShapes[H_INDEX] || sliceSize > outputShapes[W_INDEX]) {
    sliceSize = BEST_PERFORMANCE_SIZE_1;
  }
  tilingData.set_sliceSize(sliceSize);
}

bool UpsampleBicubic2dAATiling::CheckShapes() const {
  OP_TILING_CHECK(
    inputShape.GetDimNum() != 4,
    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Input tensor dim num must equal to 4"),
    return false);

  const int64_t* outputSizeArray = reinterpret_cast<const int64_t*>(outputSizeVevtor->GetData());
  int64_t inputH = inputShape.GetDim(2);
  int64_t inputW = inputShape.GetDim(3);
  int64_t outH = outputSizeArray[0];
  int64_t outW = outputSizeArray[1];

  OP_TILING_CHECK(
    !(inputH > 0 && inputW > 0 && outH > 0 && outW > 0),
    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(),
      "Input and output sizes should greater than 0, but got input (H: %ld, W: %ld) output (H: %ld, W: %ld)", 
      inputH, inputW, outH, outW),
    return false);
  
  return true;
}


void UpsampleBicubic2dAATiling::GetTCubeTilingW() {
  auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
  matmul_tiling::MatmulApiTiling mmTilingW;
  mmTilingW.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTilingW.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTilingW.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
  mmTilingW.SetShape(inputShapes[N_INDEX] * inputShapes[C_INDEX] * inputShape[H_INDEX], sliceSize, singleCoreKW);
  mmTilingW.SetOrgShape(inputShapes[N_INDEX] * inputShapes[C_INDEX] * inputShape[H_INDEX], outputShapes[W_INDEX], inputShapes[W_INDEX]);

  if (mmTilingW.GetTiling(tilingData.matmulTilingW) == -1) {
    return;
  }
}

void UpsampleBicubic2dAATiling::GetTCubeTilingH() {
  auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
  matmul_tiling::MatmulApiTiling mmTilingH;
  mmTilingH.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, true);
  mmTilingH.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
  mmTilingH.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
  mmTilingH.SetShape(sliceSize, outputShapes[W_INDEX], singleCoreKH);
  mmTilingH.SetOrgShape(sliceSize, outputShapes[W_INDEX], inputShape[H_INDEX]);

  if (mmTilingH.GetTiling(tilingData.matmulTilingH) == -1) {
    return;
  }
}

void UpsampleBicubic2dAATiling::GetWorkSpace(uint32_t needCoreNum) {
  size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
  uint64_t intermediateMatrixSize =
      static_cast<uint64_t>(outputShapes[N_INDEX]) * outputShapes[C_INDEX] * inputShapes[H_INDEX] * outputShapes[W_INDEX] * dataTypeSize;

  singleCoreKW = Ceil(sliceSize * realScaleW) + Ceil(NUM_TWO * tilingData.get_supportW());
  uint32_t radioMatrixWSize = sliceSize * singleCoreKW * dataTypeSize;
  singleCoreKH = Ceil(sliceSize * realScaleH) + Ceil(NUM_TWO * tilingData.get_supportH());
  uint32_t radioMatrixHSize = sliceSize * singleCoreKH * dataTypeSize;
  uint32_t radioMatrixWorkspaceSize = std::max(radioMatrixWSize, radioMatrixHSize);
  intermediateMatrixSize = (intermediateMatrixSize + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  if (workspaces != nullptr) {
    workspaces[0] = intermediateMatrixSize + radioMatrixWorkspaceSize * needCoreNum * BUFFER_LEN + WORK_SPACE_SIZE;
  }

  tilingData.set_radioMatrixWSize(radioMatrixWSize);
  tilingData.set_radioMatrixHSize(radioMatrixHSize);
  tilingData.set_intermediateMatrixSize(intermediateMatrixSize);
}

void UpsampleBicubic2dAATiling::GetOutputShape() {
  const int64_t* outputSizeArray = reinterpret_cast<const int64_t*>(outputSizeVevtor->GetData());
  for (int8_t i = 0; i < DIM_NUM; i++) {
    inputShapes[i] = inputShape.GetDim(i);
    outputShapes[i] = inputShape.GetDim(i);
    if (i > 1) {
      outputShapes[i] = outputSizeArray[i - NUM_TWO];
    }
  }
  tilingData.set_inputShapes(inputShapes);
  tilingData.set_outputShapes(outputShapes);
}

template <typename T1, typename T2>
inline T1 UpsampleBicubic2dAATiling::CeilA2B(T1 a, T2 b) {
  if (b != 0) {
    return (a + b - 1) / b;
  } else {
    return a;
  }
}

template <typename T1>
inline int32_t UpsampleBicubic2dAATiling::Ceil(T1 x) const {
  int32_t floorX = int32_t(x);
  float closeTo0 = float(1e-6);
  bool equalFlag = false;
  if (x > floorX) {
    equalFlag = (x - floorX < closeTo0) ? true : false;
  } else {
    equalFlag = (floorX - x < closeTo0) ? true : false;
  }
  if (equalFlag) {
    return floorX;
  }
  return floorX + 1;
}

uint8_t UpsampleBicubic2dAATiling::GetDataTypeSize() const {
  switch (dataType) {
    case ge::DT_FLOAT:
      return BYTE_LEN_4;
    case ge::DT_FLOAT16:
      return BYTE_LEN_2;
    case ge::DT_BF16:
      return BYTE_LEN_2;
    default:
      return BYTE_LEN_4;
  }
}

uint64_t UpsampleBicubic2dAATiling::GetTilingKeyVal() const {
  switch (dataType) {
    case ge::DT_FLOAT:
      return TILING_KEY_FLOAT;
    case ge::DT_FLOAT16:
      return TILING_KEY_HALF;
    case ge::DT_BF16:
      return TILING_KEY_BF16;
    default:
      return 0;
  }
}

uint32_t UpsampleBicubic2dAATiling::GetNeedCoreNumWidth(uint32_t coreNumPlatform) {
  if (coreNumPlatform == 0) {
    return 0;
  }
  int64_t outputSize = outputShapes[W_INDEX];
  int64_t sliceCount = CeilA2B(outputSize, sliceSize);
  int64_t eachCoreSliceNum = sliceCount / coreNumPlatform;
  int64_t remainder = sliceCount % coreNumPlatform;

  int64_t inputH = inputShapes[N_INDEX] * inputShapes[C_INDEX] * inputShapes[H_INDEX];

  int64_t minAvergingRows = sliceSize * 2 / dataTypeSize;
  int64_t groupCoreNum = 0;
  if (remainder > 0) {
    groupCoreNum = coreNumPlatform / remainder;
  }
  int64_t tailAvergingRows = std::max(CeilA2B(inputH, groupCoreNum), minAvergingRows);
  groupCoreNum = std::min(groupCoreNum, CeilA2B(inputH, tailAvergingRows));

  int64_t needCoreNum = 0;

  int64_t tailStartSliceNum = eachCoreSliceNum * coreNumPlatform;
  for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
    sliceStartListW[coreIndex] = coreIndex * eachCoreSliceNum * sliceSize;
    sliceEndListW[coreIndex] = (std::min((coreIndex + 1) * eachCoreSliceNum, sliceCount)) * sliceSize;

    if (groupCoreNum == 0) {
      continue;
    }
    int64_t groupIndex = coreIndex / groupCoreNum;
    if (groupIndex < remainder) {
      tailSliceStartListW[coreIndex] = (tailStartSliceNum + groupIndex) * sliceSize;
      tailsliceEndListW[coreIndex] =
          std::min(tailSliceStartListW[coreIndex] + sliceSize, static_cast<int64_t>(outputSize));
      int64_t coreIndexInGroup = coreIndex % groupCoreNum;
      tailRowStartListW[coreIndex] = coreIndexInGroup * tailAvergingRows;
      tailRowEndListW[coreIndex] =
          std::min(tailRowStartListW[coreIndex] + tailAvergingRows, static_cast<int64_t>(inputH));
      needCoreNum++;
    }
  }

  if (eachCoreSliceNum > 0) {
    needCoreNum = coreNumPlatform;
  }

  return needCoreNum;
}

uint32_t UpsampleBicubic2dAATiling::GetNeedCoreNumHeight(uint32_t coreNumPlatform) {
  if (coreNumPlatform == 0) {
    return 0;
  }
  int64_t outputSize = outputShapes[H_INDEX];
  int64_t sliceCount = CeilA2B(outputSize, sliceSize);
  int64_t eachCoreSliceNum = sliceCount / coreNumPlatform;
  int64_t remainder = sliceCount % coreNumPlatform;

  int64_t inputBatch = inputShapes[N_INDEX] * inputShapes[C_INDEX];

  int64_t groupCoreNum = 0;
  if (remainder > 0) {
    groupCoreNum = coreNumPlatform / remainder;
  }
  int64_t tailAvergingBatch = CeilA2B(inputBatch, groupCoreNum);
  groupCoreNum = std::min(groupCoreNum, CeilA2B(inputBatch, tailAvergingBatch));

  int64_t needCoreNum = 0;

  int64_t tailStartSliceNum = eachCoreSliceNum * coreNumPlatform;
  for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
    sliceStartListH[coreIndex] = coreIndex * eachCoreSliceNum * sliceSize;
    sliceEndListH[coreIndex] = (std::min((coreIndex + 1) * eachCoreSliceNum, sliceCount)) * sliceSize;

    if (groupCoreNum == 0) {
      continue;
    }
    int64_t groupIndex = coreIndex / groupCoreNum;
    if (groupIndex < remainder) {
      tailSliceStartListH[coreIndex] = (tailStartSliceNum + groupIndex) * sliceSize;
      tailSliceEndListH[coreIndex] =
          std::min(tailSliceStartListH[coreIndex] + sliceSize, static_cast<int64_t>(outputSize));
      int64_t coreIndexInGroup = coreIndex % groupCoreNum;
      tailBatchStartListH[coreIndex] = coreIndexInGroup * tailAvergingBatch;
      tailBatchEndListH[coreIndex] =
          std::min(tailBatchStartListH[coreIndex] + tailAvergingBatch, static_cast<int64_t>(inputBatch));
      needCoreNum++;
    }
  }

  if (eachCoreSliceNum > 0) {
    needCoreNum = coreNumPlatform;
  }

  return needCoreNum;
}

void UpsampleBicubic2dAATiling::FillTilingData() {
  tilingData.set_needCoreNumW(needCoreNumW);
  tilingData.set_needCoreNumH(needCoreNumH);
  tilingData.set_sliceStartListW(sliceStartListW);
  tilingData.set_sliceEndListW(sliceEndListW);
  tilingData.set_tailSliceStartListW(tailSliceStartListW);
  tilingData.set_tailSliceEndListW(tailsliceEndListW);
  tilingData.set_tailRowStartListW(tailRowStartListW);
  tilingData.set_tailRowEndListW(tailRowEndListW);

  tilingData.set_sliceStartListH(sliceStartListH);
  tilingData.set_sliceEndListH(sliceEndListH);
  tilingData.set_tailSliceStartListH(tailSliceStartListH);
  tilingData.set_tailSliceEndListH(tailSliceEndListH);
  tilingData.set_tailBatchStartListH(tailBatchStartListH);
  tilingData.set_tailBatchEndListH(tailBatchEndListH);

  tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                          tilingContext->GetRawTilingData()->GetCapacity());
  tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus Tiling4UpsampleBicubic2dAA(gert::TilingContext* context) {
    UpsampleBicubic2dAATiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(UpsampleBicubic2dAA).Tiling(Tiling4UpsampleBicubic2dAA);

}  // namespace optiling

