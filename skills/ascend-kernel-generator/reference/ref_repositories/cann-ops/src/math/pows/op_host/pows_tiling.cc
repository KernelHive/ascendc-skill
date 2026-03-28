/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file pows_tiling.cc
 * \brief
 */
#include "pows_tiling.h"
#include "register/op_impl_registry.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

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
static const uint32_t FP32_DTYPE_BYTES = 4;
static const uint32_t FP16_COEXISTING_NUM = 7;
static const uint32_t FP32_COEXISTING_NUM = 3;
static const uint32_t WORK_SPACE_SIZE = 32;
static const uint32_t SPLIT_FACTOR = 2;
static const uint32_t SPLIT_ERROR_STATUS = 10000;
static const int64_t APPROXIMATE_USING_TANH = 1;
static const int64_t APPROXIMATE_USING_ERF = 0;
static const int64_t BYTES_ONE_BLOCK = 32;
static const int64_t MULTI_CORE_SHAPE_SIZE_LIMIT = 4096;
static const int64_t BUFFER_SIZE_ALIGN_LENGTH = 256;

inline static ge::graphStatus SetTilingDataForPows(gert::TilingContext* context, PowsTilingData& tilingData) {
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

static void GetTilingDataNonCut(PowsTilingData& tilingData, TilingParam& tilingParam) {
  tilingData.set_numPerCore(tilingParam.x);
  tilingData.set_realCoreNum(1);
  tilingData.set_mainCoreLoopNum(1);
  tilingData.set_mainCoreTailLength(0);
  tilingData.set_dataLength(tilingParam.x);
  tilingData.set_tailCoreLoopNum(0);
  tilingData.set_tailCoreTailLength(0);
  tilingData.set_bufSize(tilingParam.bufSize / BUFFER_SIZE_ALIGN_LENGTH * BUFFER_SIZE_ALIGN_LENGTH);
  tilingData.set_blockSize(tilingParam.blockSize);
}
static void GetTilingDataBigCase(PowsTilingData& tilingData, TilingParam& tilingParam) {
  int64_t bufSizeAlign = tilingParam.bufSize / BUFFER_SIZE_ALIGN_LENGTH * BUFFER_SIZE_ALIGN_LENGTH;
  int64_t blockFactor = CeilDiv(tilingParam.x, tilingParam.coreNum);
  // block factor align upper
  int64_t blockFactorAlign = (blockFactor + tilingParam.blockSize - 1) / tilingParam.blockSize * tilingParam.blockSize;
  int64_t realCoreNum = CeilDiv(tilingParam.x, blockFactorAlign);

  tilingData.set_numPerCore(blockFactorAlign);
  tilingData.set_realCoreNum(realCoreNum);
  
  // ub factor align down
  int64_t ubFactorAlign = bufSizeAlign / tilingParam.blockSize * tilingParam.blockSize;
  ubFactorAlign = ubFactorAlign > blockFactorAlign ? blockFactorAlign : ubFactorAlign;
  tilingData.set_mainCoreLoopNum(blockFactorAlign /ubFactorAlign);
  tilingData.set_mainCoreTailLength(blockFactorAlign % ubFactorAlign);
  tilingData.set_dataLength(ubFactorAlign);

  if (tilingParam.x % blockFactorAlign != 0) {
    int64_t tailCoreTotalNum = tilingParam.x - blockFactorAlign * (realCoreNum - 1);
    tilingData.set_tailCoreLoopNum(tailCoreTotalNum / ubFactorAlign);
    tilingData.set_tailCoreTailLength(tailCoreTotalNum % ubFactorAlign);
  } else {
    tilingData.set_tailCoreLoopNum(0);
    tilingData.set_tailCoreTailLength(0);
  }
  tilingData.set_bufSize(bufSizeAlign);
  tilingData.set_blockSize(tilingParam.blockSize);
}

static void GetTilingData(PowsTilingData& tilingData, TilingParam& tilingParam) {
  if (tilingParam.x < MULTI_CORE_SHAPE_SIZE_LIMIT) {
    GetTilingDataNonCut(tilingData, tilingParam);
  } else {
    GetTilingDataBigCase(tilingData, tilingParam);
  }
}

static void GetFp16TilingData(PowsTilingData& tilingData, TilingParam& tilingParam) {
  tilingParam.bufSize = tilingParam.ubSize / (FP16_DTYPE_BYTES * FP16_COEXISTING_NUM);
  tilingParam.blockSize = BYTES_ONE_BLOCK / FP16_DTYPE_BYTES;
  GetTilingData(tilingData, tilingParam);
  tilingData.set_tilingKey(static_cast<int64_t>(PowsTilingKey::TILINGKEY_101));
}

static void GetBf16TilingData(PowsTilingData& tilingData, TilingParam& tilingParam) {
  tilingParam.bufSize = tilingParam.ubSize / (FP16_DTYPE_BYTES * FP16_COEXISTING_NUM);
  tilingParam.blockSize = BYTES_ONE_BLOCK / FP16_DTYPE_BYTES;
  GetTilingData(tilingData, tilingParam);
  int64_t tilingkey = static_cast<int64_t>(PowsTilingKey::TILINGKEY_201);
  tilingData.set_tilingKey(tilingkey);
}

static void GetFp32TilingData(PowsTilingData& tilingData, TilingParam& tilingParam) {
  tilingParam.bufSize = tilingParam.ubSize / (FP32_DTYPE_BYTES * FP32_COEXISTING_NUM);
  tilingParam.blockSize = BYTES_ONE_BLOCK / FP32_DTYPE_BYTES;
  GetTilingData(tilingData, tilingParam);
  int64_t tilingkey = static_cast<int64_t>(PowsTilingKey::TILINGKEY_301);
  tilingData.set_tilingKey(tilingkey);
}

static ge::graphStatus CheckInputParams(const gert::TilingContext* context) {
  auto input = context->GetInputTensor(INPUT_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input);

  auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
  int32_t typeSize = ge::GetSizeByDataType(dtype);

  OP_TILING_CHECK(dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "input dtype only support fp16, fp32, bf16 currently, please check."),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK(
      (typeSize <= 0),
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "typeSize is invalid %d, please check.", typeSize),
      return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTillingParam(const gert::TilingContext* context, TilingParam& tilingParam) {
  auto inputShape = context->GetInputTensor(INPUT_IDX)->GetStorageShape();
  // fuse dims
  int64_t x{1};
  for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
      x *= inputShape.GetDim(i);
  }

  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
      return ge::GRAPH_FAILED;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  uint64_t ubSizePlatForm = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  tilingParam.x = x;
  tilingParam.coreNum = coreNum;
  tilingParam.ubSize = ubSizePlatForm;

  OP_LOGI(context->GetNodeName(), "tilingParm is x: %ld, coreNum: %ld, ubSize: %ld", 
                                  tilingParam.x, tilingParam.coreNum, tilingParam.ubSize);
  return ge::GRAPH_SUCCESS;
}

static void GetTillingData(ge::DataType dtype, TilingParam& tilingParam, PowsTilingData& tilingData) {
  if (dtype == ge::DT_FLOAT16) {
    GetFp16TilingData(tilingData, tilingParam);
  } else if (dtype == ge::DT_BF16) {
    GetBf16TilingData(tilingData, tilingParam);
  } else {
    GetFp32TilingData(tilingData, tilingParam);
  }
}

static ge::graphStatus Tiling4Pows(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "Tiling4Pows enter.");
  OP_TILING_CHECK(CheckInputParams(context) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "InputParams not valid."),
                  return ge::GRAPH_FAILED);

  TilingParam tilingParam;
  OP_TILING_CHECK(GetTillingParam(context, tilingParam) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Get Tiling Param Failed."),
                  return ge::GRAPH_FAILED);

  auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
  PowsTilingData tilingData;

  GetTillingData(dtype, tilingParam, tilingData); 

  OP_TILING_CHECK(
      SetTilingDataForPows(context, tilingData) != ge::GRAPH_SUCCESS,
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "PowsSetTilingData set tiling data fail."),
      return ge::GRAPH_FAILED);

  context->SetBlockDim(tilingData.get_realCoreNum());
  context->SetTilingKey(tilingData.get_tilingKey());
  size_t* workspaces = context->GetWorkspaceSizes(1);
  workspaces[0] = WORK_SPACE_SIZE + tilingParam.coreNum * BYTES_ONE_BLOCK;

  OP_LOGI(context->GetNodeName(),
          "tilingData is bufSize: %ld, tilingKey: %ld, numPerCore: %ld, realCoreNum: %ld, \
           mainCoreLoopNum: %ld, mainCoreTailLength: %ld, tailCoreloopNum: %ld, tailCoreTailLength: %ld, \
           dataLength: %ld, blockSize: %ld",
           tilingData.get_bufSize(), tilingData.get_tilingKey(),
           tilingData.get_numPerCore(), tilingData.get_realCoreNum(), tilingData.get_mainCoreLoopNum(),
           tilingData.get_mainCoreTailLength(), tilingData.get_tailCoreLoopNum(), tilingData.get_tailCoreTailLength(),
           tilingData.get_dataLength(), tilingData.get_blockSize());

  OP_LOGD(context->GetNodeName(), "Tiling4Pows exit.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Pows).Tiling(Tiling4Pows);
}  // namespace optiling
