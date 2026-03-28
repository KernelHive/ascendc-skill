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
 * \file dynamic_rnnv2.cpp
 * \brief op_host of dynamic_rnnv2
 */
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "dynamic_rnnv2.h"
#include "tiling/tiling_api.h"

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)

bool AddWorkspaceRNNV2(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  *workspace_size = workspace;
  return true;
}

void LstmTilingRNNV2::GetAICoreIntrinsicDtype(fe::PlatFormInfos& platform_info, const std::string& intrinsic_name, bool& value) {
  std::string val;
  (void)platform_info.GetPlatformRes("AICoreintrinsicDtypeMap", intrinsic_name, val);

  if (!val.empty()) {
    value = true;
  } else {
    value = false;
  }

  return;
}

ge::graphStatus LstmTilingRNNV2::TilingWithAscendC(gert::TilingContext* context) {
  OP_TILING_CHECK(!CheckParamsShape(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check shape fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!CheckParamsDtype(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check dtype fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!CheckAttr(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check attr fail."),
                  return ge::GRAPH_FAILED);

  // get op info
  OP_TILING_CHECK(GetOpInfo(context, rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get opinfo fail."),
                  return ge::GRAPH_FAILED);
  // get attr
  OP_TILING_CHECK(GetAttr(context, rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get attr fail."),
                  return ge::GRAPH_FAILED);

  // get UB L1 size
  auto platformInfo = context->GetPlatformInfo();

  int32_t doubleNum = 2;
  rnnParams.sysAicCoreNum = context->GetPlatformInfo()->GetCoreNum();
  rnnParams.sysAivCoreNum = rnnParams.sysAicCoreNum * doubleNum;
  platformInfo->GetLocalMemSize(fe::LocalMemType::UB, rnnParams.ubSize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L1, rnnParams.l1Size);

  // get matmul tiling data
  OP_TILING_CHECK(
      GetMMTilingData(context, tilingData, rnnParams) != ge::GRAPH_SUCCESS,
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get matmul tiling data fail."),
      return ge::GRAPH_FAILED);

  OP_TILING_CHECK(CalcTilingKey(rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling key fail."),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK(SetTilingData(context, tilingData, rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tiling data fail."),
                  return ge::GRAPH_FAILED);

  int64_t workspaceSize = rnnParams.timeStep * rnnParams.batch * 4 * rnnParams.hiddenSize * 4 + 20 * 1024 * 1024;
  auto launchCore = (rnnParams.usedCoreNum + DEFAULT_INDEX_TWO - 1) / DEFAULT_INDEX_TWO;
  context->SetBlockDim(launchCore);  // 24上限
  context->SetTilingKey(rnnParams.tilingKey);
  AddWorkspaceRNNV2(context, workspaceSize);

  return ge::GRAPH_SUCCESS;
}

 bool LstmTilingRNNV2::CheckParamsDtype(const gert::TilingContext* context) {
  // dtype support list
  std::vector<ge::DataType> supportDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};

  // all params dtype list
  std::vector<ge::DataType> paramsDtype;
  int32_t inputDesc = 2;
  paramsDtype.push_back(context->GetInputDesc(0)->GetDataType());
  paramsDtype.push_back(context->GetInputDesc(1)->GetDataType());
  paramsDtype.push_back(context->GetInputDesc(inputDesc)->GetDataType());

  int32_t inputShapeSize = 4;
  auto inithShape = context->GetOptionalInputShape(inputShapeSize);
  if (inithShape != nullptr) {
    paramsDtype.push_back(context->GetInputDesc(inputShapeSize)->GetDataType());
  }

  for (auto dtype : paramsDtype) {
    OP_TILING_CHECK(
        std::find(supportDtype.begin(), supportDtype.end(), dtype) == supportDtype.end(),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input dtype error, please check."),
        return false);
  }

  return true;
}

bool LstmTilingRNNV2::CheckAttr(gert::TilingContext* context) {
  bool ret = CheckAttrOps(context);
  if (ret) {
    ret = CheckAttrTiling(context);
  }

  return ret;
}

ge::graphStatus LstmTilingRNNV2::GetMMTilingData(gert::TilingContext* context, DynamicRNNTilingData& tilingData,
                                          DynamicRnnTiling& rnnParams) {
  int32_t ubSeq = 10;
  int32_t ubNoSeq = 8;
  vector<int64_t> dims = {MIN_BASE_SHAPE};
  uint32_t sigmoidMaxSize = 0;
  uint32_t sigmoidMinSize = 0;
  uint32_t tanhMaxSize = 0;
  uint32_t tanhMinSize = 0;
  AscendC::GetSigmoidMaxMinTmpSize(ge::Shape(dims), CONST_TWO, false, sigmoidMaxSize, sigmoidMinSize);
  AscendC::GetTanhMaxMinTmpSize(ge::Shape(dims), CONST_TWO, false, tanhMaxSize, tanhMinSize);
  uint32_t apiUbSize = sigmoidMinSize > tanhMinSize ? sigmoidMinSize : tanhMinSize;
  uint32_t multiple = (apiUbSize + MIN_BASE_BUFFER - 1) / MIN_BASE_BUFFER;
  if (rnnParams.isSeqLength == 1) {
    rnnParams.maxUbSize = rnnParams.ubSize / (ubSeq + multiple);
  } else {
    rnnParams.maxUbSize = rnnParams.ubSize / (ubNoSeq + multiple);
  }
  auto dataType = context->GetInputDesc(0)->GetDataType();

  rnnParams.dataType = dataType;
  rnnParams.isUseMerged = false;
  rnnParams.isFullLoad = false;

  auto ret = GetMMTilingDataSplit(context, tilingData, rnnParams, static_cast<matmul_tiling::DataType>(dataType));

  return ret;
}

ge::graphStatus LstmTilingRNNV2::GetMMTilingDataSplit(const gert::TilingContext* context, DynamicRNNTilingData& tilingData,
                                                 DynamicRnnTiling& rnnParams, matmul_tiling::DataType dataType) {
  int32_t hiddenBlock = 4;
  int64_t aivDouble = 2;
  matmul_tiling::MultiCoreMatmulTiling rnnMatmul1;
  rnnParams.usedCoreNum = context->GetPlatformInfo()->GetCoreNum() * aivDouble;
  auto ret = rnnMatmul1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetAType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetAddWorkspaceBType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetCType fail."),
                  return ge::GRAPH_FAILED);

  if (rnnParams.isBias) {
    rnnMatmul1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    ret = rnnMatmul1.SetBias(true);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetBias fail."),
                    return ge::GRAPH_FAILED);
  }

  ret = rnnMatmul1.SetDim(rnnParams.sysAivCoreNum);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetDim fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetOrgShape(rnnParams.timeStep * rnnParams.batch, rnnParams.hiddenSize * hiddenBlock,
                               rnnParams.inputSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetOrgShape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetShape(rnnParams.timeStep * rnnParams.batch, rnnParams.hiddenSize * hiddenBlock,
                            rnnParams.inputSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 Set single shape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetBufferSpace(-1, -1, -1);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetBufferSpace fail."),
                  return ge::GRAPH_FAILED);

  ret = rnnMatmul1.GetTiling(tilingData.inputMMParam);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 GetTiling fail."),
                  return ge::GRAPH_FAILED);

  matmul_tiling::MultiCoreMatmulTiling rnnMatmul2;
  ret = rnnMatmul2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetAType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetBType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetCType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetDim(rnnParams.sysAivCoreNum);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetDim fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetOrgShape(rnnParams.batch, rnnParams.hiddenSize * hiddenBlock, rnnParams.hiddenSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetOrgShape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetShape(rnnParams.batch, rnnParams.hiddenSize * hiddenBlock, rnnParams.hiddenSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 Set single shape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetBufferSpace(-1, -1, -1);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetBufferSpace fail."),
                  return ge::GRAPH_FAILED);

  ret = rnnMatmul2.GetTiling(tilingData.hiddenMMParam);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 GetTiling fail."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LstmTilingRNNV2::CalcTilingKey(DynamicRnnTiling& rnnParams) {
  // 判断是否需要切分L0c输出，分次搬入UB
  int64_t tilingKey = 0;

  if (rnnParams.dataType == 1) {
    tilingKey = static_cast<int64_t>(RNNTilingKey::MM_FP16_SPLIT);
  } else if (rnnParams.dataType == 0) {
    tilingKey = static_cast<int64_t>(RNNTilingKey::MM_FP32_SPLIT);
  }

  rnnParams.tilingKey = tilingKey;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LstmTilingRNNV2::SetTilingData(gert::TilingContext* context, DynamicRNNTilingData& tilingData,
                                        DynamicRnnTiling& rnnParams) {
  // 无效数据附初值
  rnnParams.isHF32 = 0;
  rnnParams.isCached = 0;
  rnnParams.cacheLength = 0;

  tilingData.set_tilingKey(rnnParams.tilingKey);
  tilingData.set_usedCoreNum(rnnParams.usedCoreNum);
  tilingData.set_timeStep(rnnParams.timeStep);
  tilingData.set_batch(rnnParams.batch);
  tilingData.set_inputSize(rnnParams.inputSize);
  tilingData.set_hiddenSize(rnnParams.hiddenSize);
  tilingData.set_isBias(rnnParams.isBias);
  tilingData.set_isInithc(rnnParams.isInithc);
  tilingData.set_isSeqLength(rnnParams.isSeqLength);
  tilingData.set_isHF32(rnnParams.isHF32);

  tilingData.set_isCached(rnnParams.isCached);
  tilingData.set_cacheLength(rnnParams.cacheLength);

  tilingData.set_gateOrder(rnnParams.gateOrder);
  tilingData.set_direction(rnnParams.direction);
  tilingData.set_isTraining(rnnParams.isTraining);
  tilingData.set_cellClip(rnnParams.cellClip);
  tilingData.set_forgetBias(rnnParams.forgetBias);

  gert::TilingData* rnnRawTilingData = context->GetRawTilingData();
  OP_TILING_CHECK(tilingData.GetDataSize() > rnnRawTilingData->GetCapacity(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context, "actual tiling data size %zu > context tiling data size %zu",
                                                  tilingData.GetDataSize(), rnnRawTilingData->GetCapacity()),
                  return ge::GRAPH_FAILED);
  tilingData.SaveToBuffer(rnnRawTilingData->GetData(), rnnRawTilingData->GetCapacity());
  rnnRawTilingData->SetDataSize(tilingData.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

bool DynamicRNNV2Tiling::CheckInitParamsShape(gert::TilingContext* context) {
  auto bInput = context->GetOptionalInputShape(3);
  auto seqInput = context->GetOptionalInputShape(4);
  auto inithInput = context->GetOptionalInputShape(5);
  auto initcInput = context->GetOptionalInputShape(6);
  auto outputShape = context->GetOutputShape(0)->GetStorageShape();

  if (bInput != nullptr) {
    auto bShape = bInput->GetStorageShape();
  }
  
  auto inithShape = inithInput->GetStorageShape();
  auto initcShape = initcInput->GetStorageShape();
  return true;
}

bool DynamicRNNV2Tiling::CheckAttrOps(gert::TilingContext* context) {
  // get attr
  auto attrs = context->GetAttrs();
  const char* cellType = attrs->GetAttrPointer<char>(0);
  const char* direction = attrs->GetAttrPointer<char>(1);
  const int* cellDepth = attrs->GetAttrPointer<int>(2);
  const bool* usePeephole = attrs->GetAttrPointer<bool>(3);
  const float* keepProb = attrs->GetAttrPointer<float>(4);
  const float* cellClip = attrs->GetAttrPointer<float>(5);
  std::vector<std::string> supportDirection = {"UNIDIRECTIONAL", "REDIRECTIONAL"};
  return true;
}

bool DynamicRNNV2Tiling::CheckAttrTiling(gert::TilingContext* context) {
  // get attr
  auto attrs = context->GetAttrs();
  const int* numProj = attrs->GetAttrPointer<int>(6);
  const bool* timeMajor = attrs->GetAttrPointer<bool>(7);
  const char* activation = attrs->GetAttrPointer<char>(8);
  const char* recurrentActivation = attrs->GetAttrPointer<char>(9);
  const float* forgetBias = attrs->GetAttrPointer<float>(10);
  const char* gateOrder = attrs->GetAttrPointer<char>(11);
  const bool* stateful = attrs->GetAttrPointer<bool>(12);
  const char* mergeMode = attrs->GetAttrPointer<char>(13);
  const bool* isTraining = attrs->GetAttrPointer<bool>(14);
  std::vector<std::string> supportGateOrder = {"ifco", "ijfo"};
  std::vector<std::string> supportMergeMode = {"concat", "add"};
  return true;
}

ge::graphStatus DynamicRNNV2Tiling::GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& rnnParams) {

  // get x shape
  auto xTensor = context->GetInputShape(0);
  auto xShape = xTensor->GetStorageShape();
  // get w shape
  auto weightHiddenTensor = context->GetInputShape(2);
  auto weightHiddenShape = weightHiddenTensor->GetStorageShape();

  // get bias seq_length, init_h, init_C
  auto biasShape = context->GetOptionalInputShape(3);
  auto seqShape = context->GetOptionalInputShape(4);
  auto inithShape = context->GetOptionalInputShape(5);
  auto initcShape = context->GetOptionalInputShape(6);

  biasShape != nullptr ? rnnParams.isBias = 1 : rnnParams.isBias = 0;
  seqShape != nullptr ? rnnParams.isSeqLength = 1 : rnnParams.isSeqLength = 0;
  (inithShape != nullptr && initcShape != nullptr) ? rnnParams.isInithc = 1 : rnnParams.isInithc = 0;

  int32_t dim = 2;

  rnnParams.timeStep = static_cast<int64_t>(xShape.GetDim(0));
  rnnParams.batch = static_cast<int64_t>(xShape.GetDim(1));
  rnnParams.inputSize = static_cast<int64_t>(xShape.GetDim(dim));
  rnnParams.hiddenSize = static_cast<int64_t>(weightHiddenShape.GetDim(0));

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicRNNV2Tiling::GetAttr(const gert::TilingContext* context, DynamicRnnTiling& rnnParams) {
  // get attr
  auto attrs = context->GetAttrs();

  // get gate_order
  const char* gateOrder = attrs->GetAttrPointer<char>(11);
  if (strcmp(gateOrder, "ijfo") == 0) {
    rnnParams.gateOrder = static_cast<int64_t>(GateOrder::IJFO);
  } else {
    rnnParams.gateOrder = static_cast<int64_t>(GateOrder::IFJO);
  }

  // set direction
  rnnParams.direction = 0;

  // get cell_clip
  const float* cellClip = attrs->GetAttrPointer<float>(5);
  rnnParams.cellClip = *cellClip;

  // get forget_bias
  const float* forgetBias = attrs->GetAttrPointer<float>(10);
  rnnParams.forgetBias = *forgetBias;

  // get is_training
  const bool* isTraining = attrs->GetAttrPointer<bool>(14);
  rnnParams.isTraining = *isTraining;

  return ge::GRAPH_SUCCESS;
}

bool DynamicRNNV2Tiling::CheckParamsShape(gert::TilingContext* context) {
  // get input shape
  auto xInput = context->GetInputShape(0);
  auto xShape = xInput->GetStorageShape();
  // get weight input shape
  auto wInput = context->GetInputShape(1);
  auto wInputShape = wInput->GetStorageShape();

  // get weight hidden shape
  auto wHidden = context->GetInputShape(2);
  auto wHiddenShape = wHidden->GetStorageShape();

  // get output y shape
  auto outputY = context->GetOutputShape(0);
  auto outputShape = outputY->GetStorageShape();
  bool ret = CheckInitParamsShape(context);

  return ret;
}

ge::graphStatus Tiling4DynamicRNNV2(gert::TilingContext* context) {
  context->SetScheduleMode(SCHEDULE_MODE);
  // AscendC
  bool supportL0c2out = true;
  DynamicRNNV2Tiling rnnv2Tiling;
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  if (supportL0c2out) {
    OP_TILING_CHECK(rnnv2Tiling.TilingWithAscendC(context) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "DynamicrnnV2 with ascendc have error."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4DynamicRNNV2(gert::TilingParseContext* context) {
  // AscendC
  bool supportL0c2out = false;
  DynamicRNNV2Tiling rnnv2Tiling;
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicRNNV2).Tiling(Tiling4DynamicRNNV2).TilingParse<DynamicRNNV2CompileInfo>(TilingPrepare4DynamicRNNV2);
}  // namespace optiling

namespace ge {
  namespace ops {
  constexpr int DY_SHAPE_SIZE_LIMIT = 3;
  constexpr int X_SHAPE_SIZE_LIMIT = 3;
  constexpr int WEIGHT_INPUT_SIZE_LIMIT = 2;
  constexpr int ALIGN_UNITS = 16;
  constexpr int CONSTANT_FOUR = 4;
  constexpr int CONSTANT_THREE = 3;
  constexpr int CONSTANT_TWO = 2;
  constexpr int CONSTANT_ONE = 1;
  constexpr int CONSTANT_ZERO = 0;
  constexpr int GATES_NUMBERS = 3;
  constexpr int INPUT_INDEX_OFFSET = 2;
  constexpr int RNN_OUTPUT_INDEX_C = 2;
  constexpr int RNN_OUTPUT_INDEX_I = 3;
  constexpr int RNN_OUTPUT_INDEX_J = 4;
  constexpr int RNN_OUTPUT_INDEX_F = 5;
  constexpr int RNN_OUTPUT_INDEX_O = 6;
  constexpr int RNN_OUTPUT_INDEX_TANHC = 7;
  constexpr int64_t UNKNOWN_DIM_VALUE = -1;

  static ge::graphStatus InferDataType4DynamicRNNV2(gert::InferDataTypeContext* context) {
    auto input_x_dtype = context->GetInputDataType(0);
    auto output_y_dtype = input_x_dtype;

    auto bias_dtype = context->GetOptionalInputDataType(3);
    auto initc_dtype = context->GetOptionalInputDataType(6);
    if (bias_dtype != ge::DT_UNDEFINED) {
      output_y_dtype = bias_dtype;
    } else if (initc_dtype != ge::DT_UNDEFINED) {
      output_y_dtype = initc_dtype;
    }

    return ge::GRAPH_SUCCESS;
  }

  static ge::graphStatus InferShape4DynamicRNNV2(gert::InferShapeContext* context) {
    auto x_shape = context->GetInputShape(0);

    auto weight_hidden_shape = context->GetInputShape(2);

    auto y_shape = context->GetOutputShape(0);
    auto outputh_shape = context->GetOutputShape(1);
    auto outputc_shape = context->GetOutputShape(2);
    auto i_shape = context->GetOutputShape(3);
    auto j_shape = context->GetOutputShape(4);
    auto f_shape = context->GetOutputShape(5);
    auto o_shape = context->GetOutputShape(6);
    auto tanhc_shape = context->GetOutputShape(7);

    int64_t num_step = x_shape->GetDim(0);
    int64_t batch_size = x_shape->GetDim(1);
    int64_t hidden_size = weight_hidden_shape->GetDim(0);

    *y_shape = {num_step, batch_size, hidden_size};
    *outputh_shape = {1, batch_size, hidden_size};
    *outputc_shape = {1, batch_size, hidden_size};
    *i_shape = {num_step, batch_size, hidden_size};
    *j_shape = {num_step, batch_size, hidden_size};
    *f_shape = {num_step, batch_size, hidden_size};
    *o_shape = {num_step, batch_size, hidden_size};
    *tanhc_shape = {num_step, batch_size, hidden_size};

    return GRAPH_SUCCESS;
  }

  IMPL_OP_INFERSHAPE(DynamicRNNV2).InferShape(InferShape4DynamicRNNV2).InferDataType(InferDataType4DynamicRNNV2);

  }  // namespace ops
} // namespace ge

namespace ops {

class DynamicRNNV2 : public OpDef {
public:
    explicit DynamicRNNV2(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weight_input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weight_hidden")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("b")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("seq_length")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("init_h")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("init_c")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("wci")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("wcf")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("wco")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("mask")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("output_h")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("output_c")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("i")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("j")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("f")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("o")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});   
    this->Output("tanhc")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});  
    this->Attr("cell_type").AttrType(REQUIRED).String("LSTM");
    this->Attr("direction").AttrType(REQUIRED).String("UNIDIRECTIONAL");
    this->Attr("cell_depth").AttrType(REQUIRED).Int(1);
    this->Attr("use_peephole").AttrType(REQUIRED).Bool(false);
    this->Attr("keep_prob").AttrType(REQUIRED).Float(1.0);
    this->Attr("cell_clip").AttrType(REQUIRED).Float(-1.0);
    this->Attr("num_proj").AttrType(REQUIRED).Int(0);
    this->Attr("time_major").AttrType(REQUIRED).Bool(true);
    this->Attr("activation").AttrType(REQUIRED).String("tanh");
    this->Attr("recurrent_activation").AttrType(REQUIRED).String("sigmoid");
    this->Attr("forget_bias").AttrType(REQUIRED).Float(0.0);
    this->Attr("gate_order").AttrType(REQUIRED).String("ijfo");
    this->Attr("stateful").AttrType(REQUIRED).Bool(false);
    this->Attr("merge_mode").AttrType(REQUIRED).String("concat");
    this->Attr("is_training").AttrType(REQUIRED).Bool(true);
    this->AICore()
        .AddConfig("ascend910b");
    }
};

OP_ADD(DynamicRNNV2);
}  // namespace ops