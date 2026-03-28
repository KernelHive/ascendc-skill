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
 * \file dynamic_rnn.cpp
 * \brief op_host of dynamic_rnn
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
#include "dynamic_rnn.h"
#include "tiling/tiling_api.h"

namespace optiling
{
// tools api
#define OP_LOGD(nodeName, fmt, ...) \
  std::printf(fmt, ##__VA_ARGS__);  \
  std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...) \
  std::printf(fmt, ##__VA_ARGS__);  \
  std::printf("\n")
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do                                          \
  {                                           \
    if (cond)                                 \
    {                                         \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr)                                                                          \
  {                                                                                              \
    const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    std::printf(name, "is nullptr!");                                                            \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
    return ge::GRAPH_FAILED;                                                                     \
  }
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
  if ((ptr) == nullptr)                                                                          \
  {                                                                                              \
    const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    std::printf(name, "is nullptr!");                                                            \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
    return ret;                                                                                  \
  }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do                                                                                                           \
  {                                                                                                            \
    if (unlikely(condition))                                                                                   \
    {                                                                                                          \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)
#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
  do                                                          \
  {                                                           \
    std::printf("op[%s], %s", op_name, err_msg);              \
  } while (0)

#define OP_CHECK(cond, log_func, return_expr) \
  if (cond)                                   \
  {                                           \
    log_func;                                 \
    return_expr;                              \
  }

  bool AddWorkspaceRNN(gert::TilingContext *context, const size_t workspace)
  {
    size_t *workspace_size = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
    *workspace_size = workspace;
    return true;
  }

  // tiling_lstm
  void LstmTilingRNN::GetAICoreIntrinsicDtype(fe::PlatFormInfos &platform_info, const string &intrinsic_name, bool &value)
  {
    std::string val;
    (void)platform_info.GetPlatformRes("AICoreintrinsicDtypeMap", intrinsic_name, val);

    if (!val.empty())
    {
      OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s : %s", intrinsic_name.c_str(), val.c_str());
      value = true;
    }
    else
    {
      value = false;
      OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", intrinsic_name.c_str());
    }

    return;
  }

  ge::graphStatus LstmTilingRNN::TilingWithAscendC(gert::TilingContext *context)
  {
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
    context->SetBlockDim(launchCore); // 24上限
    context->SetTilingKey(rnnParams.tilingKey);
    AddWorkspaceRNN(context, workspaceSize);
    return ge::GRAPH_SUCCESS;
  }

  bool LstmTilingRNN::CheckParamsDtype(const gert::TilingContext *context)
  {
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
    if (inithShape != nullptr)
    {
      paramsDtype.push_back(context->GetInputDesc(inputShapeSize)->GetDataType());
    }

    for (auto dtype : paramsDtype)
    {
      OP_TILING_CHECK(
          std::find(supportDtype.begin(), supportDtype.end(), dtype) == supportDtype.end(),
          VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input dtype error, please check."),
          return false);
    }
    return true;
  }

  bool LstmTilingRNN::CheckAttr(gert::TilingContext *context)
  {
    bool ret = CheckAttrOps(context);
    if (ret)
    {
      ret = CheckAttrTiling(context);
    }
    return ret;
  }

  ge::graphStatus LstmTilingRNN::GetMMTilingData(gert::TilingContext *context, DynamicRNNTilingData &tilingData,
                                                 DynamicRnnTiling &rnnParams)
  {
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
    if (rnnParams.isSeqLength == 1)
    {
      rnnParams.maxUbSize = rnnParams.ubSize / (ubSeq + multiple);
    }
    else
    {
      rnnParams.maxUbSize = rnnParams.ubSize / (ubNoSeq + multiple);
    }
    auto dataType = context->GetInputDesc(0)->GetDataType();

    rnnParams.dataType = dataType;
    rnnParams.isUseMerged = false;
    rnnParams.isFullLoad = false;

    auto ret = GetMMTilingDataSplit(context, tilingData, rnnParams, static_cast<matmul_tiling::DataType>(dataType));
    return ret;
  }

  ge::graphStatus LstmTilingRNN::GetMMTilingDataSplit(const gert::TilingContext *context, DynamicRNNTilingData &tilingData,
                                                      DynamicRnnTiling &rnnParams, matmul_tiling::DataType dataType)
  {
    int32_t hiddenBlock = 4;
    int64_t aivDouble = 2;
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul1;
    rnnParams.usedCoreNum = context->GetPlatformInfo()->GetCoreNum() * aivDouble;
    auto ret = rnnMatmul1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetBType fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                              matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetCType fail."),
                    return ge::GRAPH_FAILED);

    if (rnnParams.isBias)
    {
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

  ge::graphStatus LstmTilingRNN::CalcTilingKey(DynamicRnnTiling &rnnParams)
  {
    // 判断是否需要切分L0c输出，分次搬入UB
    int64_t tilingKey = 0;

    if (rnnParams.dataType == 1)
    {
      tilingKey = static_cast<int64_t>(RNNTilingKey::MM_FP16_SPLIT);
    }
    else if (rnnParams.dataType == 0)
    {
      tilingKey = static_cast<int64_t>(RNNTilingKey::MM_FP32_SPLIT);
    }

    rnnParams.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus LstmTilingRNN::SetTilingData(gert::TilingContext *context, DynamicRNNTilingData &tilingData,
                                               DynamicRnnTiling &rnnParams)
  {
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

    gert::TilingData *rnnRawTilingData = context->GetRawTilingData();
    OP_LOGE_IF(rnnRawTilingData == nullptr, ge::GRAPH_FAILED, context->GetNodeType(), "GetRawTilingData failed.");
    OP_TILING_CHECK(tilingData.GetDataSize() > rnnRawTilingData->GetCapacity(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "actual tiling data size %zu > context tiling data size %zu",
                                                    tilingData.GetDataSize(), rnnRawTilingData->GetCapacity()),
                    return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rnnRawTilingData->GetData(), rnnRawTilingData->GetCapacity());
    rnnRawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus DynamicRNNTiling::GetOpInfo(const gert::TilingContext *context, DynamicRnnTiling &rnnParams)
  {
    OP_TILING_CHECK(context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input shape error."),
                    return ge::GRAPH_FAILED);

    // get x shape
    auto xTensor = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    auto xShape = xTensor->GetStorageShape();

    // get w shape
    auto weightTensor = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightTensor);
    auto weightShape = weightTensor->GetStorageShape();

    // get seq_length, init_h, init_C
    auto biasShape = context->GetInputShape(2);
    auto seqShape = context->GetOptionalInputShape(3);
    auto inithShape = context->GetOptionalInputShape(4);
    auto initcShape = context->GetOptionalInputShape(5);

    biasShape != nullptr ? rnnParams.isBias = 1 : rnnParams.isBias = 0;
    seqShape != nullptr ? rnnParams.isSeqLength = 1 : rnnParams.isSeqLength = 0;
    (inithShape != nullptr && initcShape != nullptr) ? rnnParams.isInithc = 1 : rnnParams.isInithc = 0;

    int32_t dim = 2;
    OP_TILING_CHECK(xShape.GetDimNum() != 3,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get x shape dim is not 3, please check."),
                    return false);
    OP_TILING_CHECK(xShape.GetDim(CONST_TWO) == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn input_size not support 0, please check."),
                    return false);
    OP_TILING_CHECK(weightShape.GetDimNum() != 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get weight shape dim is not 2, please check."),
                    return false);
    rnnParams.timeStep = static_cast<int64_t>(xShape.GetDim(0));
    rnnParams.batch = static_cast<int64_t>(xShape.GetDim(1));
    rnnParams.inputSize = static_cast<int64_t>(xShape.GetDim(dim));
    rnnParams.hiddenSize = static_cast<int64_t>(weightShape.GetDim(0)) - rnnParams.inputSize;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus DynamicRNNTiling::GetAttr(const gert::TilingContext *context, DynamicRnnTiling &rnnParams)
  {
    // get attr
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // get gate_order
    const char *gateOrder = attrs->GetAttrPointer<char>(10);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gateOrder);
    if (strcmp(gateOrder, "ijfo") == 0)
    {
      rnnParams.gateOrder = static_cast<int64_t>(GateOrder::IJFO);
    }
    else
    {
      rnnParams.gateOrder = static_cast<int64_t>(GateOrder::IFJO);
    }

    // get direction
    const char *direction = attrs->GetAttrPointer<char>(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, direction);
    if (strcmp(direction, "UNIDIRECTIONAL") == 0)
    {
      rnnParams.direction = 0;
    }
    else
    {
      rnnParams.direction = 1;
    }

    // get cell_clip
    const float *cellClip = attrs->GetAttrPointer<float>(5);
    OPS_CHECK_NULL_WITH_CONTEXT(context, cellClip);
    rnnParams.cellClip = *cellClip;

    // get forget_bias
    const float *forgetBias = attrs->GetAttrPointer<float>(9);
    OPS_CHECK_NULL_WITH_CONTEXT(context, forgetBias);
    rnnParams.forgetBias = *forgetBias;

    // get is_training
    const bool *isTraining = attrs->GetAttrPointer<bool>(11);
    OPS_CHECK_NULL_WITH_CONTEXT(context, isTraining);
    rnnParams.isTraining = *isTraining;
    return ge::GRAPH_SUCCESS;
  }

  bool DynamicRNNTiling::CheckParamsShape(gert::TilingContext *context)
  {
    // get input shape
    auto xInput = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, xInput, false);
    auto xShape = xInput->GetStorageShape();
    // get wight shape
    auto wInput = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, wInput, false);
    auto wShape = wInput->GetStorageShape();

    // get bias shape
    auto bInput = context->GetInputShape(2);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, bInput, false);
    auto bShape = bInput->GetStorageShape();

    // get output y shape
    auto outputY = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, outputY, false);
    auto outputShape = outputY->GetStorageShape();

    // check dim num
    OP_TILING_CHECK(xShape.GetDimNum() != 3,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get x shape dim is not 3, please check."),
                    return false);
    OP_TILING_CHECK(wShape.GetDimNum() != 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get w shape dim is not 2, please check."),
                    return false);
    OP_TILING_CHECK(bShape.GetDimNum() != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get b shape dim is not 1, please check."),
                    return false);

    OP_TILING_CHECK(outputShape.GetDimNum() != 3,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get output shape dim is not 3, please check."),
                    return false);

    // check batch dim
    OP_TILING_CHECK(xShape.GetDim(1) != outputShape.GetDim(1),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get x, output batch is not equal, please check."),
                    return false);

    // check x/w input_size dim
    OP_TILING_CHECK(wShape.GetDim(0) != xShape.GetDim(2) + outputShape.GetDim(2),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get w shape dim0 is wrong, please check."),
                    return false);

    // check hidden dim
    OP_TILING_CHECK(wShape.GetDim(1) != 4 * outputShape.GetDim(2),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get w shape dim1 is wrong, please check."),
                    return false);
    OP_TILING_CHECK(bShape.GetDim(0) != 4 * outputShape.GetDim(2),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get b shape dim0 is wrong, please check."),
                    return false);
    bool ret = CheckInitParamsShape(context);
    return ret;
  }

  bool DynamicRNNTiling::CheckInitParamsShape(gert::TilingContext *context)
  {
    auto seqInput = context->GetOptionalInputShape(3);
    auto inithInput = context->GetOptionalInputShape(4);
    auto initcInput = context->GetOptionalInputShape(5);

    // get output y shape
    auto outputY = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, outputY, false);
    auto outputShape = outputY->GetStorageShape();

    OP_TILING_CHECK(outputShape.GetDimNum() != 3,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn get output shape dim is not 3, please check."),
                    return false);

    if (inithInput != nullptr && initcInput != nullptr)
    {
      auto inithShape = inithInput->GetStorageShape();
      auto initcShape = initcInput->GetStorageShape();
      OP_TILING_CHECK(inithShape.GetDimNum() != 3,
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn get init_h shape dim is not 3, please check."),
                      return false);
      OP_TILING_CHECK(initcShape.GetDimNum() != 3,
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn get init_c shape dim is not 3, please check."),
                      return false);
      OP_TILING_CHECK(inithShape.GetDim(1) != outputShape.GetDim(1),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn init_h output batch is not equal, please check."),
                      return false);
      OP_TILING_CHECK(initcShape.GetDim(1) != outputShape.GetDim(1),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn init_c output batch is not equal, please check."),
                      return false);
      OP_TILING_CHECK(inithShape.GetDim(2) != outputShape.GetDim(2),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn init_h output hidden is not equal, please check."),
                      return false);
      OP_TILING_CHECK(initcShape.GetDim(2) != outputShape.GetDim(2),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn init_c output hidden is not equal, please check."),
                      return false);
    }

    if (seqInput != nullptr)
    {
      auto seqShape = seqInput->GetStorageShape();
      OP_TILING_CHECK(seqShape.GetDimNum() != 3,
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn get seq shape dim is not 3, please check."),
                      return false);
      OP_TILING_CHECK(seqShape.GetDim(0) != outputShape.GetDim(0),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn seq and x T is not equal, please check."),
                      return false);
      OP_TILING_CHECK(seqShape.GetDim(1) != outputShape.GetDim(1),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn seq output batch is not equal, please check."),
                      return false);
      OP_TILING_CHECK(seqShape.GetDim(2) != outputShape.GetDim(2),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                      "Dynamicrnn seq output hidden is not equal, please check."),
                      return false);
    }
    return true;
  }

  bool DynamicRNNTiling::CheckAttrTiling(gert::TilingContext *context)
  {
    // get attr
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
    const int *numProj = attrs->GetAttrPointer<int>(6);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, numProj, false);
    const bool *timeMajor = attrs->GetAttrPointer<bool>(7);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, timeMajor, false);
    const char *activation = attrs->GetAttrPointer<char>(8);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, activation, false);
    const float *forgetBias = attrs->GetAttrPointer<float>(9);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, forgetBias, false);
    const char *gateOrder = attrs->GetAttrPointer<char>(10);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, gateOrder, false);
    const bool *isTraining = attrs->GetAttrPointer<bool>(11);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, isTraining, false);

    OP_TILING_CHECK(*numProj != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr num_proj only support 0, please check."),
                    return false);
    OP_TILING_CHECK(!(*timeMajor),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr time_major only support true, please check."),
                    return false);

    std::vector<std::string> supportGateOrder = {"ifjo", "ijfo"};
    OP_TILING_CHECK(std::find(supportGateOrder.begin(), supportGateOrder.end(), gateOrder) == supportGateOrder.end(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr gate_order is not support, please check."),
                    return false);

    OP_TILING_CHECK(strcmp(activation, "tanh") != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr activation is not support, please check."),
                    return false);

    return true;
  }

  bool DynamicRNNTiling::CheckAttrOps(gert::TilingContext *context)
  {
    // get attr
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
    const char *cellType = attrs->GetAttrPointer<char>(0);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellType, false);
    const char *direction = attrs->GetAttrPointer<char>(1);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, direction, false);
    const int *cellDepth = attrs->GetAttrPointer<int>(2);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellDepth, false);
    const bool *usePeephole = attrs->GetAttrPointer<bool>(3);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, usePeephole, false);
    const float *keepProb = attrs->GetAttrPointer<float>(4);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, keepProb, false);
    const float *cellClip = attrs->GetAttrPointer<float>(5);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellClip, false);

    OP_TILING_CHECK(*cellDepth != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr cell_depth only support 1, please check."),
                    return false);
    OP_TILING_CHECK(*usePeephole,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr use_peephole only support false, please check."),
                    return false);
    std::vector<std::string> supportDirection = {"UNIDIRECTIONAL", "REDIRECTIONAL"};
    OP_TILING_CHECK(std::find(supportDirection.begin(), supportDirection.end(), direction) == supportDirection.end(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr direction is not support, please check."),
                    return false);
    OP_TILING_CHECK(strcmp(cellType, "LSTM") != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Dynamicrnn attr cell_type is not support, please check."),
                    return false);
    return true;
  }

  ge::graphStatus TilingForDynamicRNN(gert::TilingContext *context)
  {
    // set sync op batchmode
    context->SetScheduleMode(SCHEDULE_MODE);
    // 910B/C AscendC
    bool supportL0c2out = false;
    DynamicRNNTiling rnnTiling;
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    rnnTiling.GetAICoreIntrinsicDtype(*platformInfo, "Intrinsic_fix_pipe_l0c2out", supportL0c2out);
    if (supportL0c2out)
    {
      OP_TILING_CHECK(rnnTiling.TilingWithAscendC(context) != ge::GRAPH_SUCCESS,
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "tiling with ascendc have error."),
                      return ge::GRAPH_FAILED);
      return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus TilingPrepareForDynamicRNN(gert::TilingParseContext *context)
  {
    // 910B/C AscendC
    bool supportL0c2out = false;
    DynamicRNNTiling rnnTiling;
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    return ge::GRAPH_SUCCESS;
  }

  IMPL_OP_OPTILING(DynamicRNN).Tiling(TilingForDynamicRNN).TilingParse<DynamicRnnCompileInfo>(TilingPrepareForDynamicRNN);
} // namespace optiling

namespace ge
{
  namespace ops
  {
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

    static ge::graphStatus InferShape4DynamicRNN(gert::InferShapeContext *context)
    {
      OP_LOGD(context->GetNodeName(), "InferShape4DynamicRNN start");
      auto x_shape = context->GetInputShape(0);
      OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
      auto w_shape = context->GetInputShape(1);
      OPS_CHECK_NULL_WITH_CONTEXT(context, w_shape);

      auto y_shape = context->GetOutputShape(0);
      OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
      auto outputh_shape = context->GetOutputShape(1);
      OPS_CHECK_NULL_WITH_CONTEXT(context, outputh_shape);
      auto outputc_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_C);
      OPS_CHECK_NULL_WITH_CONTEXT(context, outputc_shape);
      auto i_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_I);
      OPS_CHECK_NULL_WITH_CONTEXT(context, i_shape);
      auto j_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_J);
      OPS_CHECK_NULL_WITH_CONTEXT(context, j_shape);
      auto f_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_F);
      OPS_CHECK_NULL_WITH_CONTEXT(context, f_shape);
      auto o_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_O);
      OPS_CHECK_NULL_WITH_CONTEXT(context, o_shape);
      auto tanhc_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_TANHC);
      OPS_CHECK_NULL_WITH_CONTEXT(context, tanhc_shape);

      if (x_shape->GetDimNum() != X_SHAPE_SIZE_LIMIT)
      {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "The input x shape dim is not 3, please check!");
        return ge::GRAPH_FAILED;
      }

      int64_t num_step = x_shape->GetDim(0);
      int64_t batch_size = x_shape->GetDim(1);
      int64_t hidden_size = 0;
      if (w_shape->GetDim(1) == UNKNOWN_DIM_VALUE)
      {
        hidden_size = UNKNOWN_DIM_VALUE;
      }
      else
      {
        hidden_size = w_shape->GetDim(1) / CONSTANT_FOUR;
      }

      auto attrs = context->GetAttrs();
      OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
      const char *direction = attrs->GetAttrPointer<char>(1);
      OPS_CHECK_NULL_WITH_CONTEXT(context, direction);

      if (strcmp(direction, "BIDIRECTIONAL") == 0)
      {
        *y_shape = {num_step, batch_size, CONSTANT_TWO * hidden_size};
        *outputh_shape = {num_step, batch_size, CONSTANT_TWO * hidden_size};
        *outputc_shape = {num_step, batch_size, CONSTANT_TWO * hidden_size};
        *i_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
        *j_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
        *f_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
        *o_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
        *tanhc_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
      }
      else
      {
        *y_shape = {num_step, batch_size, hidden_size};
        *outputh_shape = {num_step, batch_size, hidden_size};
        *outputc_shape = {num_step, batch_size, hidden_size};
        *i_shape = {num_step, batch_size, hidden_size};
        *j_shape = {num_step, batch_size, hidden_size};
        *f_shape = {num_step, batch_size, hidden_size};
        *o_shape = {num_step, batch_size, hidden_size};
        *tanhc_shape = {num_step, batch_size, hidden_size};
      }
      return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType4DynamicRNN(gert::InferDataTypeContext *context)
    {
      OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNN start");
      auto input_x_dtype = context->GetInputDataType(0);
      auto input_b_dtype = context->GetInputDataType(CONSTANT_TWO);

      OP_CHECK(context->SetOutputDataType(0, input_b_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType y Fail"), return ge::GRAPH_FAILED);
      OP_CHECK(context->SetOutputDataType(1, input_x_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType output_h Fail"), return ge::GRAPH_FAILED);
      OP_CHECK(context->SetOutputDataType(RNN_OUTPUT_INDEX_C, input_b_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType output_c Fail"), return ge::GRAPH_FAILED);
      OP_CHECK(context->SetOutputDataType(RNN_OUTPUT_INDEX_I, input_b_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType i Fail"), return ge::GRAPH_FAILED);
      OP_CHECK(context->SetOutputDataType(RNN_OUTPUT_INDEX_J, input_b_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType j Fail"), return ge::GRAPH_FAILED);
      OP_CHECK(context->SetOutputDataType(RNN_OUTPUT_INDEX_F, input_b_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType f Fail"), return ge::GRAPH_FAILED);
      OP_CHECK(context->SetOutputDataType(RNN_OUTPUT_INDEX_O, input_b_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType o Fail"), return ge::GRAPH_FAILED);
      OP_CHECK(context->SetOutputDataType(RNN_OUTPUT_INDEX_TANHC, input_b_dtype) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "SetOutputDataType tanhc Fail"), return ge::GRAPH_FAILED);

      OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNN end");
      return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_INFERSHAPE(DynamicRNN).InferShape(InferShape4DynamicRNN).InferDataType(InferDataType4DynamicRNN);
  } // namespace ops
} // namespace ge

// Ascend info
namespace ops
{
  class DynamicRNN : public OpDef
  {
  public:
    explicit DynamicRNN(const char *name) : OpDef(name)
    {
      this->Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("w")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("b")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("seq_length")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("init_h")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("init_c")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("wci")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("wcf")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("wco")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("mask")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_UINT8, ge::DT_UINT8})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("output_h")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("output_c")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("i")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("j")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("f")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("o")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("tanhc")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Attr("cell_type")
          .AttrType(REQUIRED)
          .String("LSTM");
      this->Attr("direction")
          .AttrType(REQUIRED)
          .String("UNIDIRECTIONAL");
      this->Attr("cell_depth")
          .AttrType(REQUIRED)
          .Int(1);
      this->Attr("use_peephole")
          .AttrType(REQUIRED)
          .Bool(false);
      this->Attr("keep_prob")
          .AttrType(REQUIRED)
          .Float(1.0);
      this->Attr("cell_clip")
          .AttrType(REQUIRED)
          .Float(-1.0);
      this->Attr("num_proj")
          .AttrType(REQUIRED)
          .Int(0);
      this->Attr("time_major")
          .AttrType(REQUIRED)
          .Bool(true);
      this->Attr("activation")
          .AttrType(REQUIRED)
          .String("tanh");
      this->Attr("forget_bias")
          .AttrType(REQUIRED)
          .Float(0.0);
      this->Attr("gate_order")
          .AttrType(REQUIRED)
          .String("ifjo");
      this->Attr("is_training")
          .AttrType(REQUIRED)
          .Bool(true);
      this->AICore().AddConfig("ascend910b");
    }
  };

  OP_ADD(DynamicRNN);
} // namespace ops