/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "lin_space_tiling.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

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

static const size_t INPUT_IDX_START = 0;
static const size_t INPUT_IDX_STOP = 1;
static const size_t INPUT_IDX_NUM = 2;
static const size_t POWER_BASE_NUM = 2;
static const int32_t INT16_BITS_NUM = 16;
static const int32_t BLOCK_SIZE = 32;
static const int64_t MATRIX_SIZE = 256;
static const int32_t OUT_SIZE = 16 * 1024;
static const int32_t WORK_SPACE_SIZE = 32;

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

inline static int64_t GetBaseNum(int64_t value) {
  int64_t valueNum = 0;
  if (value == 0) {
    return valueNum;
  }
  for (int64_t idex = 1; idex <= value; idex *= POWER_BASE_NUM) {
    valueNum++;
  }
  return valueNum;
}

inline static ge::graphStatus LinSpaceSetTilingData(gert::TilingContext* context, LinSpaceTilingData& tilingData) {
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

inline static int64_t CalcAlignNumPerCore(const ge::DataType outputDtype, const gert::TilingContext* context) {
  int32_t typeSize = ge::GetSizeByDataType(outputDtype);
  OP_TILING_CHECK((typeSize <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                  "Tiling4LinSpace typeSize is invalid %d, please check.", typeSize),
                  return -1);
  return BLOCK_SIZE / typeSize;
}

inline static int64_t CalcMaxOutNum(const ge::DataType outDataType, const gert::TilingContext* context) {
  int32_t outTypeSize = ge::GetSizeByDataType(outDataType);
  OP_TILING_CHECK((outTypeSize <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                  "Tiling4LinSpace outTypeSize is invalid %d, please check.", outTypeSize),
                  return -1);
  return OUT_SIZE / outTypeSize;
}

static bool InputTypeIsInvalid(const gert::TilingContext* context) {
  auto startDes = context->GetInputDesc(INPUT_IDX_START);
  OPS_CHECK_NULL_WITH_CONTEXT(context, startDes);
  auto dStart = startDes->GetDataType();
  auto stopDsc = context->GetInputDesc(INPUT_IDX_STOP);
  OPS_CHECK_NULL_WITH_CONTEXT(context, stopDsc);
  auto dStop = stopDsc->GetDataType();
  auto numDsc = context->GetInputDesc(INPUT_IDX_NUM);
  OPS_CHECK_NULL_WITH_CONTEXT(context, numDsc);
  auto dNum = numDsc->GetDataType();
  // any input dtype is neither int32 nor float, invalid
  return ((dStart != ge::DT_INT32) && (dStart != ge::DT_FLOAT) && (dStart != ge::DT_FLOAT16) &&
          (dStart != ge::DT_BF16) && (dStart != ge::DT_INT8) && (dStart != ge::DT_UINT8) && (dStart != ge::DT_INT16)) ||
         ((dStop != ge::DT_INT32) && (dStop != ge::DT_FLOAT) && (dStop != ge::DT_FLOAT16) && (dStop != ge::DT_BF16) &&
          (dStop != ge::DT_INT8) && (dStop != ge::DT_UINT8) && (dStop != ge::DT_INT16)) ||
         ((dNum != ge::DT_INT32) && (dNum != ge::DT_INT64));
}

template <typename T>
static ge::graphStatus LinSpaceGetConstValue(gert::TilingContext* context, const gert::Tensor* tensor, T& value, ge::DataType dataType) {
  if (dataType == ge::DT_INT64) {
    const int64_t* const_data_ptr = tensor->GetData<int64_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    value = static_cast<T>(*const_data_ptr);
    OP_LOGD(context->GetNodeName(), "LinSpace get const value:%ld", *const_data_ptr);
  } else if (dataType == ge::DT_INT32) {
    const int32_t* const_data_ptr = tensor->GetData<int32_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    value = static_cast<T>(*const_data_ptr);
    OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
  } else if (dataType == ge::DT_FLOAT) {
    const float* const_data_ptr = tensor->GetData<float>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    value = static_cast<T>(*const_data_ptr);
    OP_LOGD(context->GetNodeName(), "LinSpace get const value:%f", *const_data_ptr);
  } else if (dataType == ge::DT_FLOAT16) {
    const int16_t* const_data_ptr = tensor->GetData<int16_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    int32_t f32hex = (static_cast<int32_t>(*const_data_ptr)) << INT16_BITS_NUM;
    value = static_cast<T>((reinterpret_cast<float&>(f32hex)));
  } else if (dataType == ge::DT_INT16) {
    const int16_t* const_data_ptr = tensor->GetData<int16_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    value = static_cast<T>(*const_data_ptr);
    OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
  } else if (dataType == ge::DT_BF16) {
    const int16_t* const_data_ptr = tensor->GetData<int16_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    int32_t f32hex = (static_cast<int32_t>(*const_data_ptr)) << INT16_BITS_NUM;
    value = static_cast<T>((reinterpret_cast<float&>(f32hex)));
  } else if (dataType == ge::DT_INT8) {
    const int8_t* const_data_ptr = tensor->GetData<int8_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    value = static_cast<T>(*const_data_ptr);
    OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
  } else if (dataType == ge::DT_UINT8) {
    const uint8_t* const_data_ptr = tensor->GetData<uint8_t>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, const_data_ptr);
    value = static_cast<T>(*const_data_ptr);
    OP_LOGD(context->GetNodeName(), "LinSpace get const value:%d", *const_data_ptr);
  } else {
    // do nothing, impossible
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetConstNum(gert::TilingContext* context, const gert::Tensor* tensor_num, int64_t& num, int64_t input_index) {
  auto numDesc = context->GetInputDesc(input_index);
  OPS_CHECK_NULL_WITH_CONTEXT(context, numDesc);
  const ge::DataType dataType = numDesc->GetDataType();
  switch (dataType) {
    case ge::DT_INT32: {
      int32_t num_32(0);
      OP_TILING_CHECK(LinSpaceGetConstValue<int32_t>(context, tensor_num, num_32, dataType) != ge::GRAPH_SUCCESS,
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get num const value fail."),
                      return ge::GRAPH_FAILED);
      num = (int64_t)num_32;
      break;
    }
    case ge::DT_INT64:
      OP_TILING_CHECK(LinSpaceGetConstValue<int64_t>(context, tensor_num, num, dataType) != ge::GRAPH_SUCCESS,
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get const value fail."),
                      return ge::GRAPH_FAILED);
      break;
    default:
      OP_LOGE(context->GetNodeName(), "get const num fail!");
      return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetConstStartOrStop(gert::TilingContext* context, const gert::Tensor* tensor_index,
                                           float& index, int64_t input_index) {
  auto inputDesc = context->GetInputDesc(input_index);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
  const ge::DataType dataType = inputDesc->GetDataType();
  switch (dataType) {
    case ge::DT_INT32: {
      int32_t index_32(0);
      LinSpaceGetConstValue<int32_t>(context, tensor_index, index_32, dataType);
      index = static_cast<float>(index_32);
      break;
    }
    case ge::DT_FLOAT:
    case ge::DT_BF16:
    case ge::DT_FLOAT16: {
      LinSpaceGetConstValue<float>(context, tensor_index, index, dataType);
      break;
    }
    case ge::DT_INT16: {
      int16_t index_16(0);
      LinSpaceGetConstValue<int16_t>(context, tensor_index, index_16, dataType);
      index = static_cast<float>(index_16);
      break;
    }
    case ge::DT_INT8: {
      int8_t index_8(0);
      LinSpaceGetConstValue<int8_t>(context, tensor_index, index_8, dataType);
      index = static_cast<float>(index_8);
      break;
    }
    case ge::DT_UINT8: {
      uint8_t index_8(0);
      LinSpaceGetConstValue<uint8_t>(context, tensor_index, index_8, dataType);
      index = static_cast<float>(index_8);
      break;
    }
    default:
      OP_LOGE(context->GetNodeName(), "start or stop dataType is not support!");
      return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetLoopNumForLinSpace(const gert::TilingContext* context, LinSpaceTilingData& tilingData,
                                             const ge::DataType outDataType) {
  int64_t maxOutNum = CalcMaxOutNum(outDataType, context);
  int64_t matrixLen = tilingData.get_numPerCore() <= MATRIX_SIZE ? tilingData.get_numPerCore() : MATRIX_SIZE;
  tilingData.set_matrixLen(matrixLen);
  tilingData.set_outerLoopNum(CeilDiv(tilingData.get_numPerCore(), maxOutNum));
  tilingData.set_outerLoopNumTail(tilingData.get_numPerCore() % maxOutNum);
  tilingData.set_outerTailLoopNum(CeilDiv(tilingData.get_tailNum(), maxOutNum));
  tilingData.set_outerTailLoopNumTail(tilingData.get_tailNum() % maxOutNum);

  if (tilingData.get_numPerCore() <= MATRIX_SIZE) {
    tilingData.set_innerLoopNum(0);
    tilingData.set_innerLoopTail(0);
    tilingData.set_innerTailLoopNum(0);
    tilingData.set_innerTailLoopTail(0);
  } else {
    tilingData.set_innerLoopNum(tilingData.get_numPerCore() / MATRIX_SIZE / POWER_BASE_NUM);
    tilingData.set_innerLoopTail(tilingData.get_numPerCore() -
                                 (MATRIX_SIZE << GetBaseNum(tilingData.get_innerLoopNum())));
    tilingData.set_innerTailLoopNum(tilingData.get_tailNum() / MATRIX_SIZE / POWER_BASE_NUM);
    tilingData.set_innerTailLoopTail(
        tilingData.get_tailNum() > MATRIX_SIZE
            ? (tilingData.get_tailNum() - (MATRIX_SIZE << GetBaseNum(tilingData.get_innerTailLoopNum())))
            : 0);
  }
  OP_LOGD(context->GetNodeName(),
          "tilingData is matrixLen:%ld, outerLoopNum:%ld, outerLoopNumTail:%ld, \
          outerTailLoopNum:%ld, outerTailLoopNumTail:%ld, innerLoopNum:%ld, innerLoopTail:%ld, \
          innerTailLoopNum:%ld, innerTailLoopTail:%ld",
          tilingData.get_matrixLen(), tilingData.get_outerLoopNum(), tilingData.get_outerLoopNumTail(),
          tilingData.get_outerTailLoopNum(), tilingData.get_outerTailLoopNumTail(), tilingData.get_innerLoopNum(),
          tilingData.get_innerLoopTail(), tilingData.get_innerLoopTail(), tilingData.get_innerTailLoopNum());
  return ge::GRAPH_SUCCESS;
}


static ge::graphStatus SetTilingTilingKeyOneCore(const gert::TilingContext *context, LinSpaceTilingData &tilingData,
                                                 const ge::DataType outDataType)
{
  switch (outDataType) {
    case ge::DT_FLOAT:
      tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_103));
      break;
    case ge::DT_FLOAT16:
      tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_203));
      break;
    case ge::DT_INT8:
      tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_303));
      break;
    case ge::DT_UINT8:
      tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_403));
      break;
    case ge::DT_INT16:
      tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_503));
      break;
    case ge::DT_INT32:
      tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_603));
      break;
    case ge::DT_BF16:
      tilingData.set_tilingKey(static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_703));
      break;
    default:
      OP_LOGE(context->GetNodeName(), "set tilingKet fail!");
      return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingTilingKeyForLinSpace(gert::TilingContext *context, LinSpaceTilingData &tilingData,
                                                     const ge::DataType outDataType)
{
  int64_t maxOutNum = CalcMaxOutNum(outDataType, context);
  if (tilingData.get_realCoreNum() == 1) {
    OP_TILING_CHECK(SetTilingTilingKeyOneCore(context, tilingData, outDataType) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tilingKey fail."), return ge::GRAPH_FAILED);
  } else {
    switch (outDataType) {
      case ge::DT_FLOAT:
          tilingData.set_tilingKey(tilingData.get_numPerCore() <= maxOutNum ?
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_101) :
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_102));
          break;
      case ge::DT_FLOAT16:
          tilingData.set_tilingKey(tilingData.get_numPerCore() <= maxOutNum ?
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_201) :
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_202));
          break;
      case ge::DT_INT8:
          tilingData.set_tilingKey(tilingData.get_numPerCore() <= maxOutNum ?
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_301) :
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_302));
          break;
      case ge::DT_UINT8:
          tilingData.set_tilingKey(tilingData.get_numPerCore() <= maxOutNum ?
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_401) :
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_402));
          break;
      case ge::DT_INT16:
          tilingData.set_tilingKey(tilingData.get_numPerCore() <= maxOutNum ?
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_501) :
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_502));
          break;
      case ge::DT_INT32:
          tilingData.set_tilingKey(tilingData.get_numPerCore() <= maxOutNum ?
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_601) :
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_602));
          break;
      case ge::DT_BF16:
          tilingData.set_tilingKey(tilingData.get_numPerCore() <= maxOutNum ?
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_701) :
                                   static_cast<int64_t>(LinSpaceTilingKey::TILINGKEY_702));
          break;
      default:
          OP_LOGE(context->GetNodeName(), "set tilingKet fail!");
          return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingDataForLinSpace(gert::TilingContext* context, LinSpaceTilingData& tilingData,
                                                const ge::DataType outDataType) {
  OP_TILING_CHECK(SetTilingTilingKeyForLinSpace(context, tilingData, outDataType) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set loopNum fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(SetLoopNumForLinSpace(context, tilingData, outDataType) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set loopNum fail."),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK(
      LinSpaceSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "LinSpaceSetTilingData set tiling data fail."),
      return ge::GRAPH_FAILED);
  context->SetBlockDim(tilingData.get_realCoreNum());
  context->SetTilingKey(tilingData.get_tilingKey());

  size_t* workspaces = context->GetWorkspaceSizes(1);
  workspaces[0] = WORK_SPACE_SIZE;

  OP_LOGD(context->GetNodeName(), "tilingData is tilingKey:%ld, realCoreNum:%ld, numPerCore:%ld, tailNum:%ld",
          tilingData.get_tilingKey(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
          tilingData.get_tailNum());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingBasicInfo(const gert::TilingContext* context, LinSpaceTilingData& tilingData,
                                          const float& start, const float& stop, const int64_t& num) {
  tilingData.set_start(start);
  tilingData.set_stop(stop);
  tilingData.set_num(num);
  if (num > 1) {
    tilingData.set_scalar((stop - start) / (num - 1));
  } else {
    tilingData.set_scalar(0);
  }
  OP_LOGD(context->GetNodeName(), "tilingData is start:%f, stop:%f, num:%ld, scalar:%f", tilingData.get_start(),
          tilingData.get_stop(), tilingData.get_num(), tilingData.get_scalar());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4LinSpace(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "Tiling4LinSpace enter.");
  auto tensor_start = context->GetInputTensor(INPUT_IDX_START);
  OPS_CHECK_NULL_WITH_CONTEXT(context, tensor_start);
  auto tensor_stop = context->GetInputTensor(INPUT_IDX_STOP);
  OPS_CHECK_NULL_WITH_CONTEXT(context, tensor_stop);
  auto tensor_num = context->GetInputTensor(INPUT_IDX_NUM);
  OPS_CHECK_NULL_WITH_CONTEXT(context, tensor_num);
  OP_TILING_CHECK(InputTypeIsInvalid(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input dtype is invalid."),
                  return ge::GRAPH_FAILED);

  LinSpaceTilingData tilingData;
  float start(0);
  float stop(0);
  int64_t num(0);
  OP_TILING_CHECK(GetConstNum(context, tensor_num, num, INPUT_IDX_NUM) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tilingData num fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(GetConstStartOrStop(context, tensor_start, start, INPUT_IDX_START) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tilingData start fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(GetConstStartOrStop(context, tensor_stop, stop, INPUT_IDX_STOP) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tilingData stop fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(SetTilingBasicInfo(context, tilingData, start, stop, num) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set basic info fail."),
                  return ge::GRAPH_FAILED);

  auto outputDesc = context->GetOutputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
  ge::DataType outputDtype = outputDesc->GetDataType();

  int64_t alignNumPerCore = CalcAlignNumPerCore(outputDtype, context);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  
  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  int64_t tmpRealCoreNum = tilingData.get_num() < coreNum ? tilingData.get_num() : coreNum;
  int64_t tmpNumPerCore = CeilDiv(tilingData.get_num(), tmpRealCoreNum);

  tilingData.set_numPerCore(CeilDiv(tmpNumPerCore, alignNumPerCore) * alignNumPerCore);
  tilingData.set_realCoreNum(CeilDiv(tilingData.get_num(), tilingData.get_numPerCore()));
  int64_t tailNum = tilingData.get_num() - (tilingData.get_realCoreNum() - 1) * tilingData.get_numPerCore();
  tilingData.set_tailNum(tailNum);
  OP_TILING_CHECK(SetTilingDataForLinSpace(context, tilingData, outputDtype) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set TilingKey fail."),
                  return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "Tiling4LinSpace exit.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LinSpace)
.Tiling(Tiling4LinSpace)
.TilingInputsDataDependency({0, 1, 2});
// .TilingParse<LinSpaceCompileInfo>(TilingPrepare4LinSpace);
}  // namespace optiling

namespace ge {
  template <typename T>
  static void GetLinSpaceConstValue(const Operator& op, const Tensor& const_tensor, std::vector<int64_t>& const_data) {
    size_t size = 0;
  
    T* const_data_ptr = (T*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(T);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int64_t)((*(const_data_ptr + i))));
      AscendString op_name;
      (void)op.GetName(op_name);
      OP_LOGI(op_name.GetString(), "const data float fusion pass ====== %ld", (int64_t)(*(const_data_ptr + i)));
    }
  }
}