/* 
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

/**
 * @file addcmul.cpp
 */
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "diag_v2.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
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

#define SCALAR_UNEQUAL_SIZE16 101
#define SCALAR_EQUAL_SIZE16 102
#define ASSIST_UNEQUAL_SIZE16 103
#define ASSIST_EQUAL_SIZE16 104

static const int64_t SCALAR_THRESHOLD_NUM = 64;
static const int64_t SCALAR_MAX_WIDTH = 1024;
static const int64_t LEAST_NUM_PER_CORE = 64;
static const int64_t BLOCK_SIZE = 32;

enum class DiagV2TilingKey : int64_t {
  ASSIST_SIZE_1 = 2101,
  ASSIST_SIZE_2 = 2102,
  ASSIST_SIZE_4 = 2103,
  ASSIST_SIZE_8 = 2104,
  ASSIST_SIZE_16 = 2105,
  SCALAR_SIZE_1 = 2401,
  SCALAR_SIZE_2 = 2402,
  SCALAR_SIZE_4 = 2403,
  SCALAR_SIZE_8 = 2404
};

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
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

namespace optiling
{
static const int32_t SIZE_1 = 1;
static const int32_t SIZE_2 = 2;
static const int32_t SIZE_4 = 4;
static const int32_t SIZE_8 = 8;
static const int32_t SIZE_16 = 16;
static const int32_t LENGTH_1024 = 1024;
static const int32_t LENGTH_128 = 128;
static const int32_t LENGTH_64 = 64;
static const int32_t LENGTH_32 = 32;

inline static ge::graphStatus DiagV2SetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData) {
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcAuxMatrixTiling(const int32_t typeSize, DiagV2TilingData& tilingData) {
    switch (typeSize)
    {
    case SIZE_1:
        tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_1));
        tilingData.set_matrixRowLength(LENGTH_128);
        break;
    case SIZE_2:
        tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_2));
        tilingData.set_matrixRowLength(LENGTH_128);
        break;
    case SIZE_4:
        tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_4));
        tilingData.set_matrixRowLength(LENGTH_64);
        break;
    case SIZE_8:
        tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_8));
        tilingData.set_matrixRowLength(LENGTH_64);
        break;
    case SIZE_16:
        tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_16));
        tilingData.set_matrixRowLength(LENGTH_32);
        break;
    default:
        break;
    }

    return ge::GRAPH_SUCCESS;
}

static inline int64_t CalcLeastNumPerCore(const int32_t typeSize, const gert::TilingContext *context) {
    OP_TILING_CHECK((typeSize <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Tiling4DiagV2 typeSize is invalid %d, please check.",
                                                    typeSize),
                    return -1);

    int64_t leastNumPerCore = BLOCK_SIZE / typeSize;
    return (leastNumPerCore > 0) ? leastNumPerCore : 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    OP_LOGD(context->GetNodeName(), "Tiling4DiagV2 running begin");
    auto inputShapePtr = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    auto inputShape = inputShapePtr->GetStorageShape();
    OP_TILING_CHECK((inputShape.GetDimNum() != SIZE_2),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Tiling4DiagV2 get input shape dim(=%zu) is\
                                                    not 2, please check.", inputShape.GetDimNum()),
                    return ge::GRAPH_FAILED);

    DiagV2TilingData tilingData;
    tilingData.set_xHeight(inputShape.GetDim(0));
    tilingData.set_xWidth(inputShape.GetDim(1));
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, diagonalPtr);
    const int64_t diagonal = *diagonalPtr;
    OP_TILING_CHECK((diagonal >= 0 && diagonal > tilingData.get_xWidth())
                    || (diagonal < 0 && std::abs(diagonal) > tilingData.get_xHeight()),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Tiling4DiagV2 attr diagonal(=%ld) is wrong,\
                                                    please check. w=%ld, h=%ld", diagonal,
                                                    tilingData.get_xWidth(), tilingData.get_xHeight()),
                    return ge::GRAPH_FAILED);
    tilingData.set_gmOffset(diagonal > 0 ? diagonal : std::abs(diagonal) * tilingData.get_xWidth());
    tilingData.set_numOut((diagonal >= 0)
                        ? std::min((tilingData.get_xWidth() - diagonal), tilingData.get_xHeight())
                        : std::min((tilingData.get_xHeight() + diagonal), tilingData.get_xWidth()));
    auto inputDesc = context->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();
    const int32_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t leastNumPerCore = CalcLeastNumPerCore(typeSize, context);
    OP_TILING_CHECK((leastNumPerCore <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "Tiling4DiagV2 leastNumPerCore is invalid, please check."),
                    return ge::GRAPH_FAILED);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform = 0;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    int64_t tmpRealCoreNum = tilingData.get_numOut() < coreNum ? tilingData.get_numOut() : coreNum;
    int64_t tmpCorePerNum = CeilDiv(tilingData.get_numOut(), tmpRealCoreNum);
    tilingData.set_numPerCore(CeilDiv(tmpCorePerNum, leastNumPerCore) * leastNumPerCore);
    tilingData.set_realCoreNum(CeilDiv(tilingData.get_numOut(), tilingData.get_numPerCore()));
    tilingData.set_tailNum(tilingData.get_numOut()
                            - (tilingData.get_realCoreNum() - 1) * tilingData.get_numPerCore());
    OP_TILING_CHECK(CalcAuxMatrixTiling(typeSize, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CalcAuxMatrixTiling fail."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(DiagV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "DiagV2SetTilingData set tiling data fail."),
                    return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    OP_LOGD(context->GetNodeName(), "tilingData is xWidth:%ld, xHeight:%ld, gmOffset:%ld,\
                              numOut:%ld, realCoreNum:%ld, numPerCore:%ld, tailNum:%ld,\
                              tilingKey:%ld, matrixRowLength:%ld",
                              tilingData.get_xWidth(), tilingData.get_xHeight(), tilingData.get_gmOffset(),
                              tilingData.get_numOut(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
                              tilingData.get_tailNum(), tilingData.get_tilingKey(), tilingData.get_matrixRowLength());

    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class DiagV2 : public OpDef {
public:
    explicit DiagV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
                        ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,
                        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE,
                        ge::DT_BOOL, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
                        ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,
                        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE,
                        ge::DT_BOOL, ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                        ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("diagonal")
            .AttrType(OPTIONAL)
            .Int(0);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend310p");

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
          .DynamicRankSupportFlag(true)
          .DynamicShapeSupportFlag(true);
    config.Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
                    ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,
                    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE,
                    ge::DT_BOOL, ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND});
    config.Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
                    ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,
                    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE,
                    ge::DT_BOOL, ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND});
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b", config);
    this->AICore().AddConfig("ascend910_93", config);
    }
};
OP_ADD(DiagV2);
}

namespace ops {
static constexpr size_t DiagV2_IN_X_IDX = 0;
static constexpr size_t DiagV2_OUT_Y_IDX = 0;
static constexpr int64_t INT_DATA_2 = 2;
constexpr size_t kAxisAttrIdx = 0U;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t NEG_TWO = -2;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;

static ge::graphStatus Infershape4DiagV2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do DiagV2Infershape.");
  // 获取输入值shape
  const gert::Shape* input_x_shape = context->GetInputShape(DiagV2_IN_X_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);

  size_t x_dim_num = input_x_shape->GetDimNum();
  if (x_dim_num > DIM_TWO) {
    OP_LOGD(context->GetNodeName(), "The dim number of input should be less than 3.");
    return ge::GRAPH_FAILED;
  }

  // 获取属性值
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  int64_t diagonal = *(attrs->GetInt(kAxisAttrIdx));

  // 获取输出值shape
  gert::Shape* output_y_shape = context->GetOutputShape(DiagV2_OUT_Y_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

  if (x_dim_num < DIM_TWO) {  // 1D->2D
    output_y_shape->SetDimNum(0);
    // 获取元素element的个数
    auto total_element_num = input_x_shape->GetShapeSize();
    for (int64_t k = 0; k < INT_DATA_2; k++) {
      output_y_shape->AppendDim(total_element_num + std::abs(diagonal));
    }
  } else {   // 2D->1D
    output_y_shape->SetDimNum(1);
    if (diagonal > 0) {
      output_y_shape->SetDim(0, std::min(input_x_shape->GetDim(0), input_x_shape->GetDim(1) - diagonal));
      // 判断偏移量是否超出上限
      if (diagonal >= input_x_shape->GetDim(1)) {
        output_y_shape->SetDim(0, 0);
      }
    } else {
      output_y_shape->SetDim(0, std::min(input_x_shape->GetDim(0) + diagonal, input_x_shape->GetDim(1)));
      if (-diagonal >= input_x_shape->GetDim(0)) {
        output_y_shape->SetDim(0, 0);
      }
    }
  }

  OP_LOGD(context->GetNodeName(), "End to do DiagV2Infershape.");

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DiagV2(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4DiagV2");
  context->SetOutputDataType(DiagV2_OUT_Y_IDX, context->GetInputDataType(DiagV2_IN_X_IDX));
  OP_LOGD(context->GetNodeName(), "End to do InferDataType4DiagV2");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DiagV2).InferShape(Infershape4DiagV2).InferDataType(InferDataType4DiagV2);
}