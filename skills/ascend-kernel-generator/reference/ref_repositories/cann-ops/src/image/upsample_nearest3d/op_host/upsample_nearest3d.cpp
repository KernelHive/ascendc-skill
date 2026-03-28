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
 * \file upsample_nearest3d.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "upsample_nearest3d_tiling.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(op_name, ...)   std::printf(op_name, ##__VA_ARGS__)
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)        \
  do {                                                               \
    std::printf("op[%s], %s", op_name, err_msg);                     \
  } while (0)
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
constexpr int64_t BEST_PERFORMANCE_SIZE_1 = 192;
constexpr int64_t BEST_PERFORMANCE_SIZE_2 = 768;
constexpr int64_t BEST_PERFORMANCE_SIZE_3 = 1536;
constexpr int64_t BEST_PERFORMANCE_SIZE_4 = 2048;

constexpr float BEST_PERFORMANCE_SCALE_1 = 100.0f;
constexpr float BEST_PERFORMANCE_SCALE_2 = 24.0f;
constexpr float BEST_PERFORMANCE_SCALE_3 = 10.0f;
constexpr float BEST_PERFORMANCE_SCALE_4 = 6.0f;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

constexpr uint8_t RESERVED_LENGTH = 4;

constexpr uint8_t HALF_TYPE = 1;
constexpr uint8_t FLOAT_TYPE = 2;
constexpr uint8_t BFLOAT_TYPE = 3;

constexpr uint8_t BATCH_DIM = 2;
constexpr uint8_t DIM = 3;
constexpr uint8_t D_INDEX = 0;
constexpr uint8_t H_INDEX = 1;
constexpr uint8_t W_INDEX = 2;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;

class UpsampleNearest3dTiling {
public:
    explicit UpsampleNearest3dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    void SetScale();
    inline float ComputeScaleValue(int64_t inSize, int64_t outSize, const float *scale) const;
    void GetShapes();
    void GetSlideSize();
    uint8_t GetDataTypeVal() const;
    void GetNeedCoreNum(int64_t coreNumPlatform);
    void FillTilingData();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    ge::DataType dataType = ge::DT_UNDEFINED;

    gert::TilingContext *tilingContext = nullptr;
    gert::Shape inputShape;
    const gert::ContinuousVector *outputSize = nullptr;
    const float *scaleD = nullptr;
    const float *scaleH = nullptr;
    const float *scaleW = nullptr;

    float realScaleW = ONE_FLOAT;
    float realScaleH = ONE_FLOAT;
    float realScaleD = ONE_FLOAT;

    int64_t batches = 0;
    int64_t outputShapes[3] = {0};
    int64_t inputShapes[3] = {0};

    int64_t needCoreNum = 0;

    int64_t slideSizeW = BEST_PERFORMANCE_SIZE_1;
    UpsampleNearest3dTilingData tilingData;
};

inline bool FloatEqual(float a, float b)
{
    float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleNearest3dTiling::Init()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearest3dTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的参数
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t idx = 0;
    outputSize = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    scaleD = attrs->GetAttrPointer<float>(idx++);
    scaleH = attrs->GetAttrPointer<float>(idx++);
    scaleW = attrs->GetAttrPointer<float>(idx++);

    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = tilingContext->GetInputDesc(0)->GetDataType();

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();

    GetShapes();
    SetScale();
    GetSlideSize();

    // 数据分核
    auto compileInfo = reinterpret_cast<const UpsampleNearest3dCompileInfo *>(tilingContext->GetCompileInfo());
    int64_t coreNumPlatform = compileInfo->coreNum;
    GetNeedCoreNum(coreNumPlatform);
    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(1);
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dTiling::GetShapes()
{
    const int64_t *outputSizeArray = reinterpret_cast<const int64_t *>(outputSize->GetData());
    batches = inputShape.GetDim(0) * inputShape.GetDim(1);
    for (int8_t i = 0; i < DIM; i++) {
        inputShapes[i] = inputShape.GetDim(i + BATCH_DIM);
        outputShapes[i] = outputSizeArray[i];
    }
    tilingData.set_batches(batches);
    tilingData.set_inputShapes(inputShapes);
    tilingData.set_outputShapes(outputShapes);
}

void UpsampleNearest3dTiling::SetScale()
{
    realScaleD = ComputeScaleValue(inputShapes[D_INDEX], outputShapes[D_INDEX], scaleD);
    realScaleH = ComputeScaleValue(inputShapes[H_INDEX], outputShapes[H_INDEX], scaleH);
    realScaleW = ComputeScaleValue(inputShapes[W_INDEX], outputShapes[W_INDEX], scaleW);
    tilingData.set_scaleD(realScaleD);
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
}

inline float UpsampleNearest3dTiling::ComputeScaleValue(int64_t inSize, int64_t outSize, const float *scale) const
{
    if (*scale > ZERO_FLOAT) {
        return *scale;
    } else {
        return outSize != 0 ? (static_cast<float>(inSize) / outSize) : ZERO_FLOAT;
    }
}

void UpsampleNearest3dTiling::GetSlideSize()
{
    if (realScaleW <= BEST_PERFORMANCE_SCALE_4) {
        slideSizeW = BEST_PERFORMANCE_SIZE_4;
    } else if (realScaleW <= BEST_PERFORMANCE_SCALE_3) {
        slideSizeW = BEST_PERFORMANCE_SIZE_3;
    } else if (realScaleW <= BEST_PERFORMANCE_SCALE_2) {
        slideSizeW = BEST_PERFORMANCE_SIZE_2;
    } else {
        slideSizeW = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slideSizeW(slideSizeW);
}

void UpsampleNearest3dTiling::GetNeedCoreNum(int64_t coreNumPlatform)
{
    int64_t slideNumW = CeilA2B(outputShapes[W_INDEX], slideSizeW);
    int64_t tensorSizeW = Ceil(slideSizeW * std::min(realScaleW, BEST_PERFORMANCE_SCALE_1)) + RESERVED_LENGTH;

    int64_t slideNumH = outputShapes[H_INDEX];
    int64_t tensorSizeH = 1;
    if (realScaleH > ZERO_FLOAT && realScaleH < ONE_FLOAT) {
        slideNumH = inputShapes[H_INDEX];
        tensorSizeH = RESERVED_LENGTH;
    }

    int64_t slideNumD = outputShapes[D_INDEX];
    int64_t tensorSizeD = 1;
    if (realScaleD > ZERO_FLOAT && realScaleD < ONE_FLOAT) {
        slideNumD = inputShapes[D_INDEX];
        tensorSizeD = RESERVED_LENGTH;
    }

    int64_t slideNum = slideNumW * slideNumH * slideNumD;
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    int64_t inputRow = batches;
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = 1;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = CeilA2B(inputRow, groupCoreNum);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRow, tailAvergingRow));
    }

    needCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        needCoreNum = remainder * groupCoreNum;
    }

    tilingData.set_tensorSizeW(tensorSizeW);
    tilingData.set_tensorSizeH(tensorSizeH);
    tilingData.set_tensorSizeD(tensorSizeD);
    tilingData.set_slideNumH(slideNumH);
    tilingData.set_slideNumD(slideNumD);
    tilingData.set_eachCoreSlideNum(eachCoreSlideNum);
    tilingData.set_remainder(remainder);
    tilingData.set_tailStartSlideNum(eachCoreSlideNum * coreNumPlatform);
    tilingData.set_groupCoreNum(groupCoreNum);
    tilingData.set_inputRow(inputRow);
    tilingData.set_tailAvergingRow(tailAvergingRow);
    tilingData.set_needCoreNum(needCoreNum);
}

void UpsampleNearest3dTiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

uint8_t UpsampleNearest3dTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return HALF_TYPE;
        case ge::DT_FLOAT:
            return FLOAT_TYPE;
        case ge::DT_BF16:
            return BFLOAT_TYPE;
        default:
            return 0;
    }
}

template <typename T1, typename T2>
inline T1 UpsampleNearest3dTiling::CeilA2B(T1 a, T2 b) const
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleNearest3dTiling::Ceil(T1 x) const
{
    int32_t floorX = int32_t(x);
    if (FloatEqual(x, floorX)) {
        return floorX;
    }
    return floorX + 1;
}

static ge::graphStatus Tiling4UpsampleNearest3dTiling(gert::TilingContext *context)
{
    UpsampleNearest3dTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<UpsampleNearest3dCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_TILING_CHECK(compileInfo->coreNum <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "UpsampleNearest3d GetHardwareInfo Failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearest3d)
    .Tiling(Tiling4UpsampleNearest3dTiling)
    .TilingParse<UpsampleNearest3dCompileInfo>(TilingPrepareTiling);

}  // namespace optiling

namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t OUT_Y = 0;
static constexpr size_t SUPPORTED_DIM_NUM = 5;         // N,C,D,H,W
static constexpr size_t SUPPORTED_OUTPUT_DIM_NUM = 3;  // D,H,W
static constexpr size_t INDEX_OUTPUT_SIZE = 0;
static constexpr size_t INDEX_SCALES = 1;
static constexpr size_t NOT_CHANGE_DIM = 2;

static ge::graphStatus InferShape4UpsampleNearest3d(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "begin to do InferShape4UpsampleNearest3d");
    const gert::Shape *input_shape = context->GetInputShape(IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input_shape);
    gert::Shape *output_shape = context->GetOutputShape(OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    const size_t input_dim = input_shape->GetDimNum();
    OPS_CHECK(input_dim != SUPPORTED_DIM_NUM,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
            context->GetNodeName(), "Expected dim of input x should be 5."),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto output_size = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(INDEX_OUTPUT_SIZE);
    auto scales = attrs->GetAttrPointer<gert::TypedContinuousVector<float>>(INDEX_SCALES);

    *output_shape = *input_shape;

    if (output_size->GetSize() != 0 && scales->GetSize() == 0) {
        OPS_CHECK(output_size->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                "attr::output_size dims must be 3"),
            return ge::GRAPH_FAILED);
        // set output shape D,H,W dim
        auto output_size_data = reinterpret_cast<const int64_t *>(output_size->GetData());
        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            output_shape->SetDim(i + NOT_CHANGE_DIM, output_size_data[i]);
        }
    } else if (output_size->GetSize() == 0 && scales->GetSize() != 0) {
        OPS_CHECK(scales->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
                context->GetNodeName(), "attr::scales dims must be 3"),
            return ge::GRAPH_FAILED);
        // calculate and set output D,H,W dim
        auto scales_data = reinterpret_cast<const float *>(scales->GetData());
        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            output_shape->SetDim(i + NOT_CHANGE_DIM, input_shape->GetDim(i + NOT_CHANGE_DIM) * scales_data[i]);
        }
    } else {
        OP_LOGE(context->GetNodeName(),
            "only one of attr::output_size or attr::scales should be defined as a non-empty value.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UpsampleNearest3d).InferShape(InferShape4UpsampleNearest3d);

class UpsampleNearest3d : public OpDef {
public:
    explicit UpsampleNearest3d(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("output_size").AttrType(OPTIONAL).ListInt();
        this->Attr("scale_d").AttrType(OPTIONAL).Float();
        this->Attr("scale_h").AttrType(OPTIONAL).Float();
        this->Attr("scale_w").AttrType(OPTIONAL).Float();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config310p;
        config310p.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config310p.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        config310p.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend310p", config310p);
    }
};

OP_ADD(UpsampleNearest3d);
}  // namespace ops