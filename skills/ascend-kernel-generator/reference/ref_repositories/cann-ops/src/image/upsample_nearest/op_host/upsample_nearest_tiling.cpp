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
 * \file upsample_nearest_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_nearest_tiling.h"

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
constexpr int8_t NHWC_N_INDEX = 0;
constexpr int8_t NHWC_H_INDEX = 1;
constexpr int8_t NHWC_W_INDEX = 2;
constexpr int8_t NHWC_C_INDEX = 3;

constexpr int8_t NCHW_N_INDEX = 0;
constexpr int8_t NCHW_C_INDEX = 1;
constexpr int8_t NCHW_H_INDEX = 2;
constexpr int8_t NCHW_W_INDEX = 3;

constexpr int8_t OUT_H_INDEX = 0;
constexpr int8_t OUT_W_INDEX = 1;
constexpr int8_t OUT_L_INDEX = 0;

constexpr int8_t NLC_N_INDEX = 0;
constexpr int8_t NLC_L_INDEX = 1;
constexpr int8_t NLC_C_INDEX = 2;

constexpr uint32_t OUTPUT_SIZE_ATTR = 0;

constexpr uint32_t SCALE_H_ATTR = 1;
constexpr uint32_t SCALE_W_ATTR = 2;
constexpr uint32_t EXACT_ATTR = 3;
constexpr uint32_t SCALE_L_ATTR = 1;

constexpr uint32_t DATE_TYPE_FLOAT16 = 1;
constexpr uint32_t DATE_TYPE_FLOAT = 2;
constexpr uint32_t DATE_TYPE_HALF = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint32_t NHWC_DIM_SIZE = 4;
constexpr uint32_t NLC_DIM_SIZE = 3;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t COMMON_TILING_KEY = 1000;
constexpr uint32_t SMALL_CW_TILING_KEY = 1001;
constexpr uint32_t SMALL_C_TILING_KEY = 1002;
constexpr uint32_t SMALL_NCH_TILING_KEY = 1003;

constexpr uint32_t MAX_SMALL_SHPAE = 8192;
constexpr uint32_t MAX_SMALL_SACALE = 2;

class UpsampleNearestTiling {
public:
    explicit UpsampleNearestTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    ge::graphStatus ParseInputAttrs();
    inline float ComputeScaleValue(int64_t inputSize, int64_t outputSize, const float scale);
    void GetWorkSpace();
    uint32_t GetDataTypeVal();
    uint32_t GetBestAvergingCols(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNum(uint32_t maxCoreNum);
    void FillTilingData();
    void GetTilingKey();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b);

    template <typename T1>
    inline int32_t Ceil(T1 x);

private:
    UpsampleNearestTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    ge::Format inputFormat = ge::Format::FORMAT_NHWC;
    uint8_t dim = 0;
    float realScaleH = 0.0f;
    float realScaleW = 0.0f;
    uint32_t tailColStartList[MAX_CORE_CONT] = {0};
    uint32_t tailColEndList[MAX_CORE_CONT] = {0};
    uint32_t tailRowStartList[MAX_CORE_CONT] = {0};
    uint32_t tailRowEndList[MAX_CORE_CONT] = {0};

    int64_t outputShapes[4] = {0};
    int64_t inputShapes[4] = {0};

    bool exactMode = true;
    uint32_t tilingKey = 1000;
    uint32_t needCoreNum = 1;
};

ge::graphStatus UpsampleNearestTiling::RunBigKernelTiling()
{
    if (ParseInputAttrs() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    GetTilingKey();

    auto compileInfo = reinterpret_cast<const UpsampleNearestCompileInfo *>(tilingContext->GetCompileInfo());
    uint32_t totalCoreNum = compileInfo->totalCoreNum;
    needCoreNum = GetNeedCoreNum(totalCoreNum);
    GetWorkSpace();
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearestTiling::ParseInputAttrs()
{
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto srcShape = tilingContext->GetInputShape(0);
    dim = srcShape->GetStorageShape().GetDimNum();

    auto inputShape = srcShape->GetOriginShape();

    const gert::ContinuousVector *outputSizeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_ATTR);
    const int64_t *outputSizeArray = reinterpret_cast<const int64_t *>(outputSizeAttr->GetData());

    exactMode = *(attrs->GetAttrPointer<bool>(EXACT_ATTR));

    for (int8_t i = 0; i < dim; i++) {
        inputShapes[i] = inputShape.GetDim(i);
        outputShapes[i] = inputShape.GetDim(i);
    }
    inputFormat = static_cast<ge::Format>(GetPrimaryFormat(tilingContext->GetInputDesc(0)->GetStorageFormat()));
    if (dim == NHWC_DIM_SIZE) {
        const float scaleH = *(attrs->GetAttrPointer<float>(SCALE_H_ATTR));
        const float scaleW = *(attrs->GetAttrPointer<float>(SCALE_W_ATTR));
        if (inputFormat == ge::Format::FORMAT_NCHW) {
            outputShapes[NCHW_H_INDEX] = outputSizeArray[OUT_H_INDEX];
            outputShapes[NCHW_W_INDEX] = outputSizeArray[OUT_W_INDEX];
            realScaleH = ComputeScaleValue(inputShapes[NCHW_H_INDEX], outputShapes[NCHW_H_INDEX], scaleH);
            realScaleW = ComputeScaleValue(inputShapes[NCHW_W_INDEX], outputShapes[NCHW_W_INDEX], scaleW);
        } else {
            outputShapes[NHWC_H_INDEX] = outputSizeArray[OUT_H_INDEX];
            outputShapes[NHWC_W_INDEX] = outputSizeArray[OUT_W_INDEX];
            realScaleH = ComputeScaleValue(inputShapes[NHWC_H_INDEX], outputShapes[NHWC_H_INDEX], scaleH);
            realScaleW = ComputeScaleValue(inputShapes[NHWC_W_INDEX], outputShapes[NHWC_W_INDEX], scaleW);
        }
    } else if (dim == NLC_DIM_SIZE) {
        inputShapes[NHWC_H_INDEX] = 1;
        inputShapes[NHWC_W_INDEX] = inputShape.GetDim(NLC_L_INDEX);
        inputShapes[NHWC_C_INDEX] = inputShape.GetDim(NLC_C_INDEX);
        outputShapes[NHWC_H_INDEX] = 1;
        outputShapes[NHWC_W_INDEX] = outputSizeArray[OUT_L_INDEX];
        outputShapes[NHWC_C_INDEX] = inputShapes[NLC_C_INDEX];
        const float scaleL = *(attrs->GetAttrPointer<float>(SCALE_H_ATTR));
        realScaleH = 1.0;
        realScaleW = ComputeScaleValue(inputShapes[NHWC_W_INDEX], outputShapes[NHWC_W_INDEX], scaleL);
    } else {
        return ge::GRAPH_FAILED;
    }

    auto srcDtype = tilingContext->GetInputDesc(0)->GetDataType();
    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void UpsampleNearestTiling::GetWorkSpace()
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
}

void UpsampleNearestTiling::GetTilingKey()
{
    if (inputFormat == ge::Format::FORMAT_NCHW) {
        tilingKey = SMALL_NCH_TILING_KEY;
    } else {
        uint32_t inputC = inputShapes[NHWC_C_INDEX];
        uint32_t outputH = inputShapes[NHWC_H_INDEX];
        uint32_t outputW = inputShapes[NHWC_W_INDEX];
        uint32_t inputW = outputShapes[NHWC_W_INDEX];
        if (inputC * inputW < MAX_SMALL_SHPAE && inputC * outputW < MAX_SMALL_SHPAE && outputH > 1) {
            tilingKey = SMALL_CW_TILING_KEY;
        } else if (inputC < MAX_SMALL_SHPAE && outputW < MAX_SMALL_SHPAE && realScaleW < MAX_SMALL_SACALE) {
            tilingKey = SMALL_C_TILING_KEY;
        } else {
            tilingKey = COMMON_TILING_KEY;
        }
    }
}

inline float UpsampleNearestTiling::ComputeScaleValue(int64_t inputSize, int64_t outputSize, const float scale)
{
    if ((dim == NHWC_DIM_SIZE) && (inputSize == outputSize)) {
        return static_cast<float>(1);
    }
    if (scale > 0) {
        return scale;
    } else {
        return static_cast<float>(inputSize) / static_cast<float>(outputSize);
    }
}

template <typename T1, typename T2>
inline T1 UpsampleNearestTiling::CeilA2B(T1 a, T2 b)
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

uint32_t UpsampleNearestTiling::GetDataTypeVal()
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return 0;
    }
}

uint32_t UpsampleNearestTiling::GetBestAvergingCols(uint32_t coreNumPlatform)
{
    uint32_t outputH = outputShapes[NHWC_H_INDEX];
    uint32_t outputW = outputShapes[NHWC_W_INDEX];
    if (inputFormat == ge::Format::FORMAT_NCHW) {
        outputH = outputShapes[NCHW_H_INDEX];
        outputW = outputShapes[NCHW_W_INDEX];
    }
    uint32_t dataTypeSize = GetDataTypeVal();
    uint32_t minAvergingCols = dataTypeSize > 0 ? 32 / dataTypeSize : outputW;

    for (uint32_t i = 1; i <= coreNumPlatform; i++) {
        if ((coreNumPlatform % i == 0) && (outputW % i == 0)) {
            uint32_t j = coreNumPlatform / i;
            if (outputH % j == 0) {
                minAvergingCols = coreNumPlatform / i;
                break;
            }
        }
    }

    if (tilingKey == SMALL_CW_TILING_KEY) {
        minAvergingCols = outputW;
    }
    return minAvergingCols;
}

uint32_t UpsampleNearestTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t outputH = outputShapes[NHWC_H_INDEX];
    uint32_t outputW = outputShapes[NHWC_W_INDEX];
    if (inputFormat == ge::Format::FORMAT_NCHW) {
        outputH = outputShapes[NCHW_H_INDEX];
        outputW = outputShapes[NCHW_W_INDEX];
    }
    uint32_t realCoreNum = 0;
    uint32_t slideSizeW = 0;
    uint32_t slideSizeH = 0;
    uint32_t colGroupNum = 0;
    uint32_t groupRowCoreNum = 0;
    uint32_t groupColCoreNum = 0;
    uint32_t minAvergingCols = GetBestAvergingCols(coreNumPlatform);

    if (outputH < coreNumPlatform) {
        uint32_t tailAvergingCols = std::max(CeilA2B(outputW, coreNumPlatform), minAvergingCols);
        colGroupNum = std::min(coreNumPlatform, CeilA2B(outputW, tailAvergingCols));
        slideSizeW = tailAvergingCols;
    } else {
        colGroupNum = 1;
        slideSizeW = outputW;
    }
    groupColCoreNum = coreNumPlatform / colGroupNum;
    uint32_t tailAvergingRows = std::max(outputH / groupColCoreNum, (uint32_t)1);
    groupRowCoreNum = std::min(groupColCoreNum, CeilA2B(outputH, tailAvergingRows));
    realCoreNum = colGroupNum * groupRowCoreNum;
    uint32_t tailRowRemainder = outputH - groupRowCoreNum * tailAvergingRows;
    slideSizeH = tailAvergingRows;

    uint32_t realNeedCoreNum = 0;
    uint32_t tailRowOffset = 0;
    uint32_t tempTailRowRemainder = tailRowRemainder;
    for (uint32_t coreIndex = 0; coreIndex < realCoreNum; coreIndex++) {
        uint32_t groupColIndex = coreIndex / groupRowCoreNum;
        tailColStartList[coreIndex] = groupColIndex * slideSizeW;
        tailColEndList[coreIndex] = std::min((groupColIndex + 1) * slideSizeW, outputW);
        int32_t groupRowIndex = coreIndex % groupRowCoreNum;
        if (groupRowIndex == 0) {
            tempTailRowRemainder = tailRowRemainder;
            tailRowOffset = 0;
        }
        tailRowStartList[coreIndex] = groupRowIndex * slideSizeH + tailRowOffset;
        if (tempTailRowRemainder > 0) {
            tempTailRowRemainder -= 1;
            tailRowOffset += 1;
        }
        tailRowEndList[coreIndex] = std::min((groupRowIndex + 1) * slideSizeH + tailRowOffset, outputH);
        realNeedCoreNum++;
    }
    realNeedCoreNum = realNeedCoreNum < 1 ? 1 : realNeedCoreNum;

    return realNeedCoreNum;
}

void UpsampleNearestTiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
    tilingData.set_exactMode(exactMode);

    tilingData.set_needCoreNum(needCoreNum);

    tilingData.set_inputShapes(inputShapes);
    tilingData.set_outputShapes(outputShapes);

    tilingData.set_tailColStartList(tailColStartList);
    tilingData.set_tailColEndList(tailColEndList);
    tilingData.set_tailRowStartList(tailRowStartList);
    tilingData.set_tailRowEndList(tailRowEndList);

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(tilingKey);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleNearestTiling(gert::TilingContext *context)
{
    UpsampleNearestTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
    return ge::GRAPH_FAILED;
}

static ge::graphStatus tilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<UpsampleNearestCompileInfo>(context);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearest)
    .Tiling(tiling4UpsampleNearestTiling)
    .TilingParse<UpsampleNearestCompileInfo>(tilingPrepareTiling);

}  // namespace optiling
