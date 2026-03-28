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
 * \file grid_sample.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "grid_sample_tiling.h"
#include "shape_utils.h"

using namespace ge;
using namespace AscendC;

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
constexpr uint64_t TILING_OFFSET = 1000000000000UL;
static const size_t DIM_NUM_4D = 4;
static const size_t DIM_NUM_5D = 5;
static const size_t DIM_2 = 2;
static const size_t DIM_3 = 3;
static const size_t DIM_4 = 4;
static const size_t INT_16 = 16;
static const size_t INT_22 = 22;
static const size_t INT_64 = 64;
static const size_t INT_88 = 88;
static const int64_t INTERPOLATION_MODE_BILNEAR = 0;
static const int64_t INTERPOLATION_MODE_NEAREST = 1;
static const int64_t INTERPOLATION_MODE_BICUBIC = 2;
static const int64_t PADDING_MODE_ZEROS = 0;
static const int64_t PADDING_MODE_BORDER = 1;
static const int64_t PADDING_MODE_REFLECTION = 2;
static const int64_t ALIGN_CORNERS_FALSE = 0;
static const int64_t ALIGN_CORNERS_TRUE = 1;
static const int64_t MINI_IH_IW_MAX_SIZE = 65536;
static const int64_t MINI_IH_IW_MAX_SIZE_FP16 = 32768;
static const int64_t TILING_HW_FACTOR = 1024;
static const int64_t CHANEL_LAST_TRUE = 1;
static const int64_t CHANEL_LAST_FALSE = 0;
const static int64_t SIZE_16 = 16;
const static int64_t LENGTH_1024 = 1024;
const static int64_t FULL_LOAD_TYPE = 2;
const static int64_t X_MAX_HWC_FACTOR = 20480;  // 20k
const static int64_t C1_X_COUNT = 4096;
const static int64_t NUM_C32 = 32;
const static int64_t MIN_HW_C32 = 8;
const static int64_t TEMPLATE_C32 = 2;
const static int64_t DOUBLE = 2;

uint64_t GridSampleTiling::GetTilingKey() const
{
    GridSampleDtypeKey dtypeKey = GridSampleDtypeKey::FLOAT32;
    if (xDtype == ge::DT_FLOAT16) {
        dtypeKey = GridSampleDtypeKey::FLOAT16;
    } else if (xDtype == ge::DT_BF16) {
        dtypeKey = GridSampleDtypeKey::BFLOAT16;
    }

    uint64_t tilingKey =
        GET_TILINGKEY(interpolationMode, dtypeKey, dimValue, schedulerMode, dimension, templateCNum, tempType);
    OP_LOGD(context_->GetNodeName(), "schedulerMode:%ld,tilingKey:%zu.", schedulerMode, tilingKey);

    return tilingKey % TILING_OFFSET;
}

ge::graphStatus GridSampleTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo begin.");
    auto inputX = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputXDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    xDtype = inputXDesc->GetDataType();
    auto gridXDesc = context_->GetInputDesc(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, gridXDesc);
    auto gridDtype = gridXDesc->GetDataType();
    auto xShape = ops::EnsureNotScalar(inputX->GetStorageShape());
    auto inputGrid = context_->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGrid);
    auto gridShape = ops::EnsureNotScalar(inputGrid->GetStorageShape());

    OP_TILING_CHECK((xShape.GetDimNum() != DIM_NUM_4D && xShape.GetDimNum() != DIM_NUM_5D) ||
                        (gridShape.GetDimNum() != DIM_NUM_4D && gridShape.GetDimNum() != DIM_NUM_5D),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "x / grid shape length should be 4 or 5"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xShape.GetDimNum() != gridShape.GetDimNum(),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "x / grid shape length should be equal."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(gridShape.GetDim(0) != xShape.GetDim(0),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "x / grid shape[0] should be same"),
        return ge::GRAPH_FAILED);

    if (xShape.GetDimNum() == DIM_NUM_5D) {
        dimension = 1;
        dimValue = gridShape.GetDim(DIM_4);
        OP_TILING_CHECK(dimValue != DIM_3,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "only support (N, D, H, W, 3) for grid"),
            return ge::GRAPH_FAILED);
    } else {
        dimension = 0;
        dimValue = gridShape.GetDim(DIM_3);
        OP_TILING_CHECK(dimValue != DIM_2,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "only support (N, H, W, 2) for grid"),
            return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK((dimension == 0 && xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "x datatype only support FLOAT32 or FLOAT16"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((dimension == 0 && gridDtype != ge::DT_FLOAT && gridDtype != ge::DT_FLOAT16),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "grid datatype only support FLOAT32 or FLOAT16"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((dimension == 1 && xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "x datatype only support FLOAT32, FLOAT16, BFLOAT16"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (dimension == 1 && gridDtype != ge::DT_FLOAT && gridDtype != ge::DT_FLOAT16 && gridDtype != ge::DT_BF16),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "grid datatype only support FLOAT32, FLOAT16, BFLOAT16"),
        return ge::GRAPH_FAILED);

    auto *attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char *pInterpolationMode = attrs->GetAttrPointer<char>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, pInterpolationMode);
    if (strcmp(pInterpolationMode, "bilinear") == 0) {
        interpolationMode = INTERPOLATION_MODE_BILNEAR;
    } else if (strcmp(pInterpolationMode, "bicubic") == 0) {
        interpolationMode = INTERPOLATION_MODE_BICUBIC;
    } else if (strcmp(pInterpolationMode, "nearest") == 0) {
        interpolationMode = INTERPOLATION_MODE_NEAREST;
    } else {
        OP_LOGE(context_->GetNodeName(), "interpolation_mode only support bilinear or nearest or bicubic.");
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(dimension == 1 && interpolationMode == INTERPOLATION_MODE_BICUBIC,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "GridSampler3D interpolation_mode only support bilinear or nearest"),
        return ge::GRAPH_FAILED);

    const char *pPaddingMode = attrs->GetAttrPointer<char>(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, pPaddingMode);
    if (strcmp(pPaddingMode, "zeros") == 0) {
        paddingMode = PADDING_MODE_ZEROS;
    } else if (strcmp(pPaddingMode, "border") == 0) {
        paddingMode = PADDING_MODE_BORDER;
    } else if (strcmp(pPaddingMode, "reflection") == 0) {
        paddingMode = PADDING_MODE_REFLECTION;
    } else {
        OP_LOGE(context_->GetNodeName(), "padding_mode only support zeros or border or reflection.");
        return ge::GRAPH_FAILED;
    }

    const bool *pAlignCorners = attrs->GetAttrPointer<bool>(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, pAlignCorners);
    alignCorners = ALIGN_CORNERS_FALSE;
    if (*pAlignCorners) {
        alignCorners = ALIGN_CORNERS_TRUE;
    }

    const bool *pChannelLast = attrs->GetAttrPointer<bool>(3);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, pChannelLast);
    channelLast = CHANEL_LAST_FALSE;
    if (*pChannelLast) {
        channelLast = CHANEL_LAST_TRUE;
    }

    inN = xShape.GetDim(0);
    if (dimension == 0) {
        if (channelLast == 0) {
            inC = xShape.GetDim(1);
            inH = xShape.GetDim(DIM_2);
            inW = xShape.GetDim(DIM_3);
        } else {
            inH = xShape.GetDim(1);
            inW = xShape.GetDim(DIM_2);
            inC = xShape.GetDim(DIM_3);
        }
        outH = gridShape.GetDim(1);
        outW = gridShape.GetDim(DIM_2);

        if ((channelLast == 1) && (strcmp(pInterpolationMode, "bilinear") == 0) &&
            (inC * inH * inW <= X_MAX_HWC_FACTOR)) {
            tempType = FULL_LOAD_TYPE;
            hwFactor = TILING_HW_FACTOR;
            OP_LOGD(context_->GetNodeName(), "Get in FullLoad Template.");
            if ((inC == 1) && (inH * inW < C1_X_COUNT)) {
                templateCNum = 1;
            } else if ((inC == NUM_C32) && (inH > MIN_HW_C32) && (inW > MIN_HW_C32)) {
                templateCNum = TEMPLATE_C32;
            } else {
                templateCNum = 0;
            }
        }

        OP_TILING_CHECK(inN < 1 || inC < 1 || inH < 1 || inW < 1 || outW < 1 || outH < 1,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Invalid shape. Maybe empty tensor."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(inH * inW > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            VECTOR_INNER_ERR_REPORT_TILIING(
                context_->GetNodeName(), "no support for H*W of x greater than int32 max value"),
            return ge::GRAPH_FAILED);

        const int32_t *pSchedulerMode = attrs->GetAttrPointer<int32_t>(4);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, pSchedulerMode);
        OP_LOGD(context_->GetNodeName(), "scheduler_mode is: %d", *pSchedulerMode);
        schedulerMode = *pSchedulerMode;
        OP_TILING_CHECK(schedulerMode != 0 && schedulerMode != 1,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "scheduler_mode only support 0 or 1."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!(*pChannelLast) && schedulerMode == 1,
            VECTOR_INNER_ERR_REPORT_TILIING(
                context_->GetNodeName(), "scheduler_mode support 1 only in the channel last scenario."),
            return ge::GRAPH_FAILED);
    } else {
        if (channelLast == 0) {
            inC = xShape.GetDim(1);
            inD = xShape.GetDim(DIM_2);
            inH = xShape.GetDim(DIM_3);
            inW = xShape.GetDim(DIM_4);
        } else {
            inD = xShape.GetDim(1);
            inH = xShape.GetDim(DIM_2);
            inW = xShape.GetDim(DIM_3);
            inC = xShape.GetDim(DIM_4);
        }
        outD = gridShape.GetDim(1);
        outH = gridShape.GetDim(DIM_2);
        outW = gridShape.GetDim(DIM_3);

        OP_TILING_CHECK(inN < 1 || inC < 1 || inD < 1 || inH < 1 || inW < 1 || outD < 1 || outW < 1 || outH < 1,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Invalid shape. Maybe empty tensor."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(inH * inW * inD > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            VECTOR_INNER_ERR_REPORT_TILIING(
                context_->GetNodeName(), "no support for D*H*W of x greater than int32 max value"),
            return ge::GRAPH_FAILED);

        // 添加判断是否都为特例场景，若为特例场景schedulerMode为1，否则为默认值0
        if (inN == gridShape.GetDim(0) && inD == outD && inH == outH && inW == outW && inD == INT_16 && inH == INT_64 &&
            inW == INT_64 && dimValue == DIM_3 && inC == DIM_4 && (inN == INT_22 || inN == INT_88)) {
            schedulerMode = 1;
        }
    }

    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GridSampleTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const GridSampleCompileInfo *>(context_->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from compile info.");
        coreNumVar = compileInfo->coreNum;
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering into get core num from platform.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNumVar = ascendcPlatform.GetCoreNumAiv();
    }
    return ge::GRAPH_SUCCESS;
}

bool GridSampleTiling::IsCapable()
{
    return true;
}

ge::graphStatus GridSampleTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GridSampleTiling::GetWorkspaceSize()
{
    int64_t outHW = outH * outW;
    needCoreNum = coreNumVar;
    if (inN < coreNumVar && outHW <= hwFactor) {
        needCoreNum = inN;
    }
    workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
    if (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) {
        // 每个核使用inC * 512(1024) * dtype(float),再乘上核数
        size_t outputShapeSize = needCoreNum * inC * hwFactor * sizeof(float);
        workspaceSize_ = workspaceSize_ + outputShapeSize;
    }
    if (tempType == FULL_LOAD_TYPE) {
        workspaceSize_ = workspaceSize_ * DOUBLE;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GridSampleTiling::DoOpTiling()
{
    tilingData.set_coreNumVar(coreNumVar);
    tilingData.set_inN(inN);
    tilingData.set_inC(inC);
    tilingData.set_inD(inD);
    tilingData.set_inH(inH);
    tilingData.set_inW(inW);
    tilingData.set_outD(outD);
    tilingData.set_outH(outH);
    tilingData.set_outW(outW);
    tilingData.set_interpolationMode(interpolationMode);
    tilingData.set_paddingMode(paddingMode);
    tilingData.set_alignCorners(alignCorners);
    tilingData.set_channelLast(channelLast);

    // output format is [N, C, H, W]
    int64_t outputD = outD == 0 ? 1 : outD;
    int64_t outputHW = outH * outW * outputD;
    if (inN < coreNumVar && outputHW <= hwFactor) {
        tilingData.set_needCoreNum(inN);
    } else {
        tilingData.set_needCoreNum(coreNumVar);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GridSampleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData.get_needCoreNum());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;

    gert::TilingData *rawTilingData = context_->GetRawTilingData();
    OP_TILING_CHECK(tilingData.GetDataSize() > rawTilingData->GetCapacity(),
        VECTOR_INNER_ERR_REPORT_TILIING(context_,
            "actual tiling data size %zu > context tiling data size %zu",
            tilingData.GetDataSize(),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GridSample(gert::TilingContext *context)
{
    // 初始化算子Tiling类
    GridSampleTiling tiling(context);
    // 执行算子tiling框架
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4GridSample(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4GridSample running.");

    auto compileInfo = GetCompileInfoPtr<GridSampleCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->coreNum <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "Get core num failed, core num: %u",
            static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_TILING_CHECK((compileInfo->ubSizePlatForm <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
            "Get ub size failed, ub size: %u",
            static_cast<uint32_t>(compileInfo->ubSizePlatForm)),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepare4GridSample end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GridSample).Tiling(Tiling4GridSample).TilingParse<GridSampleCompileInfo>(TilingPrepare4GridSample);
}  // namespace optiling

namespace ops {
constexpr int64_t DIM_NUM_2D = 4;
constexpr int64_t DIM_NUM_3D = 5;
constexpr int64_t INTERPOLATION_DIM_2D = 2;
constexpr int64_t INTERPOLATION_DIM_3D = 3;
constexpr uint64_t X_IDX_CHANNEL = 3;
constexpr uint64_t X_IDX_CHANNEL_3D = 4;
constexpr uint64_t GRID_DIM_IDX_W = 2;
constexpr uint64_t GRID_DIM_IDX_DIMS = 3;
constexpr uint64_t GRID_3D_DIM_IDX_DIMS = 4;
constexpr uint64_t Y_DIM_IDX_DIMS_START = 2;
constexpr uint64_t Y_DIM_IDX_H = 2;
constexpr uint64_t Y_DIM_IDX_W = 3;
constexpr uint64_t ATTR_IDX_CHANNEL_LAST = 3;
constexpr uint64_t NUM_1 = 1;
constexpr uint64_t NUM_2 = 2;
constexpr uint64_t NUM_3 = 3;
constexpr uint64_t NUM_4 = 4;

static ge::graphStatus InferDataType4GridSample(gert::InferDataTypeContext *context)
{
    OPS_CHECK(context == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("GridSample", "InferDataTypeContext is nullptr"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferDataType4GridSample begin");

    context->SetOutputDataType(0, context->GetInputDataType(0));

    OP_LOGD(context->GetNodeName(), "InferDataType4GridSample end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferGridSampleShape2D(
    gert::InferShapeContext *context, const gert::Shape *xShape, const gert::Shape *gridShape, gert::Shape *yShape)
{
    OP_LOGD(context->GetNodeName(), "InferGridSampleShape2D begin");

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *channelLast = attrs->GetAttrPointer<bool>(ATTR_IDX_CHANNEL_LAST);
    OPS_CHECK_NULL_WITH_CONTEXT(context, channelLast);
    OP_LOGD(context->GetNodeName(), "channel_last attribute is :%d", *channelLast);

    int64_t nDim = xShape->GetDim(0U);
    OPS_CHECK(nDim == 0,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "no support for N is 0"),
        return ge::GRAPH_FAILED);
    if (nDim < 0) {
        nDim = gridShape->GetDim(0U);  // if N from input_x is -1, then use N value from input_grid
    }

    int64_t cDim = xShape->GetDim(1U);
    if (*channelLast) {
        cDim = xShape->GetDim(X_IDX_CHANNEL);
    }
    OPS_CHECK(cDim == 0,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "no support for C is 0"),
        return ge::GRAPH_FAILED);

    int64_t hDim = gridShape->GetDim(1U);
    int64_t wDim = gridShape->GetDim(GRID_DIM_IDX_W);

    yShape->SetDimNum(DIM_NUM_2D);
    yShape->SetDim(0, nDim);
    yShape->SetDim(1, cDim);
    yShape->SetDim(Y_DIM_IDX_H, hDim);
    yShape->SetDim(Y_DIM_IDX_W, wDim);

    OP_LOGD(context->GetNodeName(), "InferGridSampleShape2D end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferGridSampleShape3D(gert::InferShapeContext *context, const gert::Shape *xShape,
    const gert::Shape *gridShape, gert::Shape *yShape, const Format format)
{
    OP_LOGD(context->GetNodeName(), "InferGridSampleShape3D begin");

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    bool channelLast = false;
    if (format == FORMAT_NDHWC) {
        channelLast = true;
    }

    int64_t nDim = xShape->GetDim(0U);
    OPS_CHECK(nDim == 0,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "no support for N is 0"),
        return ge::GRAPH_FAILED);
    if (nDim < 0) {
        nDim = gridShape->GetDim(0U);  // if N from input_x is -1, then use N value from input_grid
    }

    int64_t cDim = xShape->GetDim(1U);
    int64_t dDim = gridShape->GetDim(NUM_1);
    int64_t hDim = gridShape->GetDim(NUM_2);
    int64_t wDim = gridShape->GetDim(NUM_3);
    if (channelLast) {
        cDim = xShape->GetDim(X_IDX_CHANNEL_3D);
    }
    OPS_CHECK(cDim == 0,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "no support for C is 0"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "cDim = %ld", cDim);
    OP_LOGD(context->GetNodeName(), "dDim = %ld", dDim);
    OP_LOGD(context->GetNodeName(), "hDim = %ld", hDim);
    OP_LOGD(context->GetNodeName(), "wDim = %ld", wDim);

    yShape->SetDimNum(DIM_NUM_3D);
    yShape->SetDim(0, nDim);
    yShape->SetDim(1, cDim);
    yShape->SetDim(NUM_2, dDim);
    yShape->SetDim(NUM_3, hDim);
    yShape->SetDim(NUM_4, wDim);

    OP_LOGD(context->GetNodeName(), "InferGridSampleShape3D end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GridSample(gert::InferShapeContext *context)
{
    OPS_CHECK(context == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("GridSample", "InferShapeContext is nullptr"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShape4GridSample begin");

    const gert::Shape *xShape = context->GetInputShape(0U);
    const gert::Shape *gridShape = context->GetInputShape(1U);
    gert::Shape *yShape = context->GetOutputShape(0U);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gridShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::Tensor *shape_tensor = context->GetInputTensor(0);
    auto format = shape_tensor->GetOriginFormat();
    OP_LOGD(context->GetNodeName(), "format = %d", format);

    if (IsUnknownRank(xShape) || IsUnknownRank(gridShape)) {
        return SetUnknownRank(yShape);
    }

    OPS_CHECK((xShape->GetDimNum() != DIM_NUM_2D || gridShape->GetDimNum() != DIM_NUM_2D) &&
                 (xShape->GetDimNum() != DIM_NUM_3D || gridShape->GetDimNum() != DIM_NUM_3D),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "shape is invalid, only support rank is 4 or 5"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(xShape->GetDim(0U) != ge::UNKNOWN_DIM && gridShape->GetDim(0U) != ge::UNKNOWN_DIM &&
                 xShape->GetDim(0U) != gridShape->GetDim(0U),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "N of x/grid should be same value"),
        return ge::GRAPH_FAILED);
    if (xShape->GetDimNum() == DIM_NUM_2D) {
        OPS_CHECK(
            gridShape->GetDim(GRID_DIM_IDX_DIMS) > 0 && gridShape->GetDim(GRID_DIM_IDX_DIMS) != INTERPOLATION_DIM_2D,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "grid shape invalid, only support rank is 4"),
            return ge::GRAPH_FAILED);

        OPS_CHECK(InferGridSampleShape2D(context, xShape, gridShape, yShape) != GRAPH_SUCCESS,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to infershape"),
            return ge::GRAPH_FAILED);
    } else {
        OPS_CHECK(gridShape->GetDim(GRID_3D_DIM_IDX_DIMS) > 0 &&
                     gridShape->GetDim(GRID_3D_DIM_IDX_DIMS) != INTERPOLATION_DIM_3D,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "grid shape invalid, only support rank is 5"),
            return ge::GRAPH_FAILED);

        OPS_CHECK(InferGridSampleShape3D(context, xShape, gridShape, yShape, format) != GRAPH_SUCCESS,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to infershape"),
            return ge::GRAPH_FAILED);
    }

    OP_LOGD(context->GetNodeName(), "InferShape4GridSample end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferGridSampleShapeRange(gert::InferShapeRangeContext *context,
    const gert::Range<gert::Shape> *xRange, const gert::Range<gert::Shape> *gridRange, gert::Range<gert::Shape> *yRange)
{
    OP_LOGD(context->GetNodeName(), "InferGridSampleShapeRange begin");

    size_t xDimNum = xRange->GetMax()->GetDimNum();
    size_t gridDimNum = xRange->GetMax()->GetDimNum();
    if (xDimNum == 0 || gridDimNum == 0) {
        yRange->GetMin()->SetDimNum(0);
        yRange->GetMax()->SetDimNum(0);
    } else if (xDimNum == 1) {
        yRange->GetMin()->SetDimNum(1);
        yRange->GetMin()->SetDim(0, xRange->GetMin()->GetDim(0));

        yRange->GetMax()->SetDimNum(1);
        yRange->GetMax()->SetDim(0, xRange->GetMax()->GetDim(0));
    } else if (gridDimNum == 1) {
        yRange->GetMin()->SetDimNum(1);
        yRange->GetMin()->SetDim(0, gridRange->GetMin()->GetDim(0));

        yRange->GetMax()->SetDimNum(1);
        yRange->GetMax()->SetDim(0, gridRange->GetMax()->GetDim(0));
    } else {
        OPS_CHECK(xDimNum != gridDimNum,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "rank of x and grid should be same"),
            return ge::GRAPH_FAILED);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const bool *channelLast = attrs->GetAttrPointer<bool>(ATTR_IDX_CHANNEL_LAST);
        OPS_CHECK_NULL_WITH_CONTEXT(context, channelLast);
        OP_LOGD(context->GetNodeName(), "channel_last attribute is :%d", *channelLast);

        // set range for N
        yRange->GetMin()->SetDimNum(xDimNum);
        yRange->GetMin()->SetDim(0, xRange->GetMin()->GetDim(0));
        yRange->GetMax()->SetDimNum(xDimNum);
        yRange->GetMax()->SetDim(0, xRange->GetMax()->GetDim(0));

        // set range for C
        if (*channelLast) {
            yRange->GetMin()->SetDim(1, xRange->GetMin()->GetDim(xDimNum - 1));
            yRange->GetMax()->SetDim(1, xRange->GetMax()->GetDim(xDimNum - 1));
        } else {
            yRange->GetMin()->SetDim(1, xRange->GetMin()->GetDim(1));
            yRange->GetMax()->SetDim(1, xRange->GetMax()->GetDim(1));
        }

        // For GridSample-2D, set range for H/W
        // For GridSample-3D, set range for D/H/W
        // For GridSample-nD, set range for H/W/....
        for (size_t axis = Y_DIM_IDX_DIMS_START; axis < xDimNum; ++axis) {
            yRange->GetMin()->SetDim(axis, gridRange->GetMin()->GetDim(axis - 1));
            yRange->GetMax()->SetDim(axis, gridRange->GetMax()->GetDim(axis - 1));
        }
    }

    OP_LOGD(context->GetNodeName(), "InferGridSampleShapeRange end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4GridSample(gert::InferShapeRangeContext *context)
{
    OPS_CHECK(context == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("GridSample", "InferShapeRangeContext is nullptr"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShapeRange4GridSample begin");

    auto xRange = context->GetInputShapeRange(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, xRange->GetMax());

    // if dim num is 0 or 1, maybe unkown rank for GridSample, infer process should not be terminated
    // if rank is known, it should be greater or equal 4
    // that is to say, GridSample op support 4-dim or 5-dim or n-dim ( n >= 4 )
    size_t xDimNum = xRange->GetMax()->GetDimNum();
    OPS_CHECK(xDimNum == 2 || xDimNum == 3,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
            context->GetNodeName(), "x range invalid, only support unkown rank or rank is greater than 3"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(xRange->GetMin()->GetDimNum() != xDimNum,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "min value of x range is invalid"),
        return ge::GRAPH_FAILED);

    auto gridRange = context->GetInputShapeRange(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gridRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gridRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, gridRange->GetMax());

    size_t gridDimNum = gridRange->GetMax()->GetDimNum();  // the explanation is similar to xDimNum
    OPS_CHECK(gridDimNum == 2 || gridDimNum == 3,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
            context->GetNodeName(), "grid range invalid, only support unkown rank or rank is greater than 3"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(gridRange->GetMin()->GetDimNum() != gridDimNum,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "min value of grid range is invalid"),
        return ge::GRAPH_FAILED);

    auto yRange = context->GetOutputShapeRange(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMax());

    OPS_CHECK(InferGridSampleShapeRange(context, xRange, gridRange, yRange) != GRAPH_SUCCESS,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to infer shape range"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "InferShapeRange4GridSample end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridSample)
    .InferDataType(InferDataType4GridSample)
    .InferShape(InferShape4GridSample)
    .InferShapeRange(InferShapeRange4GridSample);

class GridSample : public OpDef {
public:
    explicit GridSample(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("interpolation_mode").AttrType(OPTIONAL).String("bilinear");
        this->Attr("padding_mode").AttrType(OPTIONAL).String("zeros");
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("channel_last").AttrType(OPTIONAL).Bool(false);
        this->Attr("scheduler_mode").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig config_310p = Get310PCoreConfig();
        this->AICore().AddConfig("ascend310p", config_310p);

        OpAICoreConfig config_310b = Get310BCoreConfig();
        this->AICore().AddConfig("ascend310b", config_310b);
    }
private:
    OpAICoreConfig Get310PCoreConfig()
    {
        OpAICoreConfig config_310p;
        // input
        config_310p.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        config_310p.Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // output
        config_310p.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        config_310p.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        return config_310p;
    }

    OpAICoreConfig Get310BCoreConfig()
    {
        OpAICoreConfig config_310b;
        // input
        config_310b.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        config_310b.Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // output
        config_310b.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        config_310b.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        return config_310b;
    }
};

OP_ADD(GridSample);
}  // namespace ops