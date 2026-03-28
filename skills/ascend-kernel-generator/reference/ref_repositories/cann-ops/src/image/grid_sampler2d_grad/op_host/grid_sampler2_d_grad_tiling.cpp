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
 * \file grid_sampler2_d_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include <map>
#include <vector>

#include "register/op_def_registry.h"
#include "grid_sampler_2d_grad_tiling_data.h"
#include "tiling/tiling_api.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
    do {                                                      \
        std::printf("op[%s], %s", op_name, err_msg);          \
    } while (0)
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
}  // namespace optiling

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr uint32_t FP16_BLOCK_NUM = 16;
constexpr size_t INTERPOLATION_MODE_INDEX = 0;
constexpr size_t PADDING_MODE_INDEX = 1;
constexpr size_t ALIGN_CORNERS_INDEX = 2;
constexpr int32_t GRAD_INPUT_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 1;
constexpr int32_t GRID_INPUT_INDEX = 2;
constexpr int32_t DTYPE_SIZE_32 = 4;
constexpr int32_t DTYPE_SIZE_16 = 2;
constexpr size_t CHECK_DIM_NUM = 4;
constexpr int BILINEAR = 0;
constexpr int NEAREST = 1;
constexpr int BICUBIC = 2;
constexpr int ZEROS = 0;
constexpr int BORDER = 1;
constexpr int REFLECTION = 2;
constexpr int TILINGKEY_2 = 2;
constexpr int BILINEAR_DIVIDE_UB_NUM = 57;
constexpr int NEAREST_DIVIDE_UB_NUM = 27;
constexpr int CAST_DIVIDE_UB_NUM = 3;
constexpr uint32_t FP32_GROUP_SIZE_LT_256 = 32;
constexpr uint32_t FP32_GROUP_SIZE_GT_256_LT_512 = 16;
constexpr uint32_t FP32_GROUP_SIZE_GT_512_LT_1024 = 8;
constexpr uint32_t FLOAT_BILINEAR_TILING_KEY = 1;
constexpr uint32_t FLOAT_NEAREST_TILING_KEY = 2;
constexpr uint32_t FLOAT16_BILINEAR_TILING_KEY = 3;
constexpr uint32_t FLOAT16_NEAREST_TILING_KEY = 4;
constexpr uint32_t BFLOAT16_BILINEAR_TILING_KEY = 5;
constexpr uint32_t BFLOAT16_NEAREST_TILING_KEY = 6;
constexpr uint32_t CHANNEL_256 = 256;
constexpr uint32_t CHANNEL_512 = 512;
constexpr uint32_t CHANNEL_1024 = 1024;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t CONST_SEVENTEEN = 17;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t RESERVED_UB = 2 * 1024;
constexpr uint32_t RESERVED_UB_CAST = 20 * 1024;
constexpr uint32_t ALIGN_256_BYTES = 256;
static std::map<std::string, int> INTER_MODE_MAP = {{"bilinear", 0}, {"nearest", 1}, {"bicubic", 2}};
static std::map<std::string, int> PADDING_MODE_MAP = {{"zeros", 0}, {"border", 1}, {"reflection", 2}};
static std::map<bool, int> ALIGN_MODE_MAP = {{true, 1}, {false, 0}};
static std::map<ge::DataType, uint64_t> TILINGKEY_MAP = {{ge::DT_FLOAT16, 1}, {ge::DT_FLOAT, 2}};

template <typename TilingData, int32_t dataTypeLen>
class GridSampler2DGradTiling {
public:
    explicit GridSampler2DGradTiling(InputParamsInfo &param, const uint32_t inputCoreNum, const uint32_t inputUbSize)
    {
        this->batch = param.batch;
        this->coreNum = inputCoreNum;
        this->channel = param.channel;
        this->height = param.height;
        this->width = param.width;
        this->gridH = param.gridH;
        this->gridW = param.gridW;
        this->interpolation = param.interpolation;
        this->padding = param.padding;
        this->alignCorners = param.alignCorners;
        this->tilingKey = param.tilingKey;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        return;
    }

    void GetTiling(TilingData *tilingData);

private:
    void GetUsedCore();
    void GetUsedCoreCast();
    void SplitUb();
    void FillTilingData(TilingData *tilingData);
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        return b == 0 ? a : (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline T1 FloorDiv(T1 a, T2 b)
    {
        return b == 0 ? a : (a) / (b);
    }
    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline T1 FloorAlign(T1 a, T2 b)
    {
        return b == 0 ? a : (a) / b * b;
    }

private:
    uint32_t batch = 0;
    uint32_t usedCoreNum = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t usedCoreNumCast = 0;
    uint32_t pNumPerCoreCast = 0;
    uint32_t tailPNumCast = 0;
    uint32_t castElement = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t gridH;
    uint32_t gridW;
    uint32_t ubSize = 0;
    uint32_t coreNum = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t alignCorners = 0;
    uint32_t group = 0;
    uint32_t tilingKey = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t divideUbNum = 1;
    uint32_t extraUbSize = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::GetUsedCore()
{
    uint64_t mulBHW = batch * gridH * gridW;
    if (mulBHW <= this->coreNum) {
        this->usedCoreNum = mulBHW;
        this->pNumPerCore = 1;
        this->tailPNum = 0;
        return;
    }
    this->pNumPerCore = FloorDiv(mulBHW, this->coreNum);
    this->usedCoreNum = this->coreNum;
    this->tailPNum = mulBHW % usedCoreNum;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::GetUsedCoreCast()
{
    size_t size = batch * channel * height * width;
    if (size <= this->usedCoreNum) {
        this->usedCoreNumCast = size;
        this->pNumPerCoreCast = 1;
        this->tailPNumCast = 0;
        return;
    }
    this->pNumPerCoreCast = FloorDiv(size, this->usedCoreNum);
    this->usedCoreNumCast = this->usedCoreNum;
    this->tailPNumCast = size % usedCoreNumCast;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::SplitUb()
{
    uint32_t alignChannel = 0;
    if (tilingKey <= TILINGKEY_2) {
        alignChannel = CeilAlign(channel, FP32_BLOCK_NUM);
    } else {
        alignChannel = CeilAlign(channel, FP16_BLOCK_NUM);
    }
    if (interpolation == 0) {
        divideUbNum = BILINEAR_DIVIDE_UB_NUM;
        extraUbSize = CONST_SEVENTEEN * alignChannel * DTYPE_SIZE_32;
        group = 1;
    } else if (interpolation == 1) {
        divideUbNum = NEAREST_DIVIDE_UB_NUM;
        if (channel <= CHANNEL_256) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_LT_256 + 1) * alignChannel * DTYPE_SIZE_32;
            group = FP32_GROUP_SIZE_LT_256;
        } else if (channel <= CHANNEL_512) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_256_LT_512 + 1) * alignChannel * DTYPE_SIZE_32;
            group = FP32_GROUP_SIZE_GT_256_LT_512;
        } else if (channel <= CHANNEL_1024) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_512_LT_1024 + 1) * alignChannel * DTYPE_SIZE_32;
            group = FP32_GROUP_SIZE_GT_512_LT_1024;
        } else {
            extraUbSize = BUFFER_NUM * CONST_TWO * alignChannel * DTYPE_SIZE_32;
            group = 1;
        }
    }
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    if (canUseUbSize <= extraUbSize) {
        ubFactorElement = 0;
        return;
    }
    ubFactorElement = FloorAlign((canUseUbSize - extraUbSize) / divideUbNum, ALIGN_256_BYTES) / DTYPE_SIZE_32;
    castElement = (canUseUbSize - RESERVED_UB_CAST) / CAST_DIVIDE_UB_NUM / DTYPE_SIZE_16;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::FillTilingData(TilingData *tilingData)
{
    tilingData->set_batch(batch);
    tilingData->set_pNumPerCore(pNumPerCore);
    tilingData->set_tailPNum(tailPNum);
    tilingData->set_channel(channel);
    tilingData->set_height(height);
    tilingData->set_width(width);
    tilingData->set_gridH(gridH);
    tilingData->set_gridW(gridW);
    tilingData->set_blockNum(usedCoreNum);
    tilingData->set_ubFactorElement(ubFactorElement);
    tilingData->set_interpolation(interpolation);
    tilingData->set_padding(padding);
    tilingData->set_alignCorners(alignCorners);
    tilingData->set_group(group);
    tilingData->set_tilingKey(tilingKey);
    tilingData->set_usedCoreNumCast(usedCoreNumCast);
    tilingData->set_pNumPerCoreCast(pNumPerCoreCast);
    tilingData->set_tailPNumCast(tailPNumCast);
    tilingData->set_castElement(castElement);
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler2DGradTiling<TilingData, dataTypeLen>::GetTiling(TilingData *tilingData)
{
    GetUsedCore();
    GetUsedCoreCast();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData, int32_t dataTypeLen>
void GetGridSampler2DGradTiling(TilingData *tilingData, InputParamsInfo &params, uint32_t coreNum, uint32_t ubSize)
{
    class GridSampler2DGradTiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData);
}

static void PrintTilingData(gert::TilingContext *tilingContext, GridSampler2DGradTilingData &tilingData)
{
    OP_LOGI(tilingContext->GetNodeName(), "Start printing");
    OP_LOGI(tilingContext->GetNodeName(), "batch is %u.", tilingData.get_batch());
    OP_LOGI(tilingContext->GetNodeName(), "channel is %u.", tilingData.get_channel());
    OP_LOGI(tilingContext->GetNodeName(), "height is %u.", tilingData.get_height());
    OP_LOGI(tilingContext->GetNodeName(), "width is %u.", tilingData.get_width());
    OP_LOGI(tilingContext->GetNodeName(), "gridH is %u.", tilingData.get_gridH());
    OP_LOGI(tilingContext->GetNodeName(), "gridW is %u.", tilingData.get_gridW());
    OP_LOGD(tilingContext->GetNodeName(), "blockNum is %u.", tilingData.get_blockNum());
    OP_LOGI(tilingContext->GetNodeName(), "pNumPerCore is %u.", tilingData.get_pNumPerCore());
    OP_LOGI(tilingContext->GetNodeName(), "tailPNum is %u.", tilingData.get_tailPNum());
    OP_LOGI(tilingContext->GetNodeName(), "ubFactorElement is %u.", tilingData.get_ubFactorElement());
    OP_LOGI(tilingContext->GetNodeName(), "interpolation is %u.", tilingData.get_interpolation());
    OP_LOGI(tilingContext->GetNodeName(), "padding is %u.", tilingData.get_padding());
    OP_LOGI(tilingContext->GetNodeName(), "alignCorners is %u.", tilingData.get_alignCorners());
    OP_LOGI(tilingContext->GetNodeName(), "group is %u.", tilingData.get_group());
    OP_LOGI(tilingContext->GetNodeName(), "tilingKey is %u.", tilingData.get_tilingKey());
    OP_LOGI(tilingContext->GetNodeName(), "usedCoreNumCast is %u.", tilingData.get_usedCoreNumCast());
    OP_LOGI(tilingContext->GetNodeName(), "pNumPerCoreCast is %u.", tilingData.get_pNumPerCoreCast());
    OP_LOGI(tilingContext->GetNodeName(), "tailPNumCast is %u.", tilingData.get_tailPNumCast());
    OP_LOGI(tilingContext->GetNodeName(), "castElement is %u.", tilingData.get_castElement());
    OP_LOGI(tilingContext->GetNodeName(), "End printing");
}

static ge::graphStatus GetInputInfo(gert::TilingContext *tilingContext, InputParamsInfo &params, ge::DataType dtype)
{
    OP_LOGI(tilingContext->GetNodeName(), "strat to get input dims");
    const gert::StorageShape *gradShape = tilingContext->GetInputShape(GRAD_INPUT_INDEX);
    OP_TILING_CHECK((gradShape == nullptr),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Get StorageShape Failed."),
        return false);
    const gert::StorageShape *xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    OP_TILING_CHECK((xShape == nullptr),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Get xShape Failed."),
        return false);
    const gert::StorageShape *gridShape = tilingContext->GetInputShape(GRID_INPUT_INDEX);
    OP_TILING_CHECK((gridShape == nullptr),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Get gridShape Failed."),
        return false);
    if (xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM) {
        OP_LOGD(tilingContext->GetNodeName(), "input dim is not 4, please check input");
        return ge::GRAPH_FAILED;
    }
    uint32_t outH = gradShape->GetStorageShape().GetDim(DIM_INDEX1);
    uint32_t outW = gradShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.height = xShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.width = xShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.gridH = gridShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.gridW = gridShape->GetStorageShape().GetDim(DIM_INDEX2);
    if (outH != params.gridH || outW != params.gridW) {
        OP_LOGW(tilingContext->GetNodeName(), "Please check grad's dims and grid's dims");
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    OP_TILING_CHECK((attrs == nullptr),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Get attrs Failed."),
        return false);
    const std::string interpolationMode = std::string(attrs->GetAttrPointer<char>(INTERPOLATION_MODE_INDEX));
    if (interpolationMode != "bilinear" && interpolationMode != "nearest") {
        OP_LOGW(tilingContext->GetNodeName(), "%s is not supported", interpolationMode.c_str());
        return ge::GRAPH_FAILED;
    }
    const std::string paddingMode = std::string(attrs->GetAttrPointer<char>(PADDING_MODE_INDEX));
    if (paddingMode != "zeros" && paddingMode != "border") {
        OP_LOGW(tilingContext->GetNodeName(), "%s is not supported", paddingMode.c_str());
        return ge::GRAPH_FAILED;
    }
    bool alignCorners = *tilingContext->GetAttrs()->GetAttrPointer<bool>(ALIGN_CORNERS_INDEX);
    params.interpolation = INTER_MODE_MAP[interpolationMode];
    params.padding = PADDING_MODE_MAP[paddingMode];
    params.alignCorners = ALIGN_MODE_MAP[alignCorners];

    size_t xWorkspaceSize = params.batch * params.channel * params.height * params.width * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;

    if (dtype == ge::DT_FLOAT && params.interpolation == 0) {
        params.tilingKey = FLOAT_BILINEAR_TILING_KEY;  // mode1: float, bilinear
    } else if (dtype == ge::DT_FLOAT && params.interpolation == 1) {
        params.tilingKey = FLOAT_NEAREST_TILING_KEY;  // mode2: float, nearest
    } else if (dtype == ge::DT_FLOAT16 && params.interpolation == 0) {
        sysWorkspaceSize += xWorkspaceSize;
        params.tilingKey = FLOAT16_BILINEAR_TILING_KEY;  // mode1: float16, bilinear
    } else if (dtype == ge::DT_FLOAT16 && params.interpolation == 1) {
        sysWorkspaceSize += xWorkspaceSize;
        params.tilingKey = FLOAT16_NEAREST_TILING_KEY;  // mode2: float16, nearest
    } else if (dtype == ge::DT_BF16 && params.interpolation == 0) {
        sysWorkspaceSize += xWorkspaceSize;
        params.tilingKey = BFLOAT16_BILINEAR_TILING_KEY;  // mode1: bfloat16, bilinear
    } else if (dtype == ge::DT_BF16 && params.interpolation == 1) {
        sysWorkspaceSize += xWorkspaceSize;
        params.tilingKey = BFLOAT16_NEAREST_TILING_KEY;  // mode2: bfloat16, nearest
    }
    size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_TILING_CHECK((currentWorkspace == nullptr),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Get currentWorkspace Failed."),
        return false);
    currentWorkspace[0] = sysWorkspaceSize;
    OP_LOGI(tilingContext->GetNodeName(), "sysWorkspaceSize is %zu.", sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GridSampler2DGrad(gert::TilingContext *tilingContext)
{
    OP_LOGI(tilingContext->GetNodeName(), "GridSampler2DGrad tiling starts running");
    // get corenum and ubsize
    auto compileInfo = reinterpret_cast<const Tiling4GridSampler2DGradCompileInfo *>(tilingContext->GetCompileInfo());
    uint64_t ubSizePlatForm = compileInfo->ubSizePlatForm;
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;
    uint32_t coreNum = 0;

    OP_LOGI(tilingContext->GetNodeName(), "ubSizePlatForm:%lu, coreNum:%u", ubSizePlatForm, coreNum);
    ge::DataType inputDatatype = tilingContext->GetInputDesc(0)->GetDataType();
    if (inputDatatype == ge::DT_FLOAT) {
        coreNum = compileInfo->coreNum;
    } else {
        coreNum = 1;
    }
    tilingContext->SetNeedAtomic(true);

    InputParamsInfo params;
    if (GetInputInfo(tilingContext, params, inputDatatype) != ge::GRAPH_SUCCESS) {
        OP_LOGW(tilingContext->GetNodeName(), "Failed to Parse input params , please check inputs");
        return ge::GRAPH_FAILED;
    }
    GridSampler2DGradTilingData tilingData;
    if (inputDatatype == ge::DT_FLOAT16) {
        GetGridSampler2DGradTiling<GridSampler2DGradTilingData, DTYPE_SIZE_16>(
            &tilingData, params, coreNum, availableUb);
    } else if (inputDatatype == ge::DT_FLOAT) {
        GetGridSampler2DGradTiling<GridSampler2DGradTilingData, DTYPE_SIZE_32>(
            &tilingData, params, coreNum, availableUb);
    } else if (inputDatatype == ge::DT_BF16) {
        GetGridSampler2DGradTiling<GridSampler2DGradTilingData, DTYPE_SIZE_16>(
            &tilingData, params, coreNum, availableUb);
    }
    OP_TILING_CHECK(tilingData.get_ubFactorElement() <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "ub space is not enough, please check input."),
        return ge::GRAPH_FAILED);
    // set tilingdata
    tilingContext->SetTilingKey(params.tilingKey);
    tilingContext->SetBlockDim(tilingData.get_blockNum());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    PrintTilingData(tilingContext, tilingData);
    OP_LOGI(tilingContext->GetNodeName(), "GridSampler2DGrad tiling end running");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GridSampler2DGrad(gert::TilingParseContext *context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4GridSampler2DGrad start.");
    auto compileInfo = GetCompileInfoPtr<Tiling4GridSampler2DGradCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->coreNum <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_TILING_CHECK((compileInfo->ubSizePlatForm <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "TilingPrepare4GridSampler2DGrad end.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GridSampler2DGrad)
    .Tiling(Tiling4GridSampler2DGrad)
    .TilingParse<Tiling4GridSampler2DGradCompileInfo>(TilingPrepare4GridSampler2DGrad);

}  // namespace optiling
