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
 * \file add_rms_norm_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "add_rms_norm_tiling.h"

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

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
}  // namespace optiling

namespace optiling {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t UB_FACTOR_B16 = 12288;
constexpr uint32_t UB_FACTOR_B32 = 10240;
constexpr uint32_t UB_FACTOR_B16_CUTD = 12096;
constexpr uint32_t UB_FACTOR_B32_CUTD = 9696;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t SMALL_REDUCE_NUM = 2000;
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t MODE_MERGE_N = 2;
constexpr uint32_t MODE_SINGLE_N = 3;
constexpr uint32_t MODE_MULTI_N = 4;
constexpr int32_t INPUT_X1_INDEX = 0;
constexpr int32_t INPUT_X2_INDEX = 1;
constexpr int32_t INPUT_GAMMA_INDEX = 2;
constexpr int32_t OUTPUT_Y_INDEX = 0;
constexpr int32_t OUTPUT_RSTD_INDEX = 1;
constexpr int32_t OUTPUT_X_INDEX = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;
constexpr uint32_t NUM_260 = 260;
constexpr uint32_t NUM_256 = 256;
constexpr uint32_t NUM_64 = 64;
constexpr uint32_t NUM_2 = 2;

static void SetByDtype(ge::DataType dataType, uint32_t &dtypeKey, uint32_t &dataPerBlock)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BLOCK_ALIGN_NUM;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BLOCK_ALIGN_NUM;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
            break;
    }
}

static bool CheckInputOutputDim(const gert::TilingContext *context)
{
    const gert::StorageShape *x1_shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2_shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);
    const gert::StorageShape *rstd_shape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape *x_shape = context->GetOutputShape(OUTPUT_X_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2_shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();
    size_t yDimNum = y_shape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstd_shape->GetStorageShape().GetDimNum();
    size_t xDimNum = x_shape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(x1DimNum > MAX_DIM_NUM || x1DimNum < MIN_DIM_X,
        OP_LOGE(context->GetNodeName(), "Input x1's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE(context->GetNodeName(), "Input gamma's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(x1DimNum != yDimNum,
        OP_LOGE(context->GetNodeName(), "Input x's dim num must equal to output y's dim num."),
        return false);

    OP_TILING_CHECK(x1DimNum != x2DimNum,
        OP_LOGE(context->GetNodeName(), "Input x2/x1 shape invaild, dim num is not equal x1 dim."),
        return false);
    OP_TILING_CHECK((yDimNum != xDimNum) || (xDimNum != x1DimNum) || (rstdDimNum != x1DimNum),
        OP_LOGE(context->GetNodeName(), "Output y/rstd/x shape invaild, dim num is not equal x1 dim."),
        return false);
    OP_TILING_CHECK(x1DimNum < gammaDimNum,
        OP_LOGE(context->GetNodeName(), "X1 dim num should not be smaller than gamma dim num."),
        return false);
    return true;
}

static bool CheckInputOutputShape(const gert::TilingContext *context)
{
    OP_TILING_CHECK(
        !CheckInputOutputDim(context), OP_LOGE(context->GetNodeName(), "Input Dim invalid."), return ge::GRAPH_FAILED);
    const gert::StorageShape *x1_shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2_shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);
    const gert::StorageShape *rstd_shape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape *x_shape = context->GetOutputShape(OUTPUT_X_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();

    for (uint32_t i = 0; i < x1DimNum; i++) {
        OP_TILING_CHECK(x2_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i),
            OP_LOGE(context->GetNodeName(), "Input x2/x1 shape invaild, shape is not equal x1 shape."),
            return false);
        OP_TILING_CHECK((y_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i)) ||
                            (x_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Input y/x shape invaild, shape is not equal x1 shape."),
            return false);
    }
    for (uint32_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        OP_TILING_CHECK(rstd_shape->GetStorageShape().GetDim(i) != x2_shape->GetStorageShape().GetDim(i),
            OP_LOGE(context->GetNodeName(), "Output rstd shape invaild, shape is not equal x1 first few dim."),
            return false);
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_TILING_CHECK(
            gamma_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(x1DimNum - gammaDimNum + i),
            OP_LOGE(context->GetNodeName(), "Input gamma shape invaild, gamma shape is not equal x1 last few dim."),
            return false);
        OP_TILING_CHECK(rstd_shape->GetStorageShape().GetDim(x1DimNum - 1 - i) != 1,
            OP_LOGE(context->GetNodeName(), "Output rstd shape invaild, last few dim is not equal to 1."),
            return false);
    }
    return true;
}

static void GetCompileParameters(
    gert::TilingContext *context, uint32_t &numCore, uint64_t &ubSize, platform_ascendc::SocVersion &socVersion)
{
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormCompileInfo *>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        socVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
        socVersion = ptrCompileInfo->socVersion;
    }
    ubSize -= UB_USED;
}

static void CalculateRowAndColParameters(gert::TilingContext *context, uint32_t &numRow, uint32_t &numCol)
{
    const gert::Shape x1_shape = context->GetInputShape(0)->GetStorageShape();
    const size_t gammaIndex = 2;
    const gert::Shape gamma_shape = context->GetInputShape(gammaIndex)->GetStorageShape();
    numCol = gamma_shape.GetShapeSize();

    const size_t x1DimNum = x1_shape.GetDimNum();
    const size_t gammaDimNum = gamma_shape.GetDimNum();
    numRow = 1;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; ++i) {
        numRow *= x1_shape.GetDim(i);
    }
}

static ge::graphStatus GetEpsilonParameter(gert::TilingContext *context, float &epsilon)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    epsilon = *attrs->GetFloat(0);
    OP_TILING_CHECK(epsilon < 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Epsilon less than zero, please check."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void CalculateBlockParameters(
    uint32_t numRow, uint32_t numCore, uint32_t &blockFactor, uint32_t &useCoreNum)
{
    blockFactor = 1;
    uint32_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    useCoreNum = CeilDiv(numRow, blockFactor);
}

static ge::DataType SetDataTypeParameters(gert::TilingContext *context, uint32_t &dtypeKey, uint32_t &dataPerBlock)
{
    auto data_type = context->GetInputDesc(0)->GetDataType();
    dtypeKey = DTYPE_KEY_FP16;
    SetByDtype(data_type, dtypeKey, dataPerBlock);
    return data_type;
}

static void DetermineModeParameters(uint32_t numCol, uint32_t &ubFactor, uint32_t &rowFactor, uint32_t blockFactor,
    platform_ascendc::SocVersion socVersion, ge::DataType data_type, uint32_t dtypeKey, uint64_t ubSize,
    uint32_t dataPerBlock, uint32_t &modeKey)
{
    const uint32_t numColAlign = CeilDiv(numCol, dataPerBlock) * dataPerBlock;

    if (numCol > ubFactor) {
        modeKey = MODE_SPLIT_D;
        ubFactor = (data_type == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD;
        uint32_t colTileNum = CeilDiv(numCol, ubFactor);
        ubFactor = CeilDiv(numCol, colTileNum * dataPerBlock) * dataPerBlock;
    } else if (blockFactor == 1 && socVersion != platform_ascendc::SocVersion::ASCEND310P) {
        modeKey = MODE_SINGLE_N;
    } else if (numColAlign <= SMALL_REDUCE_NUM && socVersion != platform_ascendc::SocVersion::ASCEND310P) {
        modeKey = MODE_MERGE_N;
        uint64_t numColAlignWeight = (dtypeKey == DTYPE_KEY_FP32) ? 24 : 18;
        rowFactor = ubSize / (numColAlign * numColAlignWeight + NUM_260);
        ubFactor = rowFactor * numColAlign;
    } else if (data_type == ge::DT_FLOAT16 && numCol == numColAlign) {
        modeKey = MODE_MULTI_N;
        rowFactor = (ubSize - NUM_256 - numColAlign * NUM_2) / (numColAlign * BLOCK_ALIGN_NUM + NUM_64);
        ubFactor = rowFactor * numColAlign;
        if (rowFactor == 0) {
            modeKey = MODE_NORMAL;
            rowFactor = NUM_64;
            ubFactor = UB_FACTOR_B16;
        }
    }
}

static void SetTilingParameters(AddRMSNormTilingData *tiling, uint32_t numRow, uint32_t numCol, uint32_t blockFactor,
    uint32_t rowFactor, uint32_t ubFactor, float epsilon)
{
    const float avg_factor = (numCol == 0) ? 0 : 1.0f / numCol;
    tiling->set_num_row(numRow);
    tiling->set_num_col(numCol);
    tiling->set_block_factor(blockFactor);
    tiling->set_row_factor(rowFactor);
    tiling->set_ub_factor(ubFactor);
    tiling->set_epsilon(epsilon);
    tiling->set_avg_factor(avg_factor);
}

static void SaveTilingData(gert::TilingContext *context, AddRMSNormTilingData *tiling, uint32_t dtypeKey,
    uint32_t modeKey, uint32_t shape_size)
{
    const uint32_t tiling_key = shape_size != 0 ? dtypeKey * 10 + modeKey : 0;
    context->SetTilingKey(tiling_key);
    tiling->SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static void SetWorkspaceSize(gert::TilingContext *context)
{
    constexpr size_t sysWorkspaceSize = 16 * 1024 * 1024;
    constexpr size_t usrSize = 256;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static void LogTilingResults(gert::TilingContext *context, AddRMSNormTilingData *tiling, uint32_t modeKey,
    uint32_t dtypeKey, uint32_t useCoreNum, float epsilon)
{
    OP_LOGI("Tiling4AddRmsNorm", "Tiling Key: %u", dtypeKey * 10 + modeKey);
    OP_LOGI("Tiling4AddRmsNorm", "Block Dim: %u", useCoreNum);
    OP_LOGI("Tiling4AddRmsNorm", "usr Workspace: 256");
    OP_LOGI("Tiling4AddRmsNorm",
        "numRow: %d, numCol: %d, blockFactor: %d, rowFactor: %d, ubFactor: %d, epsilon: %f, avg_factor: %f",
        tiling->get_num_row(),
        tiling->get_num_col(),
        tiling->get_block_factor(),
        tiling->get_row_factor(),
        tiling->get_ub_factor(),
        epsilon,
        tiling->get_avg_factor());
}

static ge::graphStatus Tiling4AddRmsNorm(gert::TilingContext *context)
{
    OP_LOGD("Tiling4AddRmsNorm", "Enter Tiling4AddRmsNorm");
    OP_TILING_CHECK(!CheckInputOutputShape(context),
        OP_LOGE(context->GetNodeName(), "Input shape invalid."),
        return ge::GRAPH_FAILED);

    AddRMSNormTilingData tiling;
    uint32_t numCore;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
    GetCompileParameters(context, numCore, ubSize, socVersion);

    uint32_t numRow;
    uint32_t numCol;
    CalculateRowAndColParameters(context, numRow, numCol);

    float epsilon;
    GetEpsilonParameter(context, epsilon);
    if (epsilon < 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockFactor;
    uint32_t useCoreNum;
    CalculateBlockParameters(numRow, numCore, blockFactor, useCoreNum);
    context->SetBlockDim(useCoreNum);

    uint32_t dtypeKey;
    uint32_t dataPerBlock;
    ge::DataType data_type = SetDataTypeParameters(context, dtypeKey, dataPerBlock);

    uint32_t modeKey = MODE_NORMAL;
    uint32_t rowFactor = 64;
    uint32_t ubFactor = (dtypeKey == DTYPE_KEY_FP32) ? UB_FACTOR_B32 : UB_FACTOR_B16;
    DetermineModeParameters(numCol,
        ubFactor,
        rowFactor,
        blockFactor,
        socVersion,
        data_type,
        dtypeKey,
        ubSize,
        dataPerBlock,
        modeKey);

    SetTilingParameters(&tiling, numRow, numCol, blockFactor, rowFactor, ubFactor, epsilon);
    SaveTilingData(context, &tiling, dtypeKey, modeKey, numRow * numCol);

    SetWorkspaceSize(context);

    LogTilingResults(context, &tiling, modeKey, dtypeKey, useCoreNum, epsilon);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddRmsNorm(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AddRmsNorm running.");
    auto compileInfo = GetCompileInfoPtr<AddRmsNormCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplaceAddRmsNorm)
    .Tiling(Tiling4AddRmsNorm)
    .TilingParse<AddRmsNormCompileInfo>(TilingPrepare4AddRmsNorm);

}  // namespace optiling