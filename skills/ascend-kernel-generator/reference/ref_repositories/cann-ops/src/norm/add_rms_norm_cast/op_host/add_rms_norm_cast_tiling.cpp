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
 * \file add_rms_norm_cast_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "add_rms_norm_cast_tiling.h"

// tools api
namespace optiling {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
}  // namespace optiling

namespace optiling {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t NUMBER_256 = 256;
constexpr uint32_t NUMBER_64 = 64;
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
constexpr int32_t OUTPUT_Y1_INDEX = 0;
constexpr int32_t OUTPUT_Y2_INDEX = 1;
constexpr int32_t OUTPUT_RSTD_INDEX = 2;
constexpr int32_t OUTPUT_X_INDEX = 3;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;

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

static bool CheckInputOutputShape(const gert::TilingContext *context)
{
    const gert::StorageShape *x1_shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2_shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *y1_shape = context->GetOutputShape(OUTPUT_Y1_INDEX);
    const gert::StorageShape *y2_shape = context->GetOutputShape(OUTPUT_Y2_INDEX);
    const gert::StorageShape *rstd_shape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape *x_shape = context->GetOutputShape(OUTPUT_X_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y1_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y2_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2_shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();
    size_t y1DimNum = y1_shape->GetStorageShape().GetDimNum();
    size_t y2DimNum = y2_shape->GetStorageShape().GetDimNum();
    size_t xDimNum = x_shape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(x1DimNum > MAX_DIM_NUM || x1DimNum < MIN_DIM_X,
        OP_LOGE(context->GetNodeName(), "Input x1's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE(context->GetNodeName(), "Input gamma's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(x1DimNum != y1DimNum,
        OP_LOGE(context->GetNodeName(), "Input x's dim num must equal to output y1's dim num."),
        return false);
    OP_TILING_CHECK(x1DimNum != y2DimNum,
        OP_LOGE(context->GetNodeName(), "Input x's dim num must equal to output y2's dim num."),
        return false);

    OP_TILING_CHECK(x1DimNum != x2DimNum,
        OP_LOGE(context->GetNodeName(), "Input x2/x1 shape invaild, dim num is not equal x1 dim."),
        return false);
    OP_TILING_CHECK((y1DimNum != xDimNum) || (xDimNum != x1DimNum) || (y2DimNum != xDimNum),
        OP_LOGE(context->GetNodeName(), "Output y/x shape invaild, dim num is not equal x1 dim."),
        return false);
    OP_TILING_CHECK(x1DimNum < gammaDimNum,
        OP_LOGE(context->GetNodeName(), "X1 dim num should not be smaller than gamma dim num."),
        return false);

    for (uint32_t i = 0; i < x1DimNum; i++) {
        OP_TILING_CHECK(x1_shape->GetStorageShape().GetDim(i) == 0,
            OP_LOGE(context->GetNodeName(), "Input x1 shape can not be 0."),
            return false);
        OP_TILING_CHECK(x2_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i),
            OP_LOGE(context->GetNodeName(), "Input x2/x1 shape invaild, shape is not equal x1 shape."),
            return false);
        OP_TILING_CHECK((y1_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i)) ||
                            (y2_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i)) ||
                            (x_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Output y1/y2/x shape invaild, shape is not equal x1 shape."),
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
    }
    return true;
}

static ge::graphStatus Tiling4AddRmsNormCast(gert::TilingContext *context)
{
    OP_LOGD("Tiling4AddRmsNormCast", "Enter Tiling4AddRmsNormCast");
    OP_TILING_CHECK(!CheckInputOutputShape(context),
        OP_LOGE(context->GetNodeName(), "Input shape invalid."),
        return ge::GRAPH_FAILED);
    AddRMSNormCastTilingData tiling;
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormCastCompileInfo *>(context->GetCompileInfo());
    uint32_t numCore;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
    if (nullptr == ptrCompileInfo) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        socVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
        socVersion = ptrCompileInfo->socVersion;
    }
    ubSize = ubSize - UB_USED;
    uint32_t ubFactor = UB_FACTOR_B16;

    const gert::Shape x1_shape = context->GetInputShape(0)->GetStorageShape();
    size_t gammaIndex = 2;
    std::string opType(context->GetNodeType());

    const gert::Shape gamma_shape = context->GetInputShape(gammaIndex)->GetStorageShape();

    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float *epsilon = attrs->GetFloat(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_TILING_CHECK(*epsilon < 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Epsilon less than zero, please check."),
        return ge::GRAPH_FAILED);
    uint32_t numCol = gamma_shape.GetShapeSize();
    float avgFactor = (numCol == 0) ? 0 : 1.0 / numCol;
    size_t x1DimNum = x1_shape.GetDimNum();
    size_t gammaDimNum = gamma_shape.GetDimNum();
    uint32_t numRow = 1;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        numRow *= x1_shape.GetDim(i);
    }

    OP_LOGD("Tiling4AddRmsNormCast", "Core Num: %u", numCore);

    uint32_t blockFactor = 1;
    uint32_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    OP_LOGD("Tiling4AddRmsNormCast", "tile num: %d", tileNum);
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

    context->SetBlockDim(useCoreNum);

    uint32_t rowFactor = 64;
    auto data_type = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = DTYPE_KEY_FP16;
    uint32_t dataPerBlock;
    SetByDtype(data_type, dtypeKey, dataPerBlock);

    uint32_t modeKey = MODE_NORMAL;  // 0: Normal, 1: SplitD, 2: MergeN, 3: SingleN 4: MultiN
    ubFactor = (dtypeKey == DTYPE_KEY_FP32) ? UB_FACTOR_B32 : UB_FACTOR_B16;

    uint32_t numColAlign = CeilDiv(numCol, dataPerBlock) * dataPerBlock;
    if (numCol > ubFactor) {
        modeKey = MODE_SPLIT_D;
        ubFactor = (data_type == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD;
        uint32_t colTileNum = CeilDiv(numCol, ubFactor);
        ubFactor = CeilDiv(numCol, colTileNum * dataPerBlock) * dataPerBlock;
    } else if (blockFactor == 1 && socVersion != platform_ascendc::SocVersion::ASCEND310P) {
        modeKey = MODE_SINGLE_N;
    } else if (data_type == ge::DT_FLOAT16 && numCol == numColAlign) {
        modeKey = MODE_MULTI_N;
        rowFactor = (ubSize - NUMBER_256 - numColAlign * MODE_MERGE_N) / (numColAlign * BLOCK_ALIGN_NUM + NUMBER_64);
        ubFactor = rowFactor * numColAlign;
        if (rowFactor == 0) {
            modeKey = MODE_NORMAL;
            rowFactor = NUMBER_64;
            ubFactor = UB_FACTOR_B16;
        }
    }

    uint32_t tilingKey = dtypeKey * 10 + modeKey;
    context->SetTilingKey(tilingKey);

    tiling.set_num_row(numRow);
    tiling.set_num_col(numCol);
    tiling.set_block_factor(blockFactor);
    tiling.set_row_factor(rowFactor);
    tiling.set_ub_factor(ubFactor);
    tiling.set_epsilon(*epsilon);
    tiling.set_avg_factor(avgFactor);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t usrSize = 256;
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    OP_LOGI("Tiling4AddRmsNormCast", "Tiling Key: %u", tilingKey);
    OP_LOGI("Tiling4AddRmsNormCast", "Block Dim: %u", useCoreNum);
    OP_LOGI("Tiling4AddRmsNormCast", "usr Workspace: %zu", usrSize);
    OP_LOGI("Tiling4AddRmsNormCast",
        "numRow: %d, numCol: %d, blockFactor: %d, rowFactor: %d, ubFactor: %d, epsilon: %f, avgFactor: %f",
        numRow,
        numCol,
        blockFactor,
        rowFactor,
        ubFactor,
        *epsilon,
        avgFactor);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddRmsNormCast(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AddRmsNormCast running.");
    auto compileInfo = GetCompileInfoPtr<AddRmsNormCastCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormCast)
    .Tiling(Tiling4AddRmsNormCast)
    .TilingParse<AddRmsNormCastCompileInfo>(TilingPrepare4AddRmsNormCast);
}  // namespace optiling