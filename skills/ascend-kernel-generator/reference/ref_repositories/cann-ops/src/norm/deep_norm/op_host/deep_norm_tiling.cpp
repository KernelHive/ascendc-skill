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
 * \file deep_norm_tiling.cpp
 * \brief
 */
#include "deep_norm_tiling.h"
#include <iostream>
#include <vector>
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGE(op_name, ...)            \
    std::printf(op_name, ##__VA_ARGS__); \
    std::printf("\n")

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

constexpr uint32_t BLOCK_BASE_NUM = 16;
constexpr size_t MAX_DIM_X = 8;
constexpr size_t MIN_DIM_X = 2;
constexpr size_t MAX_DIM_GAMMA = 7;
constexpr size_t MIN_DIM_GAMMA = 1;

constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_GX_INDEX = 1;
constexpr int32_t INPUT_BETA_INDEX = 2;
constexpr int32_t INPUT_GAMMA_INDEX = 3;
constexpr int32_t OUTPUT_MEAN_INDEX = 0;
constexpr int32_t OUTPUT_RSTD_INDEX = 1;
constexpr int32_t OUTPUT_Y_INDEX = 2;
constexpr uint32_t TILING_ISSHORT_OFFSET = 16;
constexpr uint32_t TILING_UPPER_LIMIT_OFFSET = 8;
constexpr uint32_t TILING_BEYOND_LIMIT_OFFSET = 4;
constexpr uint32_t TILING_ISFP32_OFFSET = 2;
constexpr uint32_t TILING_ISFP16_OFFSET = 1;

static uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

static uint32_t ROUND_UP(uint32_t x, uint32_t blockNumEl)
{
    return (x + blockNumEl - 1) / blockNumEl * blockNumEl;
}

static bool CheckInputOutputShapeDim(const gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *gxShape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::StorageShape *betaShape = context->GetInputShape(INPUT_BETA_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *meanShape = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    const gert::StorageShape *rstdShape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gxShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gxDimNum = gxShape->GetStorageShape().GetDimNum();
    size_t betaDimNum = betaShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t meanDimNum = meanShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();

    // Check shape dim range
    OP_TILING_CHECK((xDimNum > MAX_DIM_X) || (xDimNum < MIN_DIM_X),
        OP_LOGE(
            context->GetNodeName(), "Input x shape invaild, dim num should in range[%lu, %lu].", MIN_DIM_X, MAX_DIM_X),
        return false);
    OP_TILING_CHECK((gammaDimNum > MAX_DIM_GAMMA) || (gammaDimNum < MIN_DIM_GAMMA),
        OP_LOGE(context->GetNodeName(),
            "Input gamma shape invaild, dim num should in range[%lu, %lu].",
            MIN_DIM_GAMMA,
            MAX_DIM_GAMMA),
        return false);
    // Check shape dim relationship
    OP_TILING_CHECK(gxDimNum != xDimNum,
        OP_LOGE(context->GetNodeName(), "Input gx shape invaild, dim num is not equal x dim."),
        return false);
    OP_TILING_CHECK((yDimNum != xDimNum) || (meanDimNum != xDimNum) || (rstdDimNum != xDimNum),
        OP_LOGE(context->GetNodeName(), "Output y/mean/rstd shape invaild, dim num is not equal x dim."),
        return false);
    OP_TILING_CHECK(betaDimNum != gammaDimNum,
        OP_LOGE(context->GetNodeName(), "Input beta shape invaild, dim num is not equal gamma dim."),
        return false);
    OP_TILING_CHECK(xDimNum <= gammaDimNum,
        OP_LOGE(context->GetNodeName(), "x dim num should not be smaller than gamma dim num."),
        return false);
    return true;
}

static bool CheckInputOutputShapeValue(const gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *gxShape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::StorageShape *betaShape = context->GetInputShape(INPUT_BETA_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *meanShape = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    const gert::StorageShape *rstdShape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();

    // Check shape value
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_TILING_CHECK(xShape->GetStorageShape().GetDim(i) == 0,
            OP_LOGE(context->GetNodeName(), "Input x shape can not be 0."),
            return false);
        OP_TILING_CHECK(gxShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i),
            OP_LOGE(context->GetNodeName(), "Input gx shape invaild, shape is not equal x shape."),
            return false);
        OP_TILING_CHECK((yShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Input y shape invaild, shape is not equal x shape."),
            return false);
    }
    for (uint32_t i = 0; i < xDimNum - gammaDimNum; i++) {
        OP_TILING_CHECK((rstdShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)) ||
                            (meanShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Output rstd/mean shape invaild, shape is not equal x first few dim."),
            return false);
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_TILING_CHECK(
            (gammaShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(xDimNum - gammaDimNum + i)) ||
                (betaShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(xDimNum - gammaDimNum + i)),
            OP_LOGE(context->GetNodeName(), "Input gamma shape invaild, gamma shape is not equal x last few dim."),
            return false);
    }
    return true;
}

static void SetTilingKey4DeepNorm(gert::TilingContext *context, uint32_t &numCol, uint32_t &shortLimit,
    uint32_t &limitLastDim, uint32_t &limitLastDim2, ge::DataType &dataType)
{
    uint32_t isShort = numCol <= shortLimit ? 1 : 0;
    uint32_t upperLimit = limitLastDim2 < numCol ? 1 : 0;
    uint32_t beyondLimit = limitLastDim < numCol ? 1 : 0;
    uint32_t isFP32 = dataType == ge::DT_FLOAT ? 1 : 0;
    uint32_t isFP16 = dataType == ge::DT_FLOAT16 ? 1 : 0;
    //       D > 15360/8192     D > 4096      D <= 4096   D <= 100
    // fp32:    1110:14         0110:6         0010:2      10010:18
    // fp16:    1101:13         0101:5         0001:1      10001:17
    // bf16:    1100:12         0100:4         0000:0      10000:16
    uint32_t dtypeKey = isShort * TILING_ISSHORT_OFFSET + upperLimit * TILING_UPPER_LIMIT_OFFSET +
                        beyondLimit * TILING_BEYOND_LIMIT_OFFSET + isFP32 * TILING_ISFP32_OFFSET +
                        isFP16 * TILING_ISFP16_OFFSET;
    context->SetTilingKey(dtypeKey);
}

static ge::graphStatus Tiling4DeepNorm(gert::TilingContext *context)
{
    DeepNormTilingData tiling;
    OP_TILING_CHECK(!CheckInputOutputShapeDim(context),
        OP_LOGE(context->GetNodeName(), "Input shape dim invalid."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputOutputShapeValue(context),
        OP_LOGE(context->GetNodeName(), "Input shape value invalid."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t maxUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    // Get basic info
    const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
    size_t gammaIndex = 3;
    const gert::Shape gammaShape = context->GetInputShape(gammaIndex)->GetStorageShape();
    uint32_t numCol = gammaShape.GetShapeSize();

    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    int32_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }

    auto dataType = context->GetInputDesc(0)->GetDataType();

    uint32_t bufferNum = 1;
    // buffer num = 1
    uint32_t limitLastDim = 4096;
    uint32_t fp16Limit = 15360;
    uint32_t fp32Limit = 8192;
    // when D > 500  Loop N is better than loop D
    uint32_t shortLimit = 500;
    uint32_t limitLastDim2 = dataType == ge::DT_FLOAT ? fp32Limit : fp16Limit;
    uint32_t doubleBuffer = 2;

    if (bufferNum == doubleBuffer) {
        // buffer num = 2
        uint32_t baseLimit = 4096;
        uint32_t fp16LimitDb = 13312;
        uint32_t fp32LimitDb = 3072;
        limitLastDim = dataType == ge::DT_FLOAT ? fp32LimitDb : baseLimit;
        limitLastDim2 = dataType == ge::DT_FLOAT ? fp32LimitDb : fp16LimitDb;
    }
    uint32_t numCore = CEIL_DIV(numRow, CEIL_DIV(numRow, maxCoreNum));
    uint32_t rowWork = CEIL_DIV(numRow, numCore);
    uint32_t lFirstdimPerCoreNum = numRow - rowWork * (numCore - 1);

    float tempAlpha = *context->GetAttrs()->GetFloat(0);
    float tempAve = numCol == 0 ? 1 : float(1.0 / numCol);
    float eps = *context->GetAttrs()->GetFloat(1);

    // About tiling
    uint32_t usedLastDim = numCol;
    if (limitLastDim < numCol) {
        uint32_t blockNum = CEIL_DIV(numCol, limitLastDim);
        usedLastDim = CEIL_DIV(numCol, blockNum * BLOCK_BASE_NUM) * BLOCK_BASE_NUM;
        tiling.set_updated_last_dim(usedLastDim);
        tiling.set_updated_last_times(blockNum);
    } else {
        tiling.set_updated_last_dim(0);
        tiling.set_updated_last_times(0);
    }

    int32_t byteNum = dataType == ge::DT_FLOAT ? 4 : 2;
    int32_t blockNum = 32 / byteNum;
    int32_t maxEleNum = maxUbSize / byteNum;
    int32_t dynDataUsed = 2;              // tensor from gm
    int32_t staticDataUsed = 3;           // local tensor
    int32_t queDataUsed = 2 * bufferNum;  // local tensor
    int32_t dynLastDim = limitLastDim2 < numCol ? usedLastDim : numCol;
    int32_t scalarUsed = 50;
    int32_t numTempBuf = 64;
    int32_t isSmallType = dataType != ge::DT_FLOAT && limitLastDim >= numCol ? 1 : 0;
    int32_t isShortCase = numCol <= shortLimit ? 1 : 0;
    int32_t totalMemNeed =
        (dynDataUsed * ROUND_UP(usedLastDim, blockNum) + ROUND_UP(dynLastDim, blockNum) + 64 / byteNum) * rowWork *
            bufferNum +
        isSmallType * (1 - isShortCase) * 4 * ROUND_UP(usedLastDim, blockNum) * rowWork +
        isShortCase * isSmallType * staticDataUsed * ROUND_UP(usedLastDim, blockNum) * 4 / byteNum * rowWork;
    int32_t sumData = maxEleNum - numTempBuf - scalarUsed - queDataUsed * ROUND_UP(usedLastDim, blockNum) -
                      (1 - isShortCase) * staticDataUsed * ROUND_UP(usedLastDim, blockNum) * 4 / byteNum -
                      ROUND_UP(dynLastDim, blockNum) * 4 / byteNum;
    uint32_t firstDimPerTime = 0;
    if (limitLastDim < numCol) {
        firstDimPerTime = 1;
    } else if (totalMemNeed > sumData) {
        uint32_t timeCopyIn = CEIL_DIV(totalMemNeed, sumData);
        if (timeCopyIn > 0) {
            firstDimPerTime = rowWork / timeCopyIn;
        }
    } else {
        firstDimPerTime = rowWork;
    }
    uint32_t maxRepeat = 255;
    if (numCol <= shortLimit) {
        firstDimPerTime = firstDimPerTime > maxRepeat ? maxRepeat : firstDimPerTime;
    }
    tiling.set_num_core(numCore);
    tiling.set_num_last_dim(numCol);
    tiling.set_num_first_dim(numRow);
    tiling.set_nl_firstdim_per_core(rowWork);
    tiling.set_l_firstdim_per_core(lFirstdimPerCoreNum);
    tiling.set_first_dim_per_times(firstDimPerTime);
    tiling.set_eps_str(*reinterpret_cast<uint32_t *>(&eps));
    tiling.set_ave_str(*reinterpret_cast<uint32_t *>(&tempAve));
    tiling.set_alpha_str(*reinterpret_cast<uint32_t *>(&tempAlpha));
    context->SetBlockDim(numCore);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    SetTilingKey4DeepNorm(context, numCol, shortLimit, limitLastDim, limitLastDim2, dataType);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 1;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DeepNorm(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

struct DeepNormCompileInfo {};
IMPL_OP_OPTILING(DeepNorm).Tiling(Tiling4DeepNorm).TilingParse<DeepNormCompileInfo>(TilingPrepare4DeepNorm);
}  // namespace optiling
