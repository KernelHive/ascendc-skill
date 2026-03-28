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
 * \file dua_quantize_add_layer_norm_tiling.cpp
 * \brief
 */
#include "dua_quantize_add_layer_norm_tiling.h"
#include <iostream>
#include <vector>
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
namespace optiling {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGE(op_name, ...)            \
    std::printf(op_name, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGW(op_name, ...)            \
    std::printf(op_name, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_CHECK(cond, log_func, return_expr) \
    if (cond) {                               \
        log_func;                             \
        return_expr;                          \
    }

}  // namespace optiling

namespace optiling {

static constexpr uint64_t TILING_BASE = 1000;
static constexpr uint64_t TILING_MUL_MODE = 1;
static constexpr int64_t AXIS_VALUE_FOR_MUL_MODE = -65535;
static constexpr int64_t BLOCK_SIZE = 32;

static constexpr size_t AXIS_IDX = 1;
static constexpr size_t EPS_IDX = 2;

static constexpr size_t ZERO_POINTS_1_IDX = 7;
static constexpr size_t ZERO_POINTS_2_IDX = 8;

inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

bool CheckUbLimit(uint64_t ubSize, int32_t numCol, int32_t dtSize)
{
    int32_t blockNum = BLOCK_SIZE / dtSize;
    int32_t numColAligned = CEIL_DIV(numCol, blockNum) * blockNum;

    // 1 inQue, 1 outQue, 3 Tbuf
    uint32_t ubRequired =
        (1 + 1) * numColAligned * dtSize + numColAligned * sizeof(int8_t) + 3 * numColAligned * sizeof(float);

    OP_LOGI("CheckUbLimit", "maxUB: %lu, ubRequired: %u", ubSize, ubRequired);
    OP_CHECK(ubRequired >= ubSize,
        OP_LOGW("DuaQuantizeAddLayerNorm", "Reduce axis is too large, Tiling is not support."),
        return false);
    return true;
}

static inline void GetSocInfos(gert::TilingContext *context, uint64_t &ubSize, uint32_t &maxCoreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    maxCoreNum = ascendcPlatform.GetCoreNumAiv();
}

static inline void GetAndSetAttrs(
    gert::TilingContext *context, DuaQuantizeAddLayerNormTilingData *tiling, uint64_t &tilingKey)
{
    float eps = *context->GetAttrs()->GetFloat(EPS_IDX);
    int64_t axis = *context->GetAttrs()->GetInt(AXIS_IDX);
    OP_LOGD("DuaQuantizeAddLayerNorm", "eps: %f, axis: %ld", eps, axis);
    tiling->set_eps(eps);
    if (axis == AXIS_VALUE_FOR_MUL_MODE) {
        tilingKey += TILING_MUL_MODE;
    }
}

static inline void GetRowCols(gert::TilingContext *context, int32_t &numRow, int32_t &numLastDim)
{
    numRow = 1;
    for (size_t i = 0; i < context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1; i++) {
        numRow *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    numLastDim = context->GetInputShape(0)->GetStorageShape().GetDim(
        context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1);
}

static inline void SetOptInputs(gert::TilingContext *context, DuaQuantizeAddLayerNormTilingData *tiling)
{
    auto zeroPoint1Shape = context->GetOptionalInputShape(ZERO_POINTS_1_IDX);
    auto zeroPoint2Shape = context->GetOptionalInputShape(ZERO_POINTS_2_IDX);
    if (zeroPoint1Shape == nullptr) {
        tiling->set_isZeroPoint1Exist(0);
    } else {
        tiling->set_isZeroPoint1Exist(1);
    }
    if (zeroPoint2Shape == nullptr) {
        tiling->set_isZeroPoint2Exist(0);
    } else {
        tiling->set_isZeroPoint2Exist(1);
    }
}

static inline void SetTilingData(DuaQuantizeAddLayerNormTilingData *tiling, int32_t numRow, int32_t numLastDim,
    int64_t numCore, int64_t nlFirstDimPerCore)
{
    tiling->set_numCore(numCore);
    tiling->set_numRow(numRow);
    tiling->set_numLastDim(numLastDim);
    tiling->set_nlFirstDimPerCore(nlFirstDimPerCore);
    if (numRow % nlFirstDimPerCore == 0) {
        tiling->set_lFirstDimPerCore(nlFirstDimPerCore);
    } else {
        tiling->set_lFirstDimPerCore(numRow - (numCore - 1) * nlFirstDimPerCore);
    }
    tiling->set_firstDimPerTime(1);
    tiling->set_lastDimPerTime(1);
    float aveNum = 1.0f / numLastDim;
    tiling->set_aveNum(aveNum);
    tiling->set_colMoveCnt(1);
    tiling->set_colTail(1);

    OP_LOGD("DuaQuantizeAddLayerNorm",
        "numCore:%ld, numLastDim:%d, numRow:%d, nlFirstDimPerCore:%ld, aveNum:%f",
        numCore,
        numLastDim,
        numRow,
        nlFirstDimPerCore,
        aveNum);
}

static ge::graphStatus Tiling4DuaQuantizeAddLayerNorm(gert::TilingContext *context)
{
    OP_LOGD("DuaQuantizeAddLayerNorm", "Enter Tiling4DuaQuantizeAddLayerNorm tiling");
    DuaQuantizeAddLayerNormTilingData tiling;
    uint64_t ubSize = 0;
    uint32_t maxCoreNum = 0;
    GetSocInfos(context, ubSize, maxCoreNum);

    uint64_t tilingKey = TILING_BASE;
    GetAndSetAttrs(context, &tiling, tilingKey);

    int32_t numRow = 1;
    int32_t numLastDim = 1;
    GetRowCols(context, numRow, numLastDim);

    int64_t nlFirstDimPerCore = CEIL_DIV(numRow, maxCoreNum);
    int64_t numCore = CEIL_DIV(numRow, nlFirstDimPerCore);

    int32_t dataTypeSize = GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    OP_CHECK(!(CheckUbLimit(ubSize, numLastDim, dataTypeSize)),
        OP_LOGE("DuaQuantizeAddLayerNorm", "CheckUbLimit Failed"),
        return ge::GRAPH_FAILED);

    SetTilingData(&tiling, numRow, numLastDim, numCore, nlFirstDimPerCore);
    SetOptInputs(context, &tiling);

    context->SetBlockDim(numCore);
    context->SetTilingKey(tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t usrSize = 0;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DuaQuantizeAddLayerNorm(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

struct DuaQuantizeAddLayerNormCompileInfo {};

IMPL_OP_OPTILING(DuaQuantizeAddLayerNorm)
    .Tiling(Tiling4DuaQuantizeAddLayerNorm)
    .TilingParse<DuaQuantizeAddLayerNormCompileInfo>(TilingPrepare4DuaQuantizeAddLayerNorm);
}  // namespace optiling
