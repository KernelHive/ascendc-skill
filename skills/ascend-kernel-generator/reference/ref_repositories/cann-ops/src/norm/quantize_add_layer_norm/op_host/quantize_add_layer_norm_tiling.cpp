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
 * \file quantize_add_layer_norm_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "quantize_add_layer_norm_tiling.h"

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

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

}  // namespace optiling

namespace optiling {

static constexpr int64_t UB_RESERVED_BYTE = 256;
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr int64_t BIG_N_MAX_D = 500;
static constexpr int64_t MAX_REPEAT_TIMES = 255;
static constexpr int64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
static constexpr int64_t AXIS_VALUE_FOR_PER_TENSOR = 65535;
static constexpr int64_t AXIS_VALUE_FOR_MUL_DIV_MODE = -65535;

static constexpr uint64_t TILING_DATATYPE_FP16 = 1000;
static constexpr uint64_t TILING_DATATYPE_FP32 = 2000;
static constexpr uint64_t TILING_DATATYPE_BF16 = 3000;
static constexpr uint64_t TILING_ADDITIONAL_OUTPUT = 100;
static constexpr uint64_t TILING_SLICED = 10;
static constexpr uint64_t TILING_SINGLE_ROW = 20;
static constexpr uint64_t TILING_SINGLE_ROW_EXT = 30;
static constexpr uint64_t TILING_NORMAL_BIG_N = 40;
static constexpr uint64_t TILING_SLICE_EXT = 50;
static constexpr uint64_t TILING_NORMAL_SPECIAL = 60;
static constexpr uint64_t TILING_PER_CHANNEL = 0;
static constexpr uint64_t TILING_PER_TENSOR = 1;
static constexpr uint64_t TILING_MUL_MODE = 2;

const std::string OP_NAME = "QuantizeAddLayerNorm";

inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

enum class TILING_TYPE { NORMAL, SLICE, SINGLE_ROW, SINGLE_ROW_EXT, NORMAL_BIG_N, SLICE_EXT, NORMAL_SPECIAL };

bool CheckUbLimit(uint64_t ubSize, int32_t numCol, int64_t axis, int32_t dtSize)
{
    uint32_t ubRequired = 0;
    int32_t blockNum = BLOCK_SIZE / dtSize;
    int32_t numColAligned = CEIL_DIV(numCol, blockNum) * blockNum;
    if (axis == AXIS_VALUE_FOR_PER_TENSOR) {
        // 2 inQue, 1 outQue, 6 = 3 TBuf + 3 gammabetabias_TBuf
        ubRequired =
            (2 + 1) * numColAligned * dtSize + numColAligned * sizeof(int8_t) + 6 * numColAligned * sizeof(float);
    } else {
        // 1 inQue, 1 outQue, 2 Tbuf + 2 gammabeta
        ubRequired = (1 + 1) * numColAligned * dtSize + numColAligned * sizeof(int8_t) +
                     2 * numColAligned * sizeof(float) + 2 * numColAligned * dtSize;
    }
    OP_LOGI("CheckUbLimit", "maxUB: %lu, ubRequired: %u", ubSize, ubRequired);
    OP_CHECK(ubRequired >= ubSize,
        OP_LOGW("QuantizeAddLayerNorm", "Reduce axis is too large, Tiling is not support."),
        return false);
    return true;
}

static inline uint64_t ComputeTilingKey(ge::DataType dataType, bool enableAdditionalOutput, int64_t axis)
{
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = TILING_DATATYPE_FP16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = TILING_DATATYPE_FP32;
    } else if (dataType == ge::DT_BF16) {
        tilingKey = TILING_DATATYPE_BF16;
    }

    if (enableAdditionalOutput) {
        tilingKey += TILING_ADDITIONAL_OUTPUT;
    }

    if (axis == AXIS_VALUE_FOR_PER_TENSOR) {
        tilingKey += TILING_PER_TENSOR;
    } else if (axis == AXIS_VALUE_FOR_MUL_DIV_MODE) {
        tilingKey += TILING_MUL_MODE;
    }
    OP_LOGD("QuantizeAddLayerNorm", "tilingKey: %lu", tilingKey);
    return tilingKey;
}

static ge::graphStatus Tiling4QuantizeAddLayerNorm(gert::TilingContext *context)
{
    OP_LOGD("QuantizeAddLayerNorm", "Enter QuantizeAddLayerNorm tiling");

    QuantizeAddLayerNormTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto dataType = context->GetInputDesc(0)->GetDataType();
    auto dataTypeSize = GetSizeByDataType(dataType);
    OP_LOGD("QuantizeAddLayerNorm", "dataTypeSize: %d", dataTypeSize);
    int64_t qdtype = *context->GetAttrs()->GetInt(0);
    int64_t axis = *context->GetAttrs()->GetInt(1);
    float eps = *context->GetAttrs()->GetFloat(2);
    bool enableAdditionalOutput = *context->GetAttrs()->GetBool(3);

    OP_LOGD("QuantizeAddLayerNorm",
        "eps: %f, x_out: %d, qdtype: %ld, axis: %ld",
        eps,
        enableAdditionalOutput,
        qdtype,
        axis);
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    int32_t numRow = 1;
    for (size_t i = 0; i < context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1; i++) {
        numRow *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    int32_t numCol = context->GetInputShape(0)->GetStorageShape().GetDim(
        context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1);

    uint64_t ubSize = -1;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK(!(CheckUbLimit(ubSize, numCol, axis, dataTypeSize)),
        VECTOR_INNER_ERR_REPORT_TILIING("QuantizeAddLayerNorm", "CheckUbLimit Failed"),
        return ge::GRAPH_FAILED);

    auto firstdimPerCore = CEIL_DIV(numRow, maxCoreNum);
    uint32_t numCore = CEIL_DIV(numRow, firstdimPerCore);

    context->SetBlockDim(numCore);

    tiling.set_numFirstDim(numRow);
    tiling.set_numLastDim(numCol);
    tiling.set_numCore(numCore);

    auto rowFactor = CEIL_DIV(numRow, numCore);
    auto rowTail = numRow - rowFactor * (numCore - 1);

    tiling.set_firstDimPerCore(rowFactor);
    tiling.set_firstDimPerCoreTail(rowTail);
    tiling.set_firstDimPerTime(1);

    float tempAve = (numCol == 0) ? 0 : float(1.0 / numCol);
    tiling.set_eps(eps);
    tiling.set_aveFactor(tempAve);
    OP_LOGD("QuantizeAddLayerNorm",
        "numRow: %d, numCol: %d, numCore: %u, maxCoreNum: %u, firstDimPerCore: %u, firstDimPerCoreTail: %u, "
        "firstDimPerTime: %d, eps: %f, aveFactor: %f",
        numRow,
        numCol,
        numCore,
        maxCoreNum,
        rowFactor,
        rowTail,
        1,
        eps,
        tempAve);

    uint64_t tilingKey = ComputeTilingKey(dataType, enableAdditionalOutput, axis);
    context->SetTilingKey(tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t usrSize = numCore * numCol * sizeof(float) *
                     4;  // useCore * column_size * sizeof(float32) * count(beta, gamma, scales, offsets)
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    OP_LOGD("QuantizeAddLayerNorm", "usrSize: %zu", usrSize);
    OP_LOGD("QuantizeAddLayerNorm", "Successful QuantizeAddLayerNorm tiling");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4QuantizeAddLayerNorm(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantizeAddLayerNormCompileInfo {};

IMPL_OP_OPTILING(QuantizeAddLayerNorm)
    .Tiling(Tiling4QuantizeAddLayerNorm)
    .TilingParse<QuantizeAddLayerNormCompileInfo>(TilingPrepare4QuantizeAddLayerNorm);
}  // namespace optiling