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
 * \file add_rms_norm_quant_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "add_rms_norm_quant_tiling.h"

namespace optiling {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
}

using namespace ge;

namespace optiling {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_FACTOR_B16 = 8192;
constexpr uint32_t UB_FACTOR_B16_CUTD = 7680;
constexpr uint32_t UB_FACTOR_B32_CUTD = 4096;
constexpr uint32_t UB_FACTOR_SINGLE_N_B16 = 12224;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr size_t INPUT_IDX_GAMMA = 2;
constexpr size_t INPUT_IDX_ZERO_POINTS1 = 5;
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t MODE_SINGLE_N = 3;

inline static int64_t CeilDiv(const int64_t dividend, const int64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

static void InitPlatformParams(gert::TilingContext *context, const AddRmsNormQuantCompileInfo *ptrCompileInfo,
    uint32_t &numCore, uint64_t &ubSize, platform_ascendc::SocVersion &socVersion)
{
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
}

ge::graphStatus GetOpDescInfo(gert::TilingContext *context, uint32_t &hasZeroPoints1, uint32_t &numCol,
    uint32_t &numRow, float &epsilon, float &avgFactor)
{
    auto zeroPoints1Desc = context->GetOptionalInputDesc(INPUT_IDX_ZERO_POINTS1);
    hasZeroPoints1 = (zeroPoints1Desc == nullptr) ? 0 : 1;
    const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
    std::string opType(context->GetNodeType());
    const gert::Shape gammaShape = context->GetInputShape(INPUT_IDX_GAMMA)->GetStorageShape();
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    epsilon = *attrs->GetFloat(1);
    numCol = gammaShape.GetShapeSize();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    avgFactor = (numCol == 0) ? 0 : 1.0 / numCol;
    return ge::GRAPH_SUCCESS;
}

static void CalcModeAndUbFactor(gert::TilingContext *context, uint32_t &modeKey, uint32_t &ubFactor,
    uint32_t blockFactor, platform_ascendc::SocVersion &socVersion, uint32_t numCol, DataType &dataType)
{
    if (blockFactor == 1 && socVersion != platform_ascendc::SocVersion::ASCEND310P &&
        numCol <= UB_FACTOR_SINGLE_N_B16) {
        modeKey = MODE_SINGLE_N;
        ubFactor = UB_FACTOR_SINGLE_N_B16;
    } else if (numCol > ubFactor) {
        modeKey = MODE_SPLIT_D;
        ubFactor = (dataType == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD;
        uint32_t colTileNum = CeilDiv(numCol, ubFactor);
        ubFactor = CeilDiv(numCol, colTileNum * BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
    } else {
        modeKey = MODE_NORMAL;
        ubFactor = UB_FACTOR_B16;
    }
}

ge::graphStatus Tiling4AddRmsNormQuant(gert::TilingContext *context)
{
    OP_LOGD("Tiling4AddRmsNormQuant", "Enter Tiling4AddRmsNormQuant");
    AddRMSNormQuantTilingData tiling;
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormQuantCompileInfo *>(context->GetCompileInfo());
    uint32_t numCore;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
    InitPlatformParams(context, ptrCompileInfo, numCore, ubSize, socVersion);
    uint32_t ubFactor = UB_FACTOR_B16;
    uint32_t numCol;
    uint32_t numRow;
    uint32_t hasZeroPoints1;
    float epsilon;
    float avgFactor;
    OP_TILING_CHECK(GetOpDescInfo(context, hasZeroPoints1, numCol, numRow, epsilon, avgFactor) != SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Get Opdesc Info failed."),
        return ge::GRAPH_FAILED);

    uint32_t blockFactor = 1;
    uint32_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    OP_LOGD("Tiling4AddRmsNormQuant", "Core Num: %u, tile num: %d", numCore, tileNum);
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

    context->SetBlockDim(useCoreNum);

    uint32_t rowFactor = 64;
    DataType dataType = context->GetInputDesc(0)->GetDataType();

    uint32_t modeKey = MODE_NORMAL;  // 0: Normal, 1: SplitD, 2: MultiN 3: SingleN
    CalcModeAndUbFactor(context, modeKey, ubFactor, blockFactor, socVersion, numCol, dataType);
    uint32_t tilingKey = modeKey;
    context->SetTilingKey(tilingKey);

    tiling.set_numRow(numRow);
    tiling.set_numCol(numCol);
    tiling.set_blockFactor(blockFactor);
    tiling.set_rowFactor(rowFactor);
    tiling.set_ubFactor(ubFactor);
    tiling.set_epsilon(epsilon);
    tiling.set_avgFactor(avgFactor);
    tiling.set_hasZeroPoints1(hasZeroPoints1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t usrSize = 256;
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    OP_LOGD("Tiling4AddRmsNormQuant", "Tiling Key: %u", tilingKey);
    OP_LOGD("Tiling4AddRmsNormQuant", "Block Dim: %u", useCoreNum);
    OP_LOGD("Tiling4AddRmsNormQuant", "usr Workspace: %zu", usrSize);
    OP_LOGD("Tiling4AddRmsNormQuant",
        "numRow: %d, numCol: %d, blockFactor: %d, rowFactor: %d, ubFactor: %d, epsilon: %f, avgFactor: %f",
        numRow,
        numCol,
        blockFactor,
        rowFactor,
        ubFactor,
        epsilon,
        avgFactor);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4AddRmsNormQuant(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AddRmsNormQuant running.");
    auto compileInfo = GetCompileInfoPtr<AddRmsNormQuantCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormQuant)
    .Tiling(Tiling4AddRmsNormQuant)
    .TilingParse<AddRmsNormQuantCompileInfo>(TilingPrepare4AddRmsNormQuant);

}  // namespace optiling