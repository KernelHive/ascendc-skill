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
 * \file diag_flat_tiling.cc
 * \brief
 */
#include <cmath>
#include "diag_v2_tiling.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...)  \
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


template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

namespace optiling {

static const int32_t B16_INPUT_LENGTH = 16;

static const int32_t L32_INPUT_BYTES = 4;
static const int32_t L64_INPUT_BYTES = 8;
static const int32_t L16_INPUT_BYTES = 2;

static const int32_t L32_SCALAR_TILING_KEY = 1011;
static const int32_t L64_SCALAR_TILING_KEY = 1012;
static const int32_t L16_SCALAR_TILING_KEY = 1013;
static const int32_t L8_SCALAR_TILING_KEY = 1014;

static const int32_t L32_MATRIX_TILING_KEY = 1031;
static const int32_t L64_MATRIX_TILING_KEY = 1032;
static const int32_t L16_MATRIX_TILING_KEY = 1033;
static const int32_t L8_MATRIX_TILING_KEY = 1034;

static ge::graphStatus DiagFlatSetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcScalarTiling(const gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    tilingData.set_totalCoreNum(tilingData.get_totalCoreNum());
    tilingData.set_usedCoreNum(1);
    tilingData.set_normalCoreHandleNum(tilingData.get_inputNum());
    tilingData.set_lastCoreHandleNum(0);
    
    auto input = context->GetInputTensor(0);
    auto dataType = input->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dataType);
    if (dtypeSize != B16_INPUT_LENGTH) { // 16
        if (dtypeSize == L32_INPUT_BYTES) {
            tilingData.set_tilingKey(L32_SCALAR_TILING_KEY); // int32/float32..
        } else if (dtypeSize == L64_INPUT_BYTES) {
            tilingData.set_tilingKey(L64_SCALAR_TILING_KEY); // int64/float64..
        } else if (dtypeSize == L16_INPUT_BYTES) {
            tilingData.set_tilingKey(L16_SCALAR_TILING_KEY); // int16/float16..
        } else {
            tilingData.set_tilingKey(L8_SCALAR_TILING_KEY); // int8/float8..
        }
    } else {
        tilingData.set_tilingKey(SCALAR_EQUAL_SIZE16); // 102
    }

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcAuxMatrixTiling(const gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    int64_t tmpNum = std::max(CeilDiv(tilingData.get_inputNum(), tilingData.get_totalCoreNum()), LEAST_NUM_PER_CORE);
    tmpNum = CeilAlign(tmpNum, LEAST_NUM_PER_CORE);
    tilingData.set_usedCoreNum(CeilDiv(tilingData.get_inputNum(), tmpNum));
    tilingData.set_normalCoreHandleNum(tmpNum);
    tilingData.set_lastCoreHandleNum(tilingData.get_inputNum()
                                     - (tilingData.get_usedCoreNum() - 1) * tilingData.get_normalCoreHandleNum());
    
    auto input = context->GetInputTensor(0);
    auto dataType = input->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dataType);
    if (dtypeSize != B16_INPUT_LENGTH) {
        if (dtypeSize == L32_INPUT_BYTES) {
            tilingData.set_tilingKey(L32_MATRIX_TILING_KEY); // int32/float32..
        } else if (dtypeSize == L64_INPUT_BYTES) {
            tilingData.set_tilingKey(L64_MATRIX_TILING_KEY); // int64/float64..
        } else if (dtypeSize == L16_INPUT_BYTES) {
            tilingData.set_tilingKey(L16_MATRIX_TILING_KEY); // int16/float16..
        } else {
            tilingData.set_tilingKey(L8_MATRIX_TILING_KEY); // int8/float8..
        }
    } else {
        tilingData.set_tilingKey(ASSIST_EQUAL_SIZE16); // 104
    }
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(DiagV2TilingData &tilingData)
{
    OP_LOGI("---[DiagFlat]---", "normalCoreHandleNum: %ld",  tilingData.get_normalCoreHandleNum());
    OP_LOGI("---[DiagFlat]---", "lastCoreHandleNum: %ld", tilingData.get_lastCoreHandleNum());
    OP_LOGI("---[DiagFlat]---", "totalCoreNum: %ld", tilingData.get_totalCoreNum());
    OP_LOGI("---[DiagFlat]---", "usedCoreNum: %ld",  tilingData.get_usedCoreNum());
    OP_LOGI("---[DiagFlat]---", "inputNum: %ld",  tilingData.get_inputNum());
    OP_LOGI("---[DiagFlat]---", "diagonal: %ld",  tilingData.get_diagonal());
    OP_LOGI("---[DiagFlat]---", "ubInputSize: %ld",  tilingData.get_ubInputSize());
    OP_LOGI("---[DiagFlat]---", "ubOutputSize: %ld",  tilingData.get_ubOutputSize());
    OP_LOGI("---[DiagFlat]---", "workspaceSize: %ld", tilingData.get_workspaceSize());
    OP_LOGI("---[DiagFlat]---", "tilingKey: %ld", tilingData.get_tilingKey());
}

ge::graphStatus TilingDiagFlat(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingDiagFlat running begin");
    auto input = context->GetInputTensor(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input);
    auto inputShape = input->GetStorageShape();
    DiagV2TilingData tilingData;
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, diagonalPtr);
    const int64_t diagonal = *diagonalPtr;
    tilingData.set_diagonal(diagonal);
    // set coreNum
    auto compileInfo = reinterpret_cast<const DiagV2CompileInfo*>(context->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    int64_t totalCoreNum = static_cast<int64_t>(compileInfo->totalCoreNum);
    tilingData.set_totalCoreNum(totalCoreNum);
    // set ubSize
    int64_t ubSize = compileInfo->ubSizePlatForm;
    auto dataType = input->GetDataType();
    auto auxMatrix = SCALAR_THRESHOLD_NUM * SCALAR_THRESHOLD_NUM * sizeof(dataType);
    auto outputDataSize = SCALAR_THRESHOLD_NUM * SCALAR_THRESHOLD_NUM * sizeof(dataType);
    auto inputDataSize = ubSize - auxMatrix - outputDataSize;
    tilingData.set_ubInputSize(inputDataSize);
    tilingData.set_ubOutputSize(outputDataSize);
    // set inputNum
    tilingData.set_inputNum(inputShape.GetShapeSize());
    if (tilingData.get_inputNum() <= SCALAR_THRESHOLD_NUM) {
        OP_TILING_CHECK(CalcScalarTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "CalcScalarTiling fail."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(CalcAuxMatrixTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "CalcAuxMatrixTiling fail."),
                        return ge::GRAPH_FAILED);
    }
    size_t userSize = tilingData.get_totalCoreNum() * 32;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userSize + sysWorkspaceSize;
    tilingData.set_workspaceSize(currentWorkspace[0]);
    // set tilingData
    OP_TILING_CHECK(DiagFlatSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "DiagFlatSetTilingData set tiling data fail."),
                    return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_totalCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForDiagFlat(gert::TilingParseContext* context)
{
    auto compileInfo = GetCompileInfoPtr<DiagV2CompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->totalCoreNum <= 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                      "TilingPrepare4DiagFlat fail to get core num."),
                      return ge::GRAPH_FAILED);
    
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_TILING_CHECK((compileInfo->ubSizePlatForm <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "TilingPrepare4DiagFlat fail to get ub size."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 向框架注册入口函数
IMPL_OP_OPTILING(DiagFlat)
    .Tiling(TilingDiagFlat)
    .TilingParse<DiagV2CompileInfo>(TilingPrepareForDiagFlat);

}  // namespace optiling
 