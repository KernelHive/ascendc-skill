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
 * \file instance_norm_v3_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "instance_norm_v3_tiling.h"

namespace optiling {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
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

#define OP_CHECK(cond, log_func, return_expr) \
    if (cond) {                               \
        log_func;                             \
        return_expr;                          \
    }
}  // namespace optiling

namespace optiling {

constexpr int X_IDX = 0;
constexpr int GAMMA_IDX = 1;
constexpr int BETA_IDX = 2;

constexpr int Y_IDX = 0;
constexpr int MEAN_IDX = 1;
constexpr int VAR_IDX = 2;

constexpr int DATA_FORMAT_IDX = 0;
constexpr int EPS_IDX = 1;

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;

constexpr int DIM_NUM = 4;

constexpr int SYS_WORKSPACE_SIZE_310P = 2 * 1024 * 1024;
constexpr int USR_WORKSPACE_SIZE_310P = 256;

constexpr uint32_t TILING_NORMAL = 1;
constexpr uint32_t TILING_CUT_D = 2;

constexpr uint32_t CAXIS_FACTOR_NCHW = 1024;
constexpr uint32_t UB_RESERVED_BYTE = 1024;
constexpr uint32_t SIZEOF_FLOAT = 4;
constexpr uint32_t SIZEOF_HALF = 2;

constexpr uint32_t BLOCK_SIZE = 32;

constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_GAMMA_INDEX = 1;
constexpr int32_t INPUT_BETA_INDEX = 2;
constexpr int32_t OUTPUT_Y_INDEX = 0;
constexpr int32_t OUTPUT_MEAN_INDEX = 1;
constexpr int32_t OUTPUT_VAR_INDEX = 2;

void InstanceNormV3TilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(InstanceNormV3TilingData *tiling)
{
    context->SetBlockDim(useCoreNums);
    tiling->set_N(N);
    tiling->set_C(C);
    tiling->set_H(H);
    tiling->set_W(W);
    tiling->set_useCoreNums(useCoreNums);
    tiling->set_nAxisPerCore(nAxisPerCore);
    tiling->set_nAxisPerCoreTail(nAxisPerCoreTail);
    tiling->set_reduceNums(reduceNums);
    tiling->set_ubFactor(ubFactor);
    tiling->set_cAxisFactor(cAxisFactor);
    tiling->set_epsilon(eps);
    tiling->set_avgFactor(avgFactor);

    // set tiling key
    uint32_t tilingKey = 0;
    tilingKey += ubTilingStrategy;
    context->SetTilingKey(tilingKey);

    tiling->SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    // set workspace
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE_310P;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t usrSize = USR_WORKSPACE_SIZE_310P;
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    OP_LOGD("SetTilingDataAndTilingKeyAndWorkSpace", "set tiling info");
    OP_LOGD("SetTilingDataAndTilingKeyAndWorkSpace", "[%u, %u, %u, %u]", N, C, H, W);
    OP_LOGD("SetTilingDataAndTilingKeyAndWorkSpace", "useCoreNums: %u", useCoreNums);
    OP_LOGD("SetTilingDataAndTilingKeyAndWorkSpace",
        "nAxisPerCore: %u, nAxisPerCoreTail: %u",
        nAxisPerCore,
        nAxisPerCoreTail);
    OP_LOGD("SetTilingDataAndTilingKeyAndWorkSpace",
        "reduceNums: %u, ubFactor: %u, cAxisFactor: %u",
        reduceNums,
        ubFactor,
        cAxisFactor);
    OP_LOGD("SetTilingDataAndTilingKeyAndWorkSpace", "eps: %f, avgFactor: %f", eps, avgFactor);
    OP_LOGD("SetTilingDataAndTilingKeyAndWorkSpace", "tilingKey = %u, usr Workspace: %zu", tilingKey, usrSize);
}

bool InstanceNormV3TilingHelper::DoTiling()
{
    OP_CHECK((nullptr == compileInfo || nullptr == context),
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "compileInfo or context get nullptr, return failed."),
        return false);

    bool status = GetBaseInfo();
    OP_CHECK(!status,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "GetBaseInfo falied, return false"),
        return false);

    status = GetShapeInfo();
    OP_CHECK(!status,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "GetShapeInfo falied, return false"),
        return false);

    status = DoBlockTiling();
    OP_CHECK(!status,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "DoBlockTiling falied, return false"),
        return false);

    status = DoUbTiling();
    OP_CHECK(!status,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "DoUbTiling falied, return false"),
        return false);

    return status;
}

bool InstanceNormV3TilingHelper::DoBlockTiling()
{
    // Block Tiling, Cut N
    nAxisPerCore = CeilDiv(N, coreNums);
    useCoreNums = CeilDiv(N, nAxisPerCore);
    nAxisPerCore = CeilDiv(N, useCoreNums);
    nAxisPerCoreTail = N - nAxisPerCore * (useCoreNums - 1);
    OP_LOGD("DoBlockTiling",
        "BlockTiling Factor: useCoreNums: %u, nAxisPerCore: %u, nAxisPerCoreTail: %u",
        useCoreNums,
        nAxisPerCore,
        nAxisPerCoreTail);
    return true;
}

bool InstanceNormV3TilingHelper::GetBaseInfo()
{
    coreNums = compileInfo->totalCoreNum;
    ubSize = compileInfo->ubSize;
    OP_LOGD("GetBaseInfo", "Soc Core Num: %u, ubSize: %u", coreNums, ubSize);

    auto attrs = context->GetAttrs();
    OP_CHECK(nullptr == attrs,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "Get attrs nullptr, return false."),
        return false);

    const float *epsilon = attrs->GetFloat(EPS_IDX);
    OP_CHECK(nullptr == epsilon,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "Get epsilon nullptr, return false."),
        return false);
    eps = *epsilon;
    OP_TILING_CHECK(eps <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "Epsilon less or equal than zero, please check."),
        return false);

    dataFormat = attrs->GetAttrPointer<char>(DATA_FORMAT_IDX);
    OP_CHECK(nullptr == dataFormat,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "Get required attr dataFormat failed, tiling falied."),
        return false);

    OP_LOGD("GetBaseInfo", "dataFormat: %s, epsilon: %f", dataFormat, eps);
    return true;
}

bool InstanceNormV3TilingHelper::GetShapeInfo()
{
    xDtype = context->GetInputDesc(X_IDX)->GetDataType();
    dtSize = (xDtype == ge::DataType::DT_FLOAT) ? SIZEOF_FLOAT : SIZEOF_HALF;

    const gert::Shape xShape = context->GetInputShape(X_IDX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();

    OP_CHECK(DIM_NUM != xDimNum,
        VECTOR_INNER_ERR_REPORT_TILIING(
            "Tiling4InstanceNormV3", "xDimNum should equals to 4, but get xDimNum=%zu.", xDimNum),
        return false);

    std::string dataFormatStr = dataFormat;
    OP_CHECK(("NCHW" != dataFormatStr),
        VECTOR_INNER_ERR_REPORT_TILIING(
            "Tiling4InstanceNormV3", "Get dataFormat: %s != NCHW, tiling failed", dataFormat),
        return false);
    N = xShape[IDX_0];
    C = xShape[IDX_1];
    H = xShape[IDX_2];
    W = xShape[IDX_3];
    reduceNums = H * W;
    avgFactor = 1.0 / ((float)reduceNums);
    OP_LOGD("GetShapeInfo", "[N, C, H, W] = [%u, %u, %u, %u]", N, C, H, W);
    OP_LOGD("GetShapeInfo", "reduceNums=%u, avgFactor=%f", reduceNums, avgFactor);
    OP_LOGD("GetShapeInfo", "dtSize=%u", dtSize);

    OP_CHECK(C * dtSize < BLOCK_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "C-axis should bigger than 32B on 310P"),
        return false);
    OP_CHECK(H * W * dtSize < BLOCK_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "HW-axis should bigger than 32B on 310P"),
        return false);
    return true;
}

bool InstanceNormV3TilingHelper::DoUbTiling()
{
    cAxisFactor = CAXIS_FACTOR_NCHW;
    // 4 Tbuf for CAXIS_FACTOR_NCHW
    uint32_t ubAvailable = ubSize - UB_RESERVED_BYTE - 4 * cAxisFactor * SIZEOF_FLOAT;

    // 2 TQue and 2 TBuf
    ubFactor = FloorAlign((ubAvailable / (2 * dtSize + 2 * SIZEOF_FLOAT)), BLOCK_SIZE);
    ubTilingStrategy = TILING_NORMAL;
    if (ubFactor < reduceNums) {
        // 2 TQue and 3 TBuf for CutD
        ubFactor = FloorAlign((ubAvailable / (2 * dtSize + 3 * SIZEOF_FLOAT)), BLOCK_SIZE);
        ubTilingStrategy = TILING_CUT_D;
    }
    OP_LOGD(
        "DoUbTiling", "ubFactor: %u, cAxisFactor: %u, ubTilingStrategy: %u", ubFactor, cAxisFactor, ubTilingStrategy);
    return true;
}

static bool CheckInputOutputShape(const gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *betaShape = context->GetInputShape(INPUT_BETA_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    const gert::StorageShape *meanShape = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    const gert::StorageShape *varianceShape = context->GetOutputShape(OUTPUT_VAR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varianceShape);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t betaDimNum = betaShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    size_t meanDimNum = meanShape->GetStorageShape().GetDimNum();
    size_t varianceDimNum = varianceShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK((meanDimNum != DIM_NUM) || (varianceDimNum != DIM_NUM),
        OP_LOGE(context->GetNodeName(), "Output mean/variance shape invaild, dim num is not equal 4."),
        return false);
    OP_TILING_CHECK((yDimNum != xDimNum),
        OP_LOGE(context->GetNodeName(), "Output y shape invaild, dim num is not equal x dim num."),
        return false);
    OP_TILING_CHECK((gammaDimNum != 1) || (betaDimNum != 1),
        OP_LOGE(context->GetNodeName(), "Input gamma/beta shape invaild, dim num is not equal 1."),
        return false);

    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_TILING_CHECK(xShape->GetStorageShape().GetDim(i) == 0,
            OP_LOGE(context->GetNodeName(), "Input x shape can not be 0."),
            return false);
        OP_TILING_CHECK(yShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i),
            OP_LOGE(context->GetNodeName(), "Output y shape invaild, shape is not equal x shape."),
            return false);
        OP_TILING_CHECK(
            (meanShape->GetStorageShape().GetDim(i) == 0) || (varianceShape->GetStorageShape().GetDim(i) == 0),
            OP_LOGE(context->GetNodeName(), "Output mean/variance shape can not be 0."),
            return false);
    }
    return true;
}
static ge::graphStatus Tiling4InstanceNormV3(gert::TilingContext *context)
{
    OP_LOGD("Tiling4InstanceNormV3", "Enter Tiling4InstanceNormV3");

    InstanceNormV3TilingData tiling;
    OP_TILING_CHECK(!CheckInputOutputShape(context),
        OP_LOGE(context->GetNodeName(), "Input shape invalid."),
        return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const InstanceNormV3CompileInfo *>(context->GetCompileInfo());
    OP_CHECK(nullptr == compileInfo,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "Get nullptr compileInfo, return Failed."),
        return ge::GRAPH_FAILED);

    InstanceNormV3TilingHelper instanceNormV3TilingHelper(context, compileInfo);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OP_CHECK(!status,
        VECTOR_INNER_ERR_REPORT_TILIING("Tiling4InstanceNormV3", "DoTiling Failed, return Failed."),
        return ge::GRAPH_FAILED);

    instanceNormV3TilingHelper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4InstanceNormV3(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4InstanceNormV3 running");
    auto compileInfo = GetCompileInfoPtr<InstanceNormV3CompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD("TilingPrepare4InstanceNormV3", "compileInfo->totalCoreNum: %d", compileInfo->totalCoreNum);
    // no vector core enabled
    OP_TILING_CHECK((compileInfo->totalCoreNum <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepare4InstanceNormV3 fail to get core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_LOGD("TilingPrepare4InstanceNormV3", "compileInfo->ubSize: %lu", compileInfo->ubSize);

    OP_TILING_CHECK((compileInfo->ubSize <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepare4InstanceNormV3 fail to get ub size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepare4InstanceNormV3 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InstanceNormV3)
    .Tiling(Tiling4InstanceNormV3)
    .TilingParse<InstanceNormV3CompileInfo>(TilingPrepare4InstanceNormV3);

}  // namespace optiling