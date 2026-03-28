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
 * \file multi_scale_deformable_attn_function.cc
 * \brief
 */
#include "multi_scale_deformable_attn_function_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "runtime2_util.h"
#include "op_log.h"

using namespace ge;
namespace {
const std::string  OP_NAME = "MultiScaleDeformableAttn";
const uint64_t INPUT_VALUE = 0;
const uint64_t INPUT_SPATIAL_SHAPE = 1;
const uint64_t INPUT_ATTN_WEIGHT = 4;

const uint64_t BATCH_SIZE_DIM = 0;
const uint64_t NUM_KEYS_DIM = 1;
const uint64_t NUM_HEADS_DIM = 2;
const uint64_t EMBED_DIMS_DIM = 3;
const uint64_t NUM_LEVEL_DIM = 0;
const uint64_t REAL_LEVEL_DIM = 3;
const uint64_t NUM_QUERIES_DIM = 1;
const uint64_t NUM_POINTS_DIM = 4;

const uint64_t NUM_KEYS_DIM_TRANSPOSE = 2;
const uint64_t NUM_HEADS_DIM_TRANSPOSE = 1;
const uint64_t REAL_LEVEL_DIM_TRANSPOSE = 2;
const uint64_t NUM_QUERIES_DIM_TRANSPOSE = 4;
const uint64_t NUM_POINTS_DIM_TRANSPOSE = 3;

const uint64_t B32_DATA_NUM_PER_BLOCK = 4;
const uint64_t NUM_POINTS_EIGHT = 8;
const uint64_t NUM_POINTS_FOUR = 4;
const uint64_t NUM_POINTS_TWO = 2;
const uint64_t EMBEDDIMS_SIXTEEN = 16;
const uint64_t sysWorkspaceSize = 16 * 1024 *1024;
const uint64_t TILING_KEY_WEIGHT = 1000;

const uint64_t NUM_QUERIES_MIN = 32;
const uint64_t NUM_HEADS_MAX = 16;
const uint64_t NUM_LEVELS_MAX = 16;
const uint64_t NUM_POINTS_MAX = 16;
const uint64_t EMBED_DIMS_DIV_EIGHT = 8;
const uint64_t EMBED_DIMS_MAX = 256;

uint64_t deterministicFlag = 0;

// the points can be grouped into 2, 4 or 8 points per block
// the numPoints has to be even, except 1
std::tuple<uint64_t, uint64_t> GroupPoints(uint64_t numPoints)
{
    if (numPoints % NUM_POINTS_EIGHT == 0) {
        return std::make_tuple(NUM_POINTS_EIGHT, numPoints / NUM_POINTS_EIGHT);
    }
    if (numPoints % NUM_POINTS_FOUR == 0) {
        return std::make_tuple(NUM_POINTS_FOUR, numPoints / NUM_POINTS_FOUR);
    }
    if (numPoints % NUM_POINTS_TWO == 0) {
        return std::make_tuple(NUM_POINTS_TWO, numPoints / NUM_POINTS_TWO);
    }
    return std::make_tuple(1, numPoints);
}
} // namespace

namespace optiling {

static void MultiScaleDeformableAttnPrintParam(MultiScaleDeformableAttnFunctionTilingData &tiling, const gert::TilingContext* context)
{
    OP_LOGD(OP_NAME, "MultiScaleDeformableAttn Tiling Data Print:");
    OP_LOGD(context->GetNodeName(), "batchSize = %lu", tiling.get_batchSize());
    OP_LOGD(context->GetNodeName(), "numKeys = %lu", tiling.get_numKeys());
    OP_LOGD(context->GetNodeName(), "numHeads = %lu", tiling.get_numHeads());
    OP_LOGD(context->GetNodeName(), "embedDims = %lu", tiling.get_embedDims());
    OP_LOGD(context->GetNodeName(), "numLevels = %lu", tiling.get_numLevels());
    OP_LOGD(context->GetNodeName(), "numQueries = %lu", tiling.get_numQueries());
    OP_LOGD(context->GetNodeName(), "numPoints = %lu", tiling.get_numPoints());
    OP_LOGD(context->GetNodeName(), "coreNum = %lu", tiling.get_coreNum());
    OP_LOGD(context->GetNodeName(), "pointLoops = %lu", tiling.get_pointLoops());
    OP_LOGD(context->GetNodeName(), "realLevels = %lu", tiling.get_realLevels());
}

static ge::graphStatus TilingFuncForMultiScaleDeformableAttn(gert::TilingContext *context)
{
    OP_LOGD(OP_NAME, "TilingFuncForMultiScaleDeformableAttn Tiling start.");
    MultiScaleDeformableAttnFunctionTilingData tiling;
    auto valueTensorPtr = context->GetInputTensor(INPUT_VALUE);
    auto spatialTensorPtr = context->GetInputTensor(INPUT_SPATIAL_SHAPE);
    auto attnWeightTensorPtr = context->GetInputTensor(INPUT_ATTN_WEIGHT);
    if (valueTensorPtr == nullptr || spatialTensorPtr == nullptr || attnWeightTensorPtr == nullptr) {
        OP_LOGD(OP_NAME, "valueTensorPtr or spatialTensorPtr or attnWeightTensorPtr is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto valueShape = valueTensorPtr->GetStorageShape();
    auto spatialShape = spatialTensorPtr->GetStorageShape();
    auto attnWeightShape = attnWeightTensorPtr->GetStorageShape();

    auto compileInfo = reinterpret_cast<const MultiScaleDeformableAttnFunctionCompileInfo*>(context->GetCompileInfo());
    uint64_t coreNum = compileInfo->totalCoreNum;

    deterministicFlag = context->GetDeterministic() == 1 ? 1 : 0;
    OP_LOGD(context->GetNodeName(), "deterministicFlag is %lu.", deterministicFlag);
    if (deterministicFlag == 1) {
        coreNum = 1;
    }
    OP_LOGD(context->GetNodeName(), "core_num is %lu.", coreNum);

    context->SetBlockDim(coreNum);
    tiling.set_coreNum(coreNum);

    uint64_t isTranspose = attnWeightShape.GetDim(1) >= 32 ? 0 : 1;
    uint64_t numLevels = spatialShape.GetDim(NUM_LEVEL_DIM);
    uint64_t embedDims = valueShape.GetDim(EMBED_DIMS_DIM);
    uint64_t batchSize = valueShape.GetDim(BATCH_SIZE_DIM);
    uint64_t numPoints, numHeads, numKeys, numQueries, realLevels;
    if (isTranspose == 0) {
        numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM);
        numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM);
        numKeys = valueShape.GetDim(NUM_KEYS_DIM);
        numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM);
        realLevels = attnWeightShape.GetDim(REAL_LEVEL_DIM);
    } else {
        numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM_TRANSPOSE);
        numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM_TRANSPOSE);
        numKeys = valueShape.GetDim(NUM_KEYS_DIM_TRANSPOSE);
        numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM_TRANSPOSE);
        realLevels = attnWeightShape.GetDim(REAL_LEVEL_DIM_TRANSPOSE);
    }
    uint64_t optPoint = numLevels <= 8 && numHeads <= 8 && (embedDims == 16 || embedDims == 32) &&
                        (numPoints % 2 == 0 || numPoints == 1);
    uint64_t pointLoops = 0;
    uint64_t point = 0;
    if (optPoint) {
        auto groups = GroupPoints(numPoints);
        pointLoops = std::get<1>(groups);
        point = std::get<0>(groups);
    }

    if (numQueries < NUM_QUERIES_MIN) {
        OP_LOGE(context->GetNodeName(), "numQueries must be greater than or equal to 32, bug got input %lu", numQueries);
        return ge::GRAPH_FAILED;
    }

    if (numHeads > NUM_HEADS_MAX) {
        OP_LOGE(context->GetNodeName(), "numHeads must be less than or equal to 16, bug got input %lu", numHeads);
        return ge::GRAPH_FAILED;
    }

    if (numLevels > NUM_LEVELS_MAX) {
        OP_LOGE(context->GetNodeName(), "numLevels must be less than or equal to 16, bug got input %lu", numLevels);
        return ge::GRAPH_FAILED;
    }

    if (numPoints > NUM_POINTS_MAX) {
        OP_LOGE(context->GetNodeName(), "numPoints must be less than or equal to 16, bug got input %lu", numPoints);
        return ge::GRAPH_FAILED;
    }

    if (!((embedDims % EMBED_DIMS_DIV_EIGHT == 0) && (embedDims <= EMBED_DIMS_MAX))) {
        OP_LOGE(context->GetNodeName(), "embedDims must be divisible by 8 and less than or equal to 256, bug got input %lu", embedDims);
        return ge::GRAPH_FAILED;
    }

    uint64_t TilingKey = optPoint == 1 ? (embedDims / EMBEDDIMS_SIXTEEN) * TILING_KEY_WEIGHT + point : 0;
    OP_LOGD(context->GetNodeName(), "TilingKey = %lu", TilingKey);
    
    context->SetTilingKey(TilingKey);
    tiling.set_batchSize(batchSize);
    tiling.set_numKeys(numKeys);
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(numQueries);
    tiling.set_numPoints(numPoints);
    tiling.set_coreNum(coreNum);
    tiling.set_pointLoops(pointLoops);
    tiling.set_realLevels(realLevels);
    MultiScaleDeformableAttnPrintParam(tiling, context);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD(OP_NAME, "TilingFuncForMultiScaleDeformableAttn Tiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4MultiScaleDeformableAttnFunction(gert::TilingParseContext* context) {
  OP_LOGD("MultiScaleDeformableAttnFunction:", "TilingPrepare4MultiScaleDeformableAttnFunction start.");
  auto compileInfo = GetCompileInfoPtr<MultiScaleDeformableAttnFunctionCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
  OP_TILING_CHECK((compileInfo->totalCoreNum <= 0), // 0 negative number
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get core num."),
                    return ge::GRAPH_FAILED);

  uint64_t ubSizePlatform = 0U; // 0, init
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
  compileInfo->ubSizePlatform = static_cast<int64_t>(ubSizePlatform);
  OP_TILING_CHECK((compileInfo->ubSizePlatform <= 0), // 0
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size"),
                    return ge::GRAPH_FAILED);
  OP_LOGD("MultiScaleDeformableAttnFunction:", "TilingPrepare4MultiScaleDeformableAttnFunction end.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MultiScaleDeformableAttnFunction)
    .Tiling(TilingFuncForMultiScaleDeformableAttn)
    .TilingParse<MultiScaleDeformableAttnFunctionCompileInfo>(TilingPrepare4MultiScaleDeformableAttnFunction);
}  // namespace optiling