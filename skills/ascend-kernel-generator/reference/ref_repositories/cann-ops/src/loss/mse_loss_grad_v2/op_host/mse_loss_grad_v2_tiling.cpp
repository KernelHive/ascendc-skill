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
 * \file mse_loss_grad_v2_tiling.cpp
 * \brief
 */

#include "mse_loss_grad_v2_tiling.h"

namespace {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)        \
if ((ptr) == nullptr)                                    \
{                                                        \
    std::printf("nullptr error!");                       \
    return ge::GRAPH_SUCCESS;                            \
}                                                        \

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

}

namespace optiling {
constexpr uint64_t INPUT_BUFFER_NUM = 4;
constexpr uint64_t FP32_SIZE = 4;
constexpr uint64_t USED_DOUBLE_BUFFER_SIZE = 256 * 256;

static constexpr uint32_t FLOAT_KEY = 1;
static constexpr uint32_t FLOAT16_KEY = 2;
static constexpr uint32_t BFLOAT16_KEY = 3;

void GetTilingKey(const uint32_t dtypeKey, uint32_t &tilingKey) {
    tilingKey = dtypeKey;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MseLossGradTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    tiling.set_totalLength(totalLength);
    auto dyDataType = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = 0;
    uint32_t sizeofDataType = 0;
    uint32_t tmpBufferNum = 4;
    if(ge::DT_FLOAT == dyDataType) {
        // 4 means the size of float data type in bytes
        sizeofDataType = 4;
        dtypeKey = FLOAT_KEY;
        tmpBufferNum = 0;
    } else if(ge::DT_FLOAT16 == dyDataType){
        // 2 means the size of float16 data type in bytes
        sizeofDataType = 2;
        dtypeKey = FLOAT16_KEY;
    } else if (ge::DT_BF16 == dyDataType) {
        // 2 means the size of bfloat16 data type in bytes
        sizeofDataType = 2;
        dtypeKey = BFLOAT16_KEY;
    }

    // get attr info
    const char* reduction = context->GetAttrs()->GetStr(0);
    const char* model = "mean";
    float reduceElts = 1.0;
    float cof = 1.0;
    auto shape_reduction = context->GetInputTensor(0)->GetOriginShape();
    int32_t dimNum = shape_reduction.GetDimNum();
    if (strcmp(reduction, model) == 0) {
        for (int32_t i=0; i < dimNum; i++) {
            reduceElts = reduceElts * shape_reduction.GetDim(i);
        }
        // 1.0 means reciprocal, 2.0 means the derivative of the square
        cof = 1.0f / reduceElts * 2.0f;
        tiling.set_cof(cof);
    } else {
        // 2.0 means the derivative of the square
        tiling.set_cof(2.0);
    }
    uint32_t tilingKey = 0;
    GetTilingKey(dtypeKey, tilingKey);
    context->SetTilingKey(tilingKey);

    // set blockDim
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    // ub size
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    // tmp buffertype is fp16 or bf16 need cast fp32
    uint64_t buffersize = INPUT_BUFFER_NUM * sizeofDataType + tmpBufferNum * FP32_SIZE;
    // tile length
    if (buffersize == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t maxTileLength = (ub_size + buffersize - 1) / buffersize;
    // allLoopNumAllign
    if (maxTileLength == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t allLoopNumAllign = (totalLength + maxTileLength - 1) / maxTileLength;
    uint64_t usedCoreNum = coreNum;
    
    if (allLoopNumAllign <= coreNum) {
        usedCoreNum = allLoopNumAllign;
    }
    uint64_t blockLength = 0;
    uint64_t padLength = totalLength % usedCoreNum;
    context->SetBlockDim(usedCoreNum);
    if (padLength == 0){
        blockLength = totalLength / usedCoreNum;
    } else {
        usedCoreNum -= 1;
        padLength = totalLength % usedCoreNum;
        blockLength = totalLength / usedCoreNum;
        if (padLength == 0) {
            context->SetBlockDim(usedCoreNum);
        }
    }
    uint64_t usedDB = 0;
    uint64_t TILE_NUM = (allLoopNumAllign + usedCoreNum - 1) / usedCoreNum;
    if (totalLength > USED_DOUBLE_BUFFER_SIZE) {
        usedDB = 1;
    }

    tiling.set_usedDb(usedDB);
    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_padLength(padLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MseLossGradV2)
    .Tiling(TilingFunc);
}// namespace optiling
