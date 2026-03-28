/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file inplace_attn_softmax_tiling_utils.h
 */
#ifndef INPLACE_ATTN_SOFTMAX_TILING_UTILS_H
#define INPLACE_ATTN_SOFTMAX_TILING_UTILS_H
#include "inplace_attn_softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {

constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t OUTPUT_X_INDEX = 0;
constexpr uint32_t ONE = 1;
constexpr uint32_t COMPARE_INT = 255;
constexpr uint32_t TEN = 10;
constexpr uint32_t MINDIM = 2;
constexpr uint32_t MAXDIM = 8;

template <typename T>
auto AlignUp(T num, T div) -> T
{
    return (div == 0) ? 0 : (num + div - 1) / div * div;
}

template <typename T>
inline auto CeilDiv(T num, T div) -> T
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

template <typename T>
inline auto Min(T num, T div) -> T
{
    return num < div ? num : div;
}

template <typename T>
inline auto AlignDown(T a, T base) -> T
{
    return base == 0 ? 0 : a / base * base;
}


ge::graphStatus CheckInputDtype(gert::TilingContext *context)
{
    auto xDtype = context->GetInputDesc(INPUT_X_INDEX)->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT) {
        printf("input x dtype is only support fp16/bf16/fp32.\n");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus CheckOpInputShape(gert::TilingContext *context)
{
    auto xShape = context->GetInputShape(INPUT_X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum < MINDIM || xDimNum > MAXDIM) {
        printf("x dimension should be in [2, 8] !\n");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus CheckOpParams(gert::TilingContext *context)
{
    if (CheckInputDtype(context) != ge::GRAPH_SUCCESS) {
        printf("x dtype is invalid!\n");
        return ge::GRAPH_FAILED;
    }
    if (CheckOpInputShape(context) != ge::GRAPH_SUCCESS) {
        printf("x shape dimension is invalid!\n");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline bool SetTotalShape(gert::Shape &inShape, gert::TilingContext *context, InplaceAttnSoftmaxTilingData &tilingData)
{
    int32_t shapeBefore = 1;
    int32_t shapeAfter = 1;
    int32_t dimNum = inShape.GetDimNum();
    int32_t inDim = -1;

    if (inDim < -dimNum) {
        printf("SetTotalShape Unsupported inDim %d\n", inDim);
        return false;
    }
    //总维度
    int32_t splitDim = inDim < 0 ? dimNum + inDim : inDim;  // inDim default -1
    //将多维数组转换二维数组
    for (int32_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape.GetDim(i);
    }
    for (int32_t j = splitDim; j < dimNum; j++) {
        shapeAfter *= inShape.GetDim(j);
    }
    if (shapeAfter == 0) {
        printf("SetTotalShape Unsupported shapeAfter %d == 0 \n", shapeAfter);
        return false;
    }
    tilingData.set_rowLen(shapeBefore);
    tilingData.set_colLen(shapeAfter);
    return true;
}

inline bool CalculateMaxUbSizePerRow(gert::TilingContext *context, const InplaceAttnSoftmaxCompileInfo &compileInfo,
    InplaceAttnSoftmaxTilingParam &tilingParam, InplaceAttnSoftmaxTilingData &tilingData)
{
    // Align ColLen
    uint32_t colLen = tilingData.get_colLen();
    uint32_t alignedColLen = AlignUp<uint32_t>(colLen, compileInfo.block_num);
    if (alignedColLen == 0) {
        printf("Unsupported alignedColLen %d == 0 \n", alignedColLen);
        return false;
    }
    uint32_t ubAvail = compileInfo.dataNumSingleUb / alignedColLen;
    if (ubAvail == 0) {
        // collen超过ub可用空间大小，需要循环处理colLen
        tilingParam.optBaseColLen = AlignDown<uint32_t>(compileInfo.dataNumSingleUb, compileInfo.block_num);
        // LargeShape
        ubAvail = ONE;
        uint32_t new_tiling_key = tilingData.get_tilingKey();
        new_tiling_key += TEN; 
        tilingData.set_tilingKey(new_tiling_key);
    } else {
        tilingParam.optBaseColLen = colLen;
        ubAvail = std::max(ubAvail, ONE);
    }
    tilingParam.optBaseRowLenHeadCore = std::min(std::min(ubAvail, tilingParam.rowLenPerHeadCore), COMPARE_INT);
    tilingParam.optBaseRowLenTailCore = std::min(std::min(ubAvail, tilingParam.rowLenPerTailCore), COMPARE_INT);
    return true;
}

// 计算每核处理的总行数和实际使用的核数
void CalTilingData(gert::TilingContext *context, InplaceAttnSoftmaxCompileInfo &compileInfo,
    InplaceAttnSoftmaxTilingParam &tilingParam, InplaceAttnSoftmaxTilingData &tilingData)
{
    uint32_t rowLen = tilingData.get_rowLen();

    tilingParam.coreNumUsed = std::max(std::min(compileInfo.totalCore, rowLen), ONE);
    tilingParam.headCoreNum = rowLen % tilingParam.coreNumUsed;

    // rowLenPerHeadCore 指的是 一共有多少个row -> 每个核心处理的row数 上取整
    tilingParam.rowLenPerHeadCore = (rowLen + tilingParam.coreNumUsed - 1) / tilingParam.coreNumUsed;
    tilingParam.rowLenPerTailCore = rowLen / tilingParam.coreNumUsed;

    CalculateMaxUbSizePerRow(context, compileInfo, tilingParam, tilingData);
}

void SetTilingData(
    InplaceAttnSoftmaxCompileInfo &compileInfo, InplaceAttnSoftmaxTilingParam &tilingParam, InplaceAttnSoftmaxTilingData &tilingData)
{
    tilingData.set_headCoreNum(tilingParam.headCoreNum);
    tilingData.set_basicRowLenHeadCore(tilingParam.optBaseRowLenHeadCore);
    tilingData.set_basicRowLenTailCore(tilingParam.optBaseRowLenTailCore);
    tilingData.set_basicColLen(tilingParam.optBaseColLen);
    tilingData.set_rowLenPerHeadCore(tilingParam.rowLenPerHeadCore);
    tilingData.set_rowLenPerTailCore(tilingParam.rowLenPerTailCore);
    tilingData.set_realCoreNum(tilingParam.coreNumUsed);
}
}  // namespace optiling
#endif // INPLACE_ATTN_SOFTMAX_TILING_H