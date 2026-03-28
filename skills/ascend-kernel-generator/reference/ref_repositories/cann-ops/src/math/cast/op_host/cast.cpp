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
 * @file cast_tiling.h
 */
#include "cast_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;
uint64_t ubDataNumMap[40][40] = {};
uint64_t tilingKeyMap[40][40] = {};
uint64_t minDataTypeLengthMap[40][40] = {};
void UbDataNumMapInit()
{
    //                                              = InputBytes * BUFFER_NUM + OutputBytes * BUFFER_NUM + AlltmpBytes
    ubDataNumMap[ge::DT_FLOAT16][ge::DT_FLOAT]      = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT16][ge::DT_INT32]      = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT16][ge::DT_INT8]       = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 4;
    ubDataNumMap[ge::DT_FLOAT16][ge::DT_UINT8]      = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 4;
    ubDataNumMap[ge::DT_FLOAT16][ge::DT_BOOL]       = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT16][ge::DT_INT16]      = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT16][ge::DT_BF16]       = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_FLOAT16]      = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_BF16]         = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_INT32]        = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_INT64]        = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_BOOL]         = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 4;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_INT8]         = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_UINT8]        = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_FLOAT][ge::DT_INT16]        = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT32][ge::DT_FLOAT]        = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT32][ge::DT_FLOAT16]      = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT32][ge::DT_BF16]         = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT32][ge::DT_INT8]         = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_INT32][ge::DT_UINT8]        = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_INT32][ge::DT_INT16]        = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT32][ge::DT_INT64]        = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT32][ge::DT_BOOL]         = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 4;
    ubDataNumMap[ge::DT_INT8][ge::DT_FLOAT16]       = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT8][ge::DT_FLOAT]         = 1 * BUFFER_NUM + 4 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT8][ge::DT_INT32]         = 1 * BUFFER_NUM + 4 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT8][ge::DT_UINT8]         = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT8][ge::DT_BOOL]          = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT8][ge::DT_INT16]         = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT8][ge::DT_INT64]         = 1 * BUFFER_NUM + 8 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_INT8][ge::DT_BF16]          = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_UINT8][ge::DT_FLOAT16]      = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_UINT8][ge::DT_FLOAT]        = 1 * BUFFER_NUM + 4 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_UINT8][ge::DT_INT32]        = 1 * BUFFER_NUM + 4 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_UINT8][ge::DT_INT8]         = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_UINT8][ge::DT_INT16]        = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_UINT8][ge::DT_INT64]        = 1 * BUFFER_NUM + 8 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_UINT8][ge::DT_BF16]         = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_BOOL][ge::DT_FLOAT16]       = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_BOOL][ge::DT_FLOAT]         = 1 * BUFFER_NUM + 4 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_BOOL][ge::DT_INT32]         = 1 * BUFFER_NUM + 4 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_BOOL][ge::DT_UINT8]         = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_BOOL][ge::DT_INT8]          = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_BOOL][ge::DT_INT64]         = 1 * BUFFER_NUM + 8 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_BOOL][ge::DT_BF16]          = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_INT64][ge::DT_FLOAT16]      = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT64][ge::DT_FLOAT]        = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT64][ge::DT_INT32]        = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT64][ge::DT_UINT8]        = 8 * BUFFER_NUM + 1 * BUFFER_NUM + 10;
    ubDataNumMap[ge::DT_INT64][ge::DT_INT8]         = 8 * BUFFER_NUM + 1 * BUFFER_NUM + 10;
    ubDataNumMap[ge::DT_INT64][ge::DT_BOOL]         = 8 * BUFFER_NUM + 1 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_INT64][ge::DT_BF16]         = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT64][ge::DT_INT16]        = 4 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_BF16][ge::DT_FLOAT16]       = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_BF16][ge::DT_FLOAT]         = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_BF16][ge::DT_INT32]         = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_BF16][ge::DT_INT8]          = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 10;
    ubDataNumMap[ge::DT_BF16][ge::DT_UINT8]         = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 10;
    ubDataNumMap[ge::DT_BF16][ge::DT_BOOL]          = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 6;
    ubDataNumMap[ge::DT_INT16][ge::DT_FLOAT16]      = 1 * BUFFER_NUM + 1 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT16][ge::DT_FLOAT]        = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 0;
    ubDataNumMap[ge::DT_INT16][ge::DT_INT32]        = 1 * BUFFER_NUM + 2 * BUFFER_NUM + 2;
    ubDataNumMap[ge::DT_INT16][ge::DT_INT8]         = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 4;
    ubDataNumMap[ge::DT_INT16][ge::DT_UINT8]        = 2 * BUFFER_NUM + 1 * BUFFER_NUM + 4;
    ubDataNumMap[ge::DT_INT16][ge::DT_INT64]        = 1 * BUFFER_NUM + 4 * BUFFER_NUM + 2;
}
void TilingKeyMapInit()
{
    //                                              = Tiling Key
    tilingKeyMap[ge::DT_FLOAT16][ge::DT_FLOAT]      = 1;
    tilingKeyMap[ge::DT_FLOAT16][ge::DT_INT32]      = 1;
    tilingKeyMap[ge::DT_FLOAT16][ge::DT_INT8]       = 3;
    tilingKeyMap[ge::DT_FLOAT16][ge::DT_UINT8]      = 3;
    tilingKeyMap[ge::DT_FLOAT16][ge::DT_BOOL]       = 1;
    tilingKeyMap[ge::DT_FLOAT16][ge::DT_INT16]      = 1;
    tilingKeyMap[ge::DT_FLOAT16][ge::DT_BF16]       = 2;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_FLOAT16]      = 1;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_BF16]         = 1;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_INT32]        = 1;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_INT64]        = 1;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_BOOL]         = 3;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_INT8]         = 4;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_UINT8]        = 4;
    tilingKeyMap[ge::DT_FLOAT][ge::DT_INT16]        = 1;
    tilingKeyMap[ge::DT_INT32][ge::DT_FLOAT]        = 1;
    tilingKeyMap[ge::DT_INT32][ge::DT_FLOAT16]      = 2;
    tilingKeyMap[ge::DT_INT32][ge::DT_BF16]         = 2;
    tilingKeyMap[ge::DT_INT32][ge::DT_INT8]         = 4;
    tilingKeyMap[ge::DT_INT32][ge::DT_UINT8]        = 4;
    tilingKeyMap[ge::DT_INT32][ge::DT_INT16]        = 1;
    tilingKeyMap[ge::DT_INT32][ge::DT_INT64]        = 1;
    tilingKeyMap[ge::DT_INT32][ge::DT_BOOL]         = 3;
    tilingKeyMap[ge::DT_INT8][ge::DT_FLOAT16]       = 1;
    tilingKeyMap[ge::DT_INT8][ge::DT_FLOAT]         = 5;
    tilingKeyMap[ge::DT_INT8][ge::DT_INT32]         = 5;
    tilingKeyMap[ge::DT_INT8][ge::DT_UINT8]         = 8;
    tilingKeyMap[ge::DT_INT8][ge::DT_BOOL]          = 5;
    tilingKeyMap[ge::DT_INT8][ge::DT_INT16]         = 5;
    tilingKeyMap[ge::DT_INT8][ge::DT_INT64]         = 6;
    tilingKeyMap[ge::DT_INT8][ge::DT_BF16]          = 6;
    tilingKeyMap[ge::DT_UINT8][ge::DT_FLOAT16]      = 1;
    tilingKeyMap[ge::DT_UINT8][ge::DT_FLOAT]        = 5;
    tilingKeyMap[ge::DT_UINT8][ge::DT_INT32]        = 5;
    tilingKeyMap[ge::DT_UINT8][ge::DT_INT8]         = 8;
    tilingKeyMap[ge::DT_UINT8][ge::DT_INT16]        = 5;
    tilingKeyMap[ge::DT_UINT8][ge::DT_INT64]        = 6;
    tilingKeyMap[ge::DT_UINT8][ge::DT_BF16]         = 6;
    tilingKeyMap[ge::DT_BOOL][ge::DT_FLOAT16]       = 1;
    tilingKeyMap[ge::DT_BOOL][ge::DT_FLOAT]         = 5;
    tilingKeyMap[ge::DT_BOOL][ge::DT_INT32]         = 5;
    tilingKeyMap[ge::DT_BOOL][ge::DT_UINT8]         = 8;
    tilingKeyMap[ge::DT_BOOL][ge::DT_INT8]          = 8;
    tilingKeyMap[ge::DT_BOOL][ge::DT_INT64]         = 6;
    tilingKeyMap[ge::DT_BOOL][ge::DT_BF16]          = 6;
    tilingKeyMap[ge::DT_INT64][ge::DT_FLOAT16]      = 2;
    tilingKeyMap[ge::DT_INT64][ge::DT_FLOAT]        = 1;
    tilingKeyMap[ge::DT_INT64][ge::DT_INT32]        = 1;
    tilingKeyMap[ge::DT_INT64][ge::DT_UINT8]        = 7;
    tilingKeyMap[ge::DT_INT64][ge::DT_INT8]         = 7;
    tilingKeyMap[ge::DT_INT64][ge::DT_BOOL]         = 6;
    tilingKeyMap[ge::DT_INT64][ge::DT_BF16]         = 2;
    tilingKeyMap[ge::DT_INT64][ge::DT_INT16]        = 2;
    tilingKeyMap[ge::DT_BF16][ge::DT_FLOAT16]       = 2;
    tilingKeyMap[ge::DT_BF16][ge::DT_FLOAT]         = 1;
    tilingKeyMap[ge::DT_BF16][ge::DT_INT32]         = 1;
    tilingKeyMap[ge::DT_BF16][ge::DT_INT8]          = 7;
    tilingKeyMap[ge::DT_BF16][ge::DT_UINT8]         = 7;
    tilingKeyMap[ge::DT_BF16][ge::DT_BOOL]          = 6;
    tilingKeyMap[ge::DT_INT16][ge::DT_FLOAT16]      = 1;
    tilingKeyMap[ge::DT_INT16][ge::DT_FLOAT]        = 1;
    tilingKeyMap[ge::DT_INT16][ge::DT_INT32]        = 2;
    tilingKeyMap[ge::DT_INT16][ge::DT_INT8]         = 3;
    tilingKeyMap[ge::DT_INT16][ge::DT_UINT8]        = 3;
    tilingKeyMap[ge::DT_INT16][ge::DT_INT64]        = 2;
}
void MinDataTypeLengthMapInit() 
{
    minDataTypeLengthMap[ge::DT_FLOAT16][ge::DT_FLOAT]      = 2;
    minDataTypeLengthMap[ge::DT_FLOAT16][ge::DT_INT32]      = 2;
    minDataTypeLengthMap[ge::DT_FLOAT16][ge::DT_INT8]       = 1;
    minDataTypeLengthMap[ge::DT_FLOAT16][ge::DT_UINT8]      = 1;
    minDataTypeLengthMap[ge::DT_FLOAT16][ge::DT_BOOL]       = 1;
    minDataTypeLengthMap[ge::DT_FLOAT16][ge::DT_INT16]      = 2;
    minDataTypeLengthMap[ge::DT_FLOAT16][ge::DT_BF16]       = 2;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_FLOAT16]      = 2;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_BF16]         = 2;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_INT32]        = 4;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_INT64]        = 4;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_BOOL]         = 1;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_INT8]         = 1;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_UINT8]        = 1;
    minDataTypeLengthMap[ge::DT_FLOAT][ge::DT_INT16]        = 2;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_FLOAT]        = 4;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_FLOAT16]      = 2;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_BF16]         = 2;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_INT8]         = 1;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_UINT8]        = 1;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_INT16]        = 2;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_INT64]        = 4;
    minDataTypeLengthMap[ge::DT_INT32][ge::DT_BOOL]         = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_FLOAT16]       = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_FLOAT]         = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_INT32]         = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_UINT8]         = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_BOOL]          = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_INT16]         = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_INT64]         = 1;
    minDataTypeLengthMap[ge::DT_INT8][ge::DT_BF16]          = 1;
    minDataTypeLengthMap[ge::DT_UINT8][ge::DT_FLOAT16]      = 1;
    minDataTypeLengthMap[ge::DT_UINT8][ge::DT_FLOAT]        = 1;
    minDataTypeLengthMap[ge::DT_UINT8][ge::DT_INT32]        = 1;
    minDataTypeLengthMap[ge::DT_UINT8][ge::DT_INT8]         = 1;
    minDataTypeLengthMap[ge::DT_UINT8][ge::DT_INT16]        = 1;
    minDataTypeLengthMap[ge::DT_UINT8][ge::DT_INT64]        = 1;
    minDataTypeLengthMap[ge::DT_UINT8][ge::DT_BF16]         = 1;
    minDataTypeLengthMap[ge::DT_BOOL][ge::DT_FLOAT16]       = 1;
    minDataTypeLengthMap[ge::DT_BOOL][ge::DT_FLOAT]         = 1;
    minDataTypeLengthMap[ge::DT_BOOL][ge::DT_INT32]         = 1;
    minDataTypeLengthMap[ge::DT_BOOL][ge::DT_UINT8]         = 1;
    minDataTypeLengthMap[ge::DT_BOOL][ge::DT_INT8]          = 1;
    minDataTypeLengthMap[ge::DT_BOOL][ge::DT_INT64]         = 1;
    minDataTypeLengthMap[ge::DT_BOOL][ge::DT_BF16]          = 1;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_FLOAT16]      = 2;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_FLOAT]        = 4;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_INT32]        = 4;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_UINT8]        = 1;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_INT8]         = 1;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_BOOL]         = 1;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_BF16]         = 2;
    minDataTypeLengthMap[ge::DT_INT64][ge::DT_INT16]        = 2;
    minDataTypeLengthMap[ge::DT_BF16][ge::DT_FLOAT16]       = 2;
    minDataTypeLengthMap[ge::DT_BF16][ge::DT_FLOAT]         = 2;
    minDataTypeLengthMap[ge::DT_BF16][ge::DT_INT32]         = 2;
    minDataTypeLengthMap[ge::DT_BF16][ge::DT_INT8]          = 1;
    minDataTypeLengthMap[ge::DT_BF16][ge::DT_UINT8]         = 1;
    minDataTypeLengthMap[ge::DT_BF16][ge::DT_BOOL]          = 1;
    minDataTypeLengthMap[ge::DT_INT16][ge::DT_FLOAT16]      = 2;
    minDataTypeLengthMap[ge::DT_INT16][ge::DT_FLOAT]        = 2;
    minDataTypeLengthMap[ge::DT_INT16][ge::DT_INT32]        = 2;
    minDataTypeLengthMap[ge::DT_INT16][ge::DT_INT8]         = 1;
    minDataTypeLengthMap[ge::DT_INT16][ge::DT_UINT8]        = 1;
    minDataTypeLengthMap[ge::DT_INT16][ge::DT_INT64]        = 2;
}
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CastTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310P && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int32_t *dst_type = attrs->GetAttrPointer<int32_t>(0);
    if (*dst_type != context->GetOutputDesc(0)->GetDataType()) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    UbDataNumMapInit();
    TilingKeyMapInit();
    MinDataTypeLengthMapInit();
    auto inputDatatype = context->GetInputDesc(0)->GetDataType();
    auto outputDatatype = context->GetOutputDesc(0)->GetDataType();
    uint64_t tilingKey = tilingKeyMap[inputDatatype][outputDatatype];
    uint64_t typeLength = minDataTypeLengthMap[inputDatatype][outputDatatype];
    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = inputLength / inputNum;
    
    uint64_t ubDataNumber = ubDataNumMap[inputDatatype][outputDatatype];
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint64_t tileDataNum =  (tileBlockNum * BLOCK_SIZE) / inputBytes;
    
    uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    if (inputNum <= tileDataNum)
    {
        coreNum = 1;
    } 
    else 
    {
        coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    }
    if (coreNum == 0 || BLOCK_SIZE == 0) 
    {
        return ge::GRAPH_FAILED;
    } 
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum; 

    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_smallTailDataNum(smallTailDataNum);
    tiling.set_bigTailDataNum(bigTailDataNum);
    tiling.set_finalSmallTileNum(finalSmallTileNum);
    tiling.set_finalBigTileNum(finalBigTileNum);
    tiling.set_tailBlockNum(tailBlockNum);

    context->SetBlockDim(coreNum);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Cast : public OpDef {
public:
    explicit Cast(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({             ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16   , ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT            , ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32            , ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8                    , ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8          , ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL                 , ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64           , ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16              , ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16        })
            .Format({               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND          , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  })
            .UnknownShapeFormat({   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND          , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  });
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({             ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_BOOL, ge::DT_BF16                    , ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16             , ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT64, ge::DT_BOOL             , ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8, ge::DT_BOOL, ge::DT_BF16, ge::DT_INT16, ge::DT_INT64            , ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_INT16, ge::DT_BF16, ge::DT_INT64          , ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8, ge::DT_BF16, ge::DT_INT64          , ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, ge::DT_UINT8, ge::DT_BOOL, ge::DT_INT16, ge::DT_BF16            , ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, ge::DT_UINT8, ge::DT_BOOL        , ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, ge::DT_UINT8, ge::DT_INT64       })
            .Format({               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND          , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  })
            .UnknownShapeFormat({   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND          , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND    , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND   , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  , ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND  });
        this->Attr("dst_type")
            .AttrType(REQUIRED)
            .Int();
        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Cast);
}
