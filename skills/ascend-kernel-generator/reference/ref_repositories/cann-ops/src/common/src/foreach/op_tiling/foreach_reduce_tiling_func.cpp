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
 * \file foreach_reduce_tiling_func.cpp
 * \brief
 */

#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "foreach/op_tiling/foreach_reduce_tiling_func.h"

namespace optiling {

ge::graphStatus ForeachReduceTiling::Init() {
    // Get shape, dtype information, and the total number of data.
    for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
        auto srcTensor = tilingContext->GetDynamicInputTensor(0, i);
        if (srcTensor == nullptr) {
            break;
        }
        auto srcDtype = srcTensor->GetDataType();
        // Determine whether all data types are consistent.
        if (dataType == ge::DT_UNDEFINED) {
            dataType = srcDtype;
            dataTypeSize = GetDataTypeSize(dataType);
            if (dataTypeSize == 0) {
                dataTypeSize = BYTE_LEN_4;
            }
            elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        } else if (srcDtype != dataType) {
            return ge::GRAPH_FAILED;
        }
        gert::Shape tempShape = srcTensor->GetStorageShape();
        // Make a 32-byte alignment for each Tensor
        tensorDataCountList[i] = tempShape.GetShapeSize();
        if (tensorDataCountList[i] == 0) {
            isExistEmptyTensor = true;
        }
        totalBlockCount += CeilA2B(tensorDataCountList[i], elementsPerBlock);
        totalTensorCount++;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachReduceTiling::RunBigKernelTiling() {
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

    uint64_t ubSizePlatForm = 0;

    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    tilingContext->SetTilingKey(GetTilingKeyVal());

    needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    AssignDataToEachCore();
    DivideUbMemory(ubSizePlatForm);
    
    // Reduce Op Addition
    AssignTensorMiddleCountList();

    FillTilingData();

    tilingContext->SetBlockDim(needCoreNum);

    size_t usrSize = (MAX_CORE_CONT + MAX_TENSOR_CONT) * sizeof(float);
    size_t sysWorkspaceSize = WORK_SPACE_SIZE; 
    size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    if (currentWorkspace==nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    
    return ge::GRAPH_SUCCESS;
}

void ForeachReduceTiling::AssignDataToEachCore() {
    // Kernel the input data according to 32 byte alignment.
    // Divisible, representing the amount of data each core needs to process.
    int64_t tempPerCoreCount = totalBlockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = totalBlockCount % needCoreNum;  // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (uint32_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempRealCount = tensorDataCountList[i] - cursorPos;
        int64_t tempCount = CeilA2B(tempRealCount, elementsPerBlock) * elementsPerBlock;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        // ReduceOp need more currect value
        tensorEndOffsetList[coreIndex] = dataCount + tempRealCount < curCmpCount ? tensorDataCountList[i] - 1 : cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPos;
            --i;  // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
    }
}

void ForeachReduceTiling::DivideUbMemory(uint64_t ubSizePlatForm) {
    uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 16384);
    if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
        totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
    }
    uint32_t canUseUbSize = totalSize / 2;
    inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
        canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
        canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
}

void ForeachReduceTiling::AssignTensorMiddleCountList() {
    uint16_t preCoreTensorIndex = 0;
    for (uint32_t i = 1; i < needCoreNum; i++) {
        if (preCoreTensorIndex==tensorStartList[i]) {
            tensorMiddleCountList[preCoreTensorIndex]+=1;
        } else {
            if (tensorStartOffsetList[i]>0) {
                tensorMiddleCountList[tensorStartList[i]]+=1;
            }
        }
        preCoreTensorIndex=tensorStartList[i];
    }
    uint16_t tensorMiddleStart = 0;
    for (uint32_t j = 0; j < totalTensorCount; j++) {
        tensorMiddleCountList[j]++;
        tensorMiddleStartList[j] = tensorMiddleStart;
        tensorMiddleStart += tensorMiddleCountList[j];
    }
    uint16_t coreMiddleOffset = 0;
    for (uint32_t j = 0; j<needCoreNum; j++) {
        coreMiddleOffsetList[j] = coreMiddleOffset;
        coreMiddleOffset += tensorEndList[j] - tensorStartList[j] + 1;
    }
}

static ge::graphStatus Tiling4ForeachNormTiling(gert::TilingContext* context) {
    ForeachReduceTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepare4ForeachTiling(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachNorm)
.Tiling(Tiling4ForeachNormTiling)
.TilingParse<ForeachNormCompileInfo>(TilingPrepare4ForeachTiling);
} // namespace optiling
