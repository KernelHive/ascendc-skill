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
 * \file foreach_tiling_func.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "foreach/op_tiling/foreach_tiling_func.h"

namespace optiling {

ge::graphStatus ForeachCommonTiling::Init(uint8_t theCode) {
    opCode = theCode;
    int dynamicIdx = opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE ? 1 : 0;
    for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
        auto srcTensor = tilingContext->GetDynamicInputTensor(dynamicIdx, i);
        if (srcTensor == nullptr) {
            break;
        }

        auto temp = tilingContext->GetInputDesc(0);
        if (temp == nullptr) {
            return ge::GRAPH_FAILED;
        }

        auto srcDtype = temp->GetDataType();

        if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            if (tilingContext->GetInputDesc(1) != nullptr) {
                srcDtype = tilingContext->GetInputDesc(1)->GetDataType();
            } else {
                return ge::GRAPH_FAILED;
            }
        }
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
        totalDataCount += tensorDataCountList[i];
        totalTensorCount++;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachCommonTiling::RunBigKernelTiling() {
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

    uint64_t ubSizePlatForm = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    tilingContext->SetTilingKey(GetTilingKeyByDtypeOnly(dataType));

    uint32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    AssignDataToEachCore(needCoreNum);
    DivideUbMemory(ubSizePlatForm);
    FillTilingData();
    tilingContext->SetBlockDim(needCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = WORK_SPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachCommonTiling::RunBigScalarKernelTiling() {
    auto platformInfo = tilingContext->GetPlatformInfo();
    auto compileInfoPtr = (const ForeachCompileInfo *)(tilingContext->GetCompileInfo());

    uint64_t ubSizePlatForm = 0;
    uint32_t needCoreNum = 0;
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        needCoreNum = GetNeedCoreNum(ascendcPlatform.GetCoreNumAiv());
    } else {
        ubSizePlatForm = compileInfoPtr->ubSize;
        needCoreNum = GetNeedCoreNum(compileInfoPtr->aivCoreNum);
    }
    tilingContext->SetTilingKey(GetTilingKeyByDtypeOnly(dataType));
    AssignDataToEachCore(needCoreNum);
    DivideUbMemory(ubSizePlatForm);
    FillTilingData();
    tilingContext->SetBlockDim(needCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = WORK_SPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}


void ForeachCommonTiling::AssignDataToEachCore(int64_t needCoreNum) {
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
    // Divisible, representing the amount of data each core needs to process.
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum;  // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPosition = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountList[i] - cursorPosition;

        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPosition = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPosition = cursorPosition + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPosition - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPosition < tensorDataCountList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPosition;
            --i;  // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPosition = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
    }
}


void ForeachCommonTiling::DivideUbMemory(uint64_t ubSizePlatForm) {
    if (opCode <= FOREACH_POINTWISE_OP_CODE) {
        DivideUbMemory1(ubSizePlatForm);
    } else if (opCode <= FOREACH_POW_TENSOR_OP_CODE) {
        DivideUbMemory2(ubSizePlatForm);
    } else if (opCode <= FOREACH_ERF_OP_CODE) {
        DivideUbMemory3(ubSizePlatForm);
    } else if (opCode <= FOREACH_TAN_OP_CODE) {
        DivideUbMemory4(ubSizePlatForm);
    } else if (opCode <= FOREACH_ATAN_OP_CODE) {
        DivideUbMemory5(ubSizePlatForm);
    } else if (opCode <= FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
        DivideUbMemory6(ubSizePlatForm);
    } else if (opCode <= FOREACH_MUL_SCALAR_OP_CODE) {
        DivideUbMemory7(ubSizePlatForm);
    } else if (opCode <= FOREACH_ADD_LIST_OP_CODE) {
        DivideUbMemory8(ubSizePlatForm);
    } else if (opCode <= FOREACH_DIV_SCALAR_OP_CODE) {
        DivideUbMemory9(ubSizePlatForm);
    } else if (opCode <= FOREACH_COPY_OP_CODE) {
        DivideUbMemory9(ubSizePlatForm);
    } else if (opCode <= FOREACH_SIGN_OP_CODE) {
        DivideUbMemory10(ubSizePlatForm);
    }
}

void ForeachCommonTiling::DivideUbMemory1(uint64_t ubSizePlatForm) {
    if (opCode == ZERO_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_add_scalar/add_scalar_list/expm1/sqrt/zero_inplace
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == SOLO_LOG_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_log/log1p/log10
        uint32_t totalSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == BINARY_LIST_OP_CODE) {
        // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
        // foreach_div_list/minimum_list/mul_list/sub_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_POINTWISE_OP_CODE) {
        // foreach_addcdiv_scalar/addcdiv_scalar_list/addcmul_scalar/addcmul_scalar_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_POINTWISE_DIVIDER; // double buffer
        inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory2(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_COS_OP_CODE) {  // foreach_cos
        uint32_t tilingConstant = 6;
        if (dataTypeSize == BYTE_LEN_4) {
            tilingConstant = TILING_FLOAT_N_SCALAR;
        }
        uint32_t reserveUbSize = BYTE_BASIC_BLOCK * tilingConstant * dataTypeSize;
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - reserveUbSize);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == SOLO_LOG2_OP_CODE) {  // foreach_log2
        uint32_t extraBuf = 0;      // need extra space
        GetLog2TmpBufferFactorSize(dataTypeSize, extraBuf, LOG2_HALF_FOR_LOG2, LOG2_FLOAT_FOR_LOG2, LOG2_BASIC_FOR_LOG2); // reuse source is true
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2 - extraBuf;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == SOLO_NEG_OP_CODE) {  // need extra buffer of one block: 32 bytes  foreach_neg/reciprocal
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2 - BYTE_PER_BLOCK;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_POW_TENSOR_OP_CODE) { // foreach_pow_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        uint32_t canUseUbSize;
        if (dataType == ge::DT_BF16) {
            canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER * UB_DIVIDER_FOR_TEMP_CASTING + POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType)-1]);
        } else{
            canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER + POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType)-1]);
        }
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory3(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_BINARY_SCALAR_OP_CODE) {
        // foreach_maximum_scalar/maximum_scalar_list/minimum_scalar/minimum_scalar_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / BINARY_SCALAR_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_POINTWISE_LIST_OP_CODE) {
        // foreach_addcdiv_list, foreach_addcmul_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_POINTWISE_LIST_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_SIGMOID_OP_CODE) {
        // foreach_sigmoid
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 1024);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / BINARY_SCALAR_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_ERF_OP_CODE) {
        // foreach_erf
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        // erf ascend C need 3 times for float or 8 times for half inputData size reserved for every buffer 
        uint32_t canUseUbSize = totalSize / FOREACH_ERF_FLOAT_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
        if (dataTypeSize == BYTE_LEN_2) {
            canUseUbSize = totalSize / FOREACH_ERF_HALF_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
        }
        // 32 bytes align
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory4(uint64_t ubSizePlatForm) {
    if ((opCode == FOREACH_ASIN_OP_CODE)) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_cosh/asin/acos
        uint32_t calcPro = COSH_HALF_CALC_PROC;
        if (dataTypeSize == BYTE_LEN_4) {
            calcPro = COSH_FLOAT_CALC_PROC;
        }
        uint32_t extraBuffer = calcPro * dataTypeSize * COSH_BASIC_BLOCK_SIZE * 8;
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_SINH_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_sinh
        uint32_t calcPro = SINH_HALF_CALC_PROC;
        if (dataTypeSize == BYTE_LEN_4) {
            calcPro = SINH_FLOAT_CALC_PROC;
        }
        uint32_t extraBuffer = calcPro * dataTypeSize * SINH_BASIC_BLOCK_SIZE;
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_TAN_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_tan
        uint32_t calcPro = TAN_HALF_CALC_PROC;
        if (dataTypeSize == BYTE_LEN_4) {
            calcPro = TAN_FLOAT_CALC_PROC;
        }
        uint32_t extraBuffer = calcPro * dataTypeSize * TAN_BASIC_BLOCK_SIZE * 8;
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory5(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_ERFC_OP_CODE) {
        // foreach_erfc
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        // erfc ascend C need 7 times for float or 16 times for half inputData size reserved for every buffer
        uint32_t canUseUbSize = totalSize / FOREACH_ERFC_FLOAT_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
        if (dataTypeSize == BYTE_LEN_2) {
            canUseUbSize = totalSize / FOREACH_ERFC_HALF_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
        }
        // 32 bytes align
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_TANH_OP_CODE) {
        // foreach_tanh
        uint32_t calcPro = TANH_FLOAT_CALC_PROC;
        if (dataTypeSize == BYTE_LEN_2) {
            calcPro = TANH_HALF_CALC_PROC;
        }
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 1024);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / (calcPro + FOREACH_TANH_DIVIDER);
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_ATAN_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_atan
        uint32_t calcPro = ATAN_HALF_CALC_PROC;
        if (dataTypeSize == BYTE_LEN_4) {
            calcPro = ATAN_FLOAT_CALC_PROC;
        }
        uint32_t extraBuffer = calcPro * dataTypeSize * ATAN_BASIC_BLOCK_SIZE * 8;
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory6(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_LERP_SCALAR_OP_CODE) {
        // foreach_lerp_scalar
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 128);
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_LERP_SCALAR_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_LERP_LIST_OP_CODE) {
        // foreach_lerp_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_LERP_LIST_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        inputsTensorUbSize = inputsTensorUbSize / BYTE_PER_REPEAT * BYTE_PER_REPEAT;
    } else if ((opCode == FOREACH_POW_SCALAR_OP_CODE) || (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE)) {
        // foreach_pow_scalar/pow_scalar_list/pow_scalar_and_tensor
        uint32_t reserveUbSize = BYTE_BASIC_BLOCK * GetTilingN() * dataTypeSize;
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - reserveUbSize);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / FOREACH_POW_SCALAR_DIVIDER; // double buffer
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory7(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_SIN_OP_CODE) {
        // foreach_sin
        uint32_t calcPro = SIN_HALF_CALC_FAC;
        if (dataTypeSize == BYTE_LEN_4) {
            calcPro = SIN_FLOAT_CALC_FAC;
        }
        uint32_t reservedUbSize = 4 * SIN_BASIC_BLOCK * calcPro * dataTypeSize;
        uint32_t totalSize = static_cast<uint32_t>(ubSizePlatForm - static_cast<uint32_t>(tilingData.GetDataSize()) - reservedUbSize);
        if (dataType == ge::DT_BF16) {
            totalSize = static_cast<uint32_t>(totalSize / UB_DIVIDER_FOR_TEMP_CASTING);
        }
        uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / FOREACH_SIN_DIVIDER); // 4
        inputsTensorUbSize = static_cast<uint32_t>(canUseUbSize / BYTE_BLOCK * BYTE_BLOCK);
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_ABS_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_abs
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 2048);
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / BYTE_LEN_4;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_MUL_SCALAR_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_mul_scalar/mul_scalar_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory8(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_EXP_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_exp
        uint32_t totalSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_MAXIMUM_LIST_OP_CODE) {
        // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
        // foreach_maximum_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_ADD_LIST_OP_CODE) {
        // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
        // foreach_add_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory9(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_ROUND_OFF_NUM_OP_CODE) {
        // foreach_round_off_number
        uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / 2;
        uint32_t predictSGUbSize = uint32_t(BYTE_REPEAT / (BYTE_REPEAT + 2.0 * dataTypeSize) * canUseUbSize);
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            predictSGUbSize = predictSGUbSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
            predictSGUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_SUB_SCALAR_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_sub_scalar/sub_scalar_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - BYTE_BLOCK);
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 4; // one block bytes is 32
        inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_DIV_SCALAR_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_div_scalar/div_scalar_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - BYTE_BLOCK);
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 4; // one block bytes is 32
        inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    } else if (opCode == FOREACH_COPY_OP_CODE) {
        // The remaining UB size is one buffer enabled, and rounded down 32 bytes.
        // foreach_copy
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize;
        inputsTensorUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

void ForeachCommonTiling::DivideUbMemory10(uint64_t ubSizePlatForm) {
    if (opCode == FOREACH_SIGN_OP_CODE) {
        // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
        // foreach_sign
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_FLOAT || dataType == ge::DT_FLOAT16) {
            uint32_t extraBuffer = SIGN_CALC_PROC * dataTypeSize * SIGN_BASIC_BLOCK_SIZE * 8;
            totalSize = totalSize - extraBuffer;
        }
        if (dataType == ge::DT_BF16 || dataType == ge::DT_INT64 || dataType == ge::DT_INT8) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 4; // one block bytes is 32
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? 
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }
}

static ge::graphStatus Tiling4ForeachAbsTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ABS_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachCopyTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_COPY_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachSignTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_SIGN_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAcosTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ACOS_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ADD_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(ZERO_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(ZERO_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddcdivListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POINTWISE_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddcdivScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POINTWISE_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddcdivScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POINTWISE_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddcmulListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POINTWISE_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddcmulScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POINTWISE_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachAddcmulScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POINTWISE_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAsinTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ASIN_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachAtanTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ATAN_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachCosTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_COS_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachCoshTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_COSH_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachDivListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(BINARY_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachDivScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_DIV_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachDivScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_DIV_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachErfTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ERF_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachErfcTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ERFC_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachExpTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_EXP_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachExpm1Tiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(ZERO_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachLerpListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_LERP_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachLerpScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_LERP_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachLogTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(SOLO_LOG_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachLog1pTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(SOLO_LOG_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachLog2Tiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(SOLO_LOG2_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachLog10Tiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(SOLO_LOG_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachMaximumListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_MAXIMUM_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachMaximumScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_BINARY_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachMaximumScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_BINARY_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachMinimumListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(BINARY_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachMinimumScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_BINARY_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachMinimumScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_BINARY_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachMulListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(BINARY_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachMulScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_MUL_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachMulScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_MUL_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachNegTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(SOLO_NEG_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachPowListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POW_TENSOR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachPowScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POW_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachPowScalarAndTensorTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachPowScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POW_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachReciprocalTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(SOLO_NEG_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachRoundOffNumberTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_ROUND_OFF_NUM_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachSigmoidTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_SIGMOID_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachSinTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_SIN_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachSinhTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_SINH_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachSqrtTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(ZERO_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachSubListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(BINARY_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachSubScalarTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_SUB_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus Tiling4ForeachSubScalarListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_SUB_SCALAR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachTanTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_TAN_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachTanhTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_TANH_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus Tiling4ForeachZeroInplaceTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(ZERO_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepare4ForeachTiling(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ForeachScalarTiling(gert::TilingParseContext* context) {
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    // OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<ForeachCompileInfo>();
    // OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    compileInfoPtr->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicCoreNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachAbs)
    .Tiling(Tiling4ForeachAbsTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachCopy)
    .Tiling(Tiling4ForeachCopyTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachSign)
    .Tiling(Tiling4ForeachSignTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAcos)
    .Tiling(Tiling4ForeachAcosTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAddList)
.Tiling(Tiling4ForeachAddListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachAddScalar)
.Tiling(Tiling4ForeachAddScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachAddScalarList)
.Tiling(Tiling4ForeachAddScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAddcdivList)
.Tiling(Tiling4ForeachAddcdivListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAddcdivScalar)
.Tiling(Tiling4ForeachAddcdivScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachAddcdivScalarList)
.Tiling(Tiling4ForeachAddcdivScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAddcmulList)
.Tiling(Tiling4ForeachAddcmulListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAddcmulScalar)
.Tiling(Tiling4ForeachAddcmulScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachAddcmulScalarList)
.Tiling(Tiling4ForeachAddcmulScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAsin)
.Tiling(Tiling4ForeachAsinTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachAtan)
.Tiling(Tiling4ForeachAtanTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachCos)
.Tiling(Tiling4ForeachCosTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachCosh)
.Tiling(Tiling4ForeachCoshTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachDivList)
.Tiling(Tiling4ForeachDivListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachDivScalar)
.Tiling(Tiling4ForeachDivScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachDivScalarList)
.Tiling(Tiling4ForeachDivScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachErf)
.Tiling(Tiling4ForeachErfTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachErfc)
.Tiling(Tiling4ForeachErfcTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachExp)
.Tiling(Tiling4ForeachExpTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachExpm1)
.Tiling(Tiling4ForeachExpm1Tiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachLerpList)
.Tiling(Tiling4ForeachLerpListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachLerpScalar)
.Tiling(Tiling4ForeachLerpScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachLog)
.Tiling(Tiling4ForeachLogTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachLog1p)
.Tiling(Tiling4ForeachLog1pTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachLog2)
.Tiling(Tiling4ForeachLog2Tiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachLog10)
.Tiling(Tiling4ForeachLog10Tiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachMaximumList)
.Tiling(Tiling4ForeachMaximumListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachMaximumScalar)
.Tiling(Tiling4ForeachMaximumScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachMaximumScalarList)
.Tiling(Tiling4ForeachMaximumScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachMinimumList)
.Tiling(Tiling4ForeachMinimumListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachMinimumScalar)
.Tiling(Tiling4ForeachMinimumScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachMinimumScalarList)
.Tiling(Tiling4ForeachMinimumScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachMulList)
.Tiling(Tiling4ForeachMulListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachMulScalar)
.Tiling(Tiling4ForeachMulScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachMulScalarList)
.Tiling(Tiling4ForeachMulScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachNeg)
.Tiling(Tiling4ForeachNegTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachPowList)
.Tiling(Tiling4ForeachPowListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachPowScalar)
.Tiling(Tiling4ForeachPowScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachPowScalarAndTensor)
.Tiling(Tiling4ForeachPowScalarAndTensorTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachPowScalarList)
.Tiling(Tiling4ForeachPowScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachReciprocal)
.Tiling(Tiling4ForeachReciprocalTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachRoundOffNumber)
.Tiling(Tiling4ForeachRoundOffNumberTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachSigmoid)
.Tiling(Tiling4ForeachSigmoidTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachSin)
.Tiling(Tiling4ForeachSinTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachSinh)
.Tiling(Tiling4ForeachSinhTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachSqrt)
.Tiling(Tiling4ForeachSqrtTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachSubList)
.Tiling(Tiling4ForeachSubListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachSubScalar)
.Tiling(Tiling4ForeachSubScalarTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

IMPL_OP_OPTILING(ForeachSubScalarList)
.Tiling(Tiling4ForeachSubScalarListTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachTan)
.Tiling(Tiling4ForeachTanTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachTanh)
.Tiling(Tiling4ForeachTanhTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

IMPL_OP_OPTILING(ForeachZeroInplace)
.Tiling(Tiling4ForeachZeroInplaceTiling)
.TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

}  // namespace optiling

