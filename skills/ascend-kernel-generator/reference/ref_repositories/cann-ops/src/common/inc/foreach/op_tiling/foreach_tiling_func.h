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
 * \file foreach_tiling_func.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_FUNC_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_FUNC_H_

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "foreach_tiling_def.h"
#include "common_dtype.h"

namespace optiling {
    constexpr uint64_t TILING_HALF_N_SCALAR = 14;
    constexpr uint64_t TILING_FLOAT_N_SCALAR = 4;
    constexpr uint64_t TILING_INT_N_SCALAR = 4;
    constexpr uint64_t TILING_BF16_N_SCALAR = 14;
    constexpr uint32_t TILING_FLOAT_ERF = 5;
    constexpr uint32_t TILING_HALF_ERF = 12;

    constexpr uint64_t WORK_SPACE_SIZE = 32;// foreach(vector) not need workspace

    constexpr uint32_t TANH_HALF_CALC_PROC = 5;
    constexpr uint32_t TANH_FLOAT_CALC_PROC = 6;
    constexpr uint32_t FOREACH_TANH_DIVIDER = 2;
    constexpr uint32_t SIN_HALF_CALC_FAC = 6;
    constexpr uint32_t SIN_FLOAT_CALC_FAC = 2;
    constexpr uint32_t SIN_BASIC_BLOCK = 2048;

    constexpr uint32_t COSH_HALF_CALC_PROC = 6;
    constexpr uint32_t COSH_FLOAT_CALC_PROC = 2;
    constexpr uint32_t COSH_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t SINH_HALF_CALC_PROC = 4;
    constexpr uint32_t SINH_FLOAT_CALC_PROC = 1;
    constexpr uint32_t SINH_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t ATAN_HALF_CALC_PROC = 10;
    constexpr uint32_t ATAN_FLOAT_CALC_PROC = 4;
    constexpr uint32_t ATAN_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t TAN_HALF_CALC_PROC = 10;
    constexpr uint32_t TAN_FLOAT_CALC_PROC = 4;
    constexpr uint32_t TAN_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t SIGN_CALC_PROC = 3;
    constexpr uint32_t SIGN_BASIC_BLOCK_SIZE = 1024;

    constexpr uint32_t BINARY_LIST_UB_DIVIDER = 6;
    constexpr uint32_t BINARY_SCALAR_UB_DIVIDER = 4;
    constexpr uint32_t FOREACH_POINTWISE_DIVIDER = 8;
    constexpr uint32_t FOREACH_POW_SCALAR_DIVIDER = 4;
    constexpr uint32_t FOREACH_COS_DIVIDER = 4;
    constexpr uint32_t FOREACH_POINTWISE_LIST_DIVIDER = 8;
    constexpr uint32_t FOREACH_LERP_SCALAR_UB_DIVIDER = 6;
    constexpr uint32_t FOREACH_LERP_LIST_UB_DIVIDER = 11;
    constexpr uint32_t FOREACH_SIN_DIVIDER = 4;
    constexpr uint32_t FOREACH_ERF_BUFFER_DIVIDER = 4;
    constexpr uint32_t FOREACH_ERF_FLOAT_DIVIDER = 4; // erf float 预留 3 倍的输入空间
    constexpr uint32_t FOREACH_ERF_HALF_DIVIDER = 9; // erf half 预留 8 倍的输入空间
    constexpr uint32_t FOREACH_ERFC_FLOAT_DIVIDER = 8; // erfc float 预留 7 倍的输入空间
    constexpr uint32_t FOREACH_ERFC_HALF_DIVIDER = 17; // erfc half 预留 16 倍的输入空间

    constexpr uint8_t ZERO_OP_CODE = 1;
    constexpr uint8_t SOLO_LOG_OP_CODE = 2;
    constexpr uint8_t BINARY_LIST_OP_CODE = 3;
    constexpr uint8_t FOREACH_POINTWISE_OP_CODE = 4;
    constexpr uint8_t FOREACH_COS_OP_CODE = 5;
    constexpr uint8_t SOLO_LOG2_OP_CODE = 6;
    constexpr uint8_t SOLO_NEG_OP_CODE = 7;
    constexpr uint8_t FOREACH_POW_TENSOR_OP_CODE = 8;
    constexpr uint8_t FOREACH_BINARY_SCALAR_OP_CODE = 9;
    constexpr uint8_t FOREACH_POINTWISE_LIST_OP_CODE = 10;
    constexpr uint8_t FOREACH_SIGMOID_OP_CODE = 11;
    constexpr uint8_t FOREACH_ERF_OP_CODE = 12;
    constexpr uint8_t FOREACH_COSH_OP_CODE = 13;
    constexpr uint8_t FOREACH_ASIN_OP_CODE = 13;
    constexpr uint8_t FOREACH_ACOS_OP_CODE = 13;
    constexpr uint8_t FOREACH_SINH_OP_CODE = 14;
    constexpr uint8_t FOREACH_TAN_OP_CODE = 15;
    constexpr uint8_t FOREACH_ERFC_OP_CODE = 16;
    constexpr uint8_t FOREACH_TANH_OP_CODE= 17;
    constexpr uint8_t FOREACH_ATAN_OP_CODE = 18;
    constexpr uint8_t FOREACH_LERP_SCALAR_OP_CODE = 19;
    constexpr uint8_t FOREACH_LERP_LIST_OP_CODE = 20;
    constexpr uint8_t FOREACH_POW_SCALAR_OP_CODE = 21;
    constexpr uint8_t FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE = 22;
    constexpr uint8_t FOREACH_SIN_OP_CODE = 23;
    constexpr uint8_t FOREACH_ABS_OP_CODE = 24;
    constexpr uint8_t FOREACH_MUL_SCALAR_OP_CODE = 25;
    constexpr uint8_t FOREACH_EXP_OP_CODE = 26;
    constexpr uint8_t FOREACH_MAXIMUM_LIST_OP_CODE = 27;
    constexpr uint8_t FOREACH_ADD_LIST_OP_CODE = 28;
    constexpr uint8_t FOREACH_ROUND_OFF_NUM_OP_CODE = 29;
    constexpr uint8_t FOREACH_SUB_SCALAR_OP_CODE = 30;
    constexpr uint8_t FOREACH_DIV_SCALAR_OP_CODE = 31;
    constexpr uint8_t FOREACH_COPY_OP_CODE = 32;
    constexpr uint8_t FOREACH_SIGN_OP_CODE = 33;

    constexpr uint16_t LOG2_BASIC_FOR_LOG2 = 1024;
    constexpr uint32_t LOG2_HALF_FOR_LOG2 = 4;
    constexpr uint32_t LOG2_FLOAT_FOR_LOG2 = 0;

    constexpr uint8_t BYTE_PER_BLOCK = 32;
    constexpr uint32_t BYTE_PER_REPEAT = 256;
    constexpr int32_t POW_TENSOR_TENSOR_CALC_PROC[9] = {12, 3, 5, 3, 12, 12, 12, 12, 12};

    constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class ForeachCommonTiling {
public:
    explicit ForeachCommonTiling(gert::TilingContext* context) : tilingContext(context){};

    ge::graphStatus Init(uint8_t theCode = 0);

    ge::graphStatus RunBigKernelTiling();
    ge::graphStatus RunBigScalarKernelTiling();
private:
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    /**
     ** function: GetTilingN
    */
    uint64_t GetTilingN() {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_FLOAT_N_SCALAR;
            case ge::DT_FLOAT16:
                return TILING_HALF_N_SCALAR;
            case ge::DT_INT32:
                return TILING_INT_N_SCALAR;
            case ge::DT_BF16:
                return TILING_BF16_N_SCALAR;
            default:
                return TILING_HALF_N_SCALAR;
        }
    }

    /**
     ** function: GetNeedCoreNum
    */
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) {
        uint32_t tempCoreNum = (uint32_t)CeilA2B(totalDataCount, elementsPerBlock);
        if (tempCoreNum == 0) {
            tempCoreNum = 1;
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    /**
     ** function: FillTilingData
    */
    void FillTilingData() {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_tensorDataCountList(tensorDataCountList);
        tilingData.set_tensorStartList(tensorStartList);
        tilingData.set_tensorEndList(tensorEndList);
        tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
        tilingData.set_tensorEndOffsetList(tensorEndOffsetList);

        tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                                tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

    /**
     ** function: GetLog2TmpBufferFactorSize
    */
    void GetLog2TmpBufferFactorSize(const uint32_t typeSize, uint32_t &extraBuf,
                                    uint32_t LOG2_HALF = LOG2_HALF_FOR_LOG2, uint32_t LOG2_FLOAT = LOG2_FLOAT_FOR_LOG2,
                                    uint32_t LOG2_BASIC = LOG2_BASIC_FOR_LOG2) {
        auto caclFactor = (typeSize == sizeof(float)) ? LOG2_FLOAT : LOG2_HALF;
        extraBuf = LOG2_BASIC * caclFactor * typeSize;
    }

    void AssignDataToEachCore(int64_t needCoreNum);
    void DivideUbMemory(uint64_t ubSizePlatForm);
    void DivideUbMemory1(uint64_t ubSizePlatForm);
    void DivideUbMemory2(uint64_t ubSizePlatForm);
    void DivideUbMemory3(uint64_t ubSizePlatForm);
    void DivideUbMemory4(uint64_t ubSizePlatForm);
    void DivideUbMemory5(uint64_t ubSizePlatForm); 
    void DivideUbMemory6(uint64_t ubSizePlatForm);
    void DivideUbMemory7(uint64_t ubSizePlatForm);
    void DivideUbMemory8(uint64_t ubSizePlatForm);
    void DivideUbMemory9(uint64_t ubSizePlatForm);
    void DivideUbMemory10(uint64_t ubSizePlatForm);

private:
    ForeachCommonTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    int64_t totalDataCount = 0;
    uint8_t dataTypeSize = 4;
    uint8_t elementsPerBlock = 0;
    uint16_t totalTensorCount = 0;
    uint8_t opCode = 0;
};
}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_FUNC_H_
