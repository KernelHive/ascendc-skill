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
 * \file foreach_reduce_tiling_func.h
 * \brief
 */
 
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_FUNC_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_FUNC_H_

#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "foreach_reduce_tiling_def.h"
#include "common_dtype.h"

namespace optiling {
constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class ForeachReduceTiling {
public:
    explicit ForeachReduceTiling(gert::TilingContext* context) : tilingContext(context){};

    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    uint64_t GetTilingKeyVal() {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_KEY_FLOAT;
            case ge::DT_FLOAT16:
                return TILING_KEY_HALF;
            case ge::DT_BF16:
                return TILING_KEY_BF16;
            default:
                return 0;
        }
    }

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) {
        uint32_t tempCoreNum = (uint32_t)totalBlockCount;
        if (tempCoreNum == 0) {
            tempCoreNum = 1;
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    void FillTilingData() {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_needCoreNum(needCoreNum);
        tilingData.set_totalTensorCount(totalTensorCount);
        tilingData.set_tensorDataCountList(tensorDataCountList);
        tilingData.set_tensorStartList(tensorStartList);
        tilingData.set_tensorEndList(tensorEndList);
        tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
        tilingData.set_tensorEndOffsetList(tensorEndOffsetList);

        // Reduce Op Addition
        tilingData.set_tensorMiddleCountList(tensorMiddleCountList);
        tilingData.set_tensorMiddleStartList(tensorMiddleStartList);
        tilingData.set_coreMiddleOffsetList(coreMiddleOffsetList);

        tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                                tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

    void AssignDataToEachCore();
    void DivideUbMemory(uint64_t ubSizePlatForm);
    void AssignTensorMiddleCountList();

private:
    ForeachReduceTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    int64_t totalBlockCount = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;

    bool isExistEmptyTensor = false;

    uint32_t modelCode = 0;

    uint16_t tensorMiddleCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorMiddleStartList[MAX_TENSOR_CONT] = {0};
    uint16_t coreMiddleOffsetList[MAX_CORE_CONT] = {0};
};
} // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_FUNC_H_
