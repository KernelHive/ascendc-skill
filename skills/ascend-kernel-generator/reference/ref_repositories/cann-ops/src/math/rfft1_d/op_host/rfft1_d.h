/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file rfft1d_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RFFT1D_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RFFT1D_H_

#pragma once
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
//  #include "op_tiling_util.h"
//  #include "register/op_compile_info_base.h"
#include "tiling/tiling_base.h"
//  #include "runtime/runtime2_util.h"
 
namespace optiling {

struct Rfft1DCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

BEGIN_TILING_DATA_DEF(Rfft1DTilingData)
    TILING_DATA_FIELD_DEF(int32_t, length);
    TILING_DATA_FIELD_DEF(uint8_t, isBluestein);
    TILING_DATA_FIELD_DEF(int32_t, lengthPad);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, factors);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, prevRadices);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, nextRadices);
    TILING_DATA_FIELD_DEF_ARR(uint8_t, 3, prevRadicesAlign);
    TILING_DATA_FIELD_DEF(int32_t, outLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailSize); 
    
    TILING_DATA_FIELD_DEF(uint32_t, batchesPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, leftOverBatches);
    
    TILING_DATA_FIELD_DEF(uint32_t, tmpLenPerBatch); 
    TILING_DATA_FIELD_DEF(uint32_t, tmpSizePerBatch);
    TILING_DATA_FIELD_DEF(uint32_t, matmulTmpsLen);
    TILING_DATA_FIELD_DEF(uint32_t, matmulTmpsSize);
  
    TILING_DATA_FIELD_DEF(int32_t, normal);
  
    TILING_DATA_FIELD_DEF(uint32_t, dftRealOverallSize);
    TILING_DATA_FIELD_DEF(uint32_t, dftImagOverallSize);
    TILING_DATA_FIELD_DEF(uint32_t, twiddleOverallSize);
    TILING_DATA_FIELD_DEF(uint32_t, fftMatrOverallSize);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, dftRealOffsets);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, dftImagOffsets);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, twiddleOffsets);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Rfft1D, Rfft1DTilingData);

class Rfft1DBaseTiling : public TilingBaseClass {
    public:
    explicit Rfft1DBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {
    }

    protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    
    protected:
    uint32_t ubSize;
    uint32_t coreNum;
    uint32_t batches;
    int32_t length;
    int32_t normal;
    ge::DataType dtype;
};
}  // namespace optiling
#endif
 