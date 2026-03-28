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
 * \file add_layer_norm_grad_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_LAYER_NORM_GRAD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_LAYER_NORM_GRAD_H_
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddLayerNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
TILING_DATA_FIELD_DEF(uint32_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint32_t, nInOneCoreLength);
TILING_DATA_FIELD_DEF(uint32_t, nInOneCoreLengthTail);
TILING_DATA_FIELD_DEF(uint32_t, ndInOneCoreLength);
TILING_DATA_FIELD_DEF(uint32_t, nAvailInUb);
TILING_DATA_FIELD_DEF(uint32_t, dInnerLength);
TILING_DATA_FIELD_DEF(uint32_t, dInnerLengthTail);
TILING_DATA_FIELD_DEF(uint32_t, dOuterLength);
TILING_DATA_FIELD_DEF(uint32_t, nInOneCoreNorm);
TILING_DATA_FIELD_DEF(uint32_t, gmOneCoreElemXYNorm);
TILING_DATA_FIELD_DEF(uint32_t, nAvailInUbNorm);
TILING_DATA_FIELD_DEF(uint32_t, nMiddleCountNorm);
TILING_DATA_FIELD_DEF(uint32_t, ndRoundUpDtypeNorm);
TILING_DATA_FIELD_DEF(uint32_t, n1RoundUpFloatNorm);
TILING_DATA_FIELD_DEF(uint32_t, nInUbTotalNormTail);
TILING_DATA_FIELD_DEF(uint32_t, nInOneCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, gmOneCoreElemXYTail);
TILING_DATA_FIELD_DEF(uint32_t, nAvailInUbTail);
TILING_DATA_FIELD_DEF(uint32_t, nMiddleCountTail);
TILING_DATA_FIELD_DEF(uint32_t, ndRoundUpDtypeTail);
TILING_DATA_FIELD_DEF(uint32_t, n1RoundUpFloatTail);
TILING_DATA_FIELD_DEF(uint32_t, nInUbTotalTailTail);
TILING_DATA_FIELD_DEF(uint32_t, dyPadRight);
TILING_DATA_FIELD_DEF(uint32_t, rstdPadRight);
TILING_DATA_FIELD_DEF(uint32_t, roundUpNumLastDim);
TILING_DATA_FIELD_DEF(uint32_t, roundUpNumLastDimDtype);
TILING_DATA_FIELD_DEF(uint32_t, roundUp1Dtype);
TILING_DATA_FIELD_DEF(uint32_t, roundUpNumLastDimFloat);
END_TILING_DATA_DEF;

struct TilingStruct {
    uint32_t numCore;
    uint32_t numLastDim;
    uint32_t numFirstDim;
    uint32_t nInOneCoreLength;
    uint32_t nInOneCoreLengthTail;
    uint32_t ndInOneCoreLength;
    uint32_t nAvailInUb;
    uint32_t dInnerLength;
    uint32_t dInnerLengthTail;
    uint32_t dOuterLength;
    uint32_t nInOneCoreNorm;
    uint32_t gmOneCoreElemXYNorm;
    uint32_t nAvailInUbNorm;
    uint32_t nMiddleCountNorm;
    uint32_t ndRoundUpDtypeNorm;
    uint32_t n1RoundUpFloatNorm;
    uint32_t nInUbTotalNormTail;
    uint32_t nInOneCoreTail;
    uint32_t gmOneCoreElemXYTail;
    uint32_t nAvailInUbTail;
    uint32_t nMiddleCountTail;
    uint32_t ndRoundUpDtypeTail;
    uint32_t n1RoundUpFloatTail;
    uint32_t nInUbTotalTailTail;
    uint32_t dyPadRight;
    uint32_t rstdPadRight;
    uint32_t roundUpNumLastDim;
    uint32_t roundUpNumLastDimDtype;
    uint32_t roundUp1Dtype;
    uint32_t roundUpNumLastDimFloat;
};

struct AddLayerNormGradCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

REGISTER_TILING_DATA_CLASS(AddLayerNormGrad, AddLayerNormGradTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_LAYER_NORM_GRAD_H_