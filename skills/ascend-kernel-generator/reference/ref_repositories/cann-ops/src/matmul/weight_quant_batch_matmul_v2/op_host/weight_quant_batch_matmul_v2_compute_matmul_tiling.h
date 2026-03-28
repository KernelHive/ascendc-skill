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
 * \file weight_quant_batch_matmul_v2_compute_matmul_tiling.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H

#include "weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

struct MatmulMultiCoreResult {
    uint8_t mDim;
    uint8_t nDim;
    uint8_t batchDim;
};

struct MatmulParams {
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    ge::DataType aDtype;
    ge::DataType bDtype;
    ge::DataType cDtype;
    ge::DataType biasDtype;
    bool transA;
    bool transB;
    bool hasBias;
    ge::Format format_a;
    ge::Format format_b;
    ge::Format format_out;
    QuantType quantType;
    bool kbAlign;
};

class ComputeMatmulTiling {
public:
    static bool GetTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult, const MatmulParams &params,
                   const AiCoreParams &aicoreParams, gert::TilingContext *context);

private:
    static bool GetCacheTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult, const MatmulParams &params,
                        gert::TilingContext *context);

    static void CalcCommonTiling(TCubeTiling &matmulTiling, const MatmulParams &params,
                                 const AiCoreParams &aicoreParams);

    static void CalcMsdBufferSize(TCubeTiling &matmulTiling, const MatmulParams &params);

    static bool MsdA16W8CommonTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult,
                                     const MatmulParams &params, const AiCoreParams &aicoreParams);

    static bool SimpleIncreTiling(TCubeTiling &matmulTiling, MatmulMultiCoreResult &multiCoreResult,
                           const MatmulParams &params, const AiCoreParams &aicoreParams);

    static void Convert2AscendCTiling(const Tiling &tbeTiling, TCubeTiling &matmulTiling, const MatmulParams &params,
                               MatmulMultiCoreResult &multiCoreResult);
    static MatrixTraverse GetIteratorOrder(const Tiling &tbeTiling, int32_t singleCoreM, int32_t singleCoreN,
                                    int32_t singleCoreK, ge::DataType aDtype);

    static bool tryComputeSimpleTiling(TCubeTiling &matmulTiling, const MatmulParams &params,
                                const AiCoreParams &aicoreParams);

    static bool tryAFullLoad(TCubeTiling &matmulTiling, const MatmulParams &params, const AiCoreParams &aicoreParams);

    static bool trySimpleTilingNormalLoad(TCubeTiling &matmulTiling, const MatmulParams &params,
                                   const AiCoreParams &aicoreParams);
};
}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H

