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
 * \file quant_batch_matmul_v3_tiling_cache.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_TILING_CACHE_H
#define QUANT_BATCH_MATMUL_V3_TILING_CACHE_H

#include "quant_batch_matmul_v3_basic_tiling.h"

namespace optiling {
struct QuantBatchMatmulV3BitField {
    // 这里要保证是32bit
    uint16_t transA : 1;
    uint16_t transB : 1;
    uint16_t hasBias : 1;
    uint16_t aFormatNd : 1;
    uint16_t bFormatNd : 1;
    uint16_t cFormatNd : 1;
    uint16_t isPerTensor : 1;
    uint16_t isPertoken : 1;
    uint32_t reserved : 24;
};

class QuantBatchMatmulV3HashInput {
public:
    explicit QuantBatchMatmulV3HashInput(const QuantBatchMatmulInfo &params);
    ~QuantBatchMatmulV3HashInput() = default;
    bool operator==(const QuantBatchMatmulV3HashInput &params) const
    {
        return memcmp(this, &params, sizeof(params)) == 0;
    }
    std::string ToString() const;

private:
    uint64_t mSize;
    uint64_t mSizePerNpu;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t batchA1;
    uint64_t batchA2;
    uint64_t batchA3;
    uint64_t batchA4;
    uint64_t batchB1;
    uint64_t batchB2;
    uint64_t batchB3;
    uint64_t batchB4;
    uint64_t batchBias;
    int32_t aDtype;
    int32_t bDtype;
    int32_t cDtype;
    int32_t biasDtype;
    int32_t scaleDtype;
    int32_t outDtype;
    QuantBatchMatmulV3BitField bitField;
};

class QuantBatchMatmulV3HashItem {
public:
    explicit QuantBatchMatmulV3HashItem(const QuantBatchMatmulInfo &params)
        : hashKey_(params)
    {
    }
    const QuantBatchMatmulV3HashInput &input() const { return hashKey_; }
    const BasicTiling &GetTiling() const { return tiling_; }
    void SetTiling(const BasicTiling &tiling) { tiling_ = tiling; }

private:
    QuantBatchMatmulV3HashInput hashKey_;
    BasicTiling tiling_;
};

using MMBasicTilingHash = cachetiling::TilingCache<QuantBatchMatmulV3HashInput, QuantBatchMatmulV3HashItem>;
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V3_TILING_CACHE_H
