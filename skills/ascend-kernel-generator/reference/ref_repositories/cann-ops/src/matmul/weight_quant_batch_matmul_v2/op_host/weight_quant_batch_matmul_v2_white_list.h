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
 * \file weight_quant_batch_matmul_v2_white_list.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H

namespace optiling {

class WhiteListShape {
public:
    bool operator<(const WhiteListShape &right) const { return memcmp(this, &right, sizeof(WhiteListShape)) < 0; }

    uint64_t mSize_;
    uint64_t kSize_;
    uint64_t nSize_;
    bool hasBias_;
    bool transA_;
    bool transB_;
    uint64_t aicNum_ : 40;
};

class MatMulTilingCache {
public:
    int32_t nDim_;
    int32_t mDim_;
    int32_t m_;
    int32_t n_;
    int32_t ka_;
    int32_t kb_;
    int32_t singleCoreM_;
    int32_t singleCoreN_;
    int32_t singleCoreK_;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    int32_t depthA1_;
    int32_t depthB1_;
    int32_t stepM_;
    int32_t stepN_;
    int32_t stepKa_;
    int32_t stepKb_;
    int32_t transLength_;
    int32_t iterateOrder_;
    int32_t shareL1Size_;
    int32_t shareL0CSize_;
    int32_t dbL0A_;
    int32_t dbL0B_;
    int32_t dbL0C_;

    void SetMatmulTilingFromCacheData(optiling::TCubeTiling &matmulTiling, uint64_t m, uint64_t n, int32_t isBias) const
    {
        matmulTiling.set_M(m);
        matmulTiling.set_N(n);
        matmulTiling.set_Ka(this->ka_);
        matmulTiling.set_Kb(this->kb_);
        matmulTiling.set_singleCoreM(ops::CeilDiv(m, static_cast<uint64_t>(this->mDim_)));
        matmulTiling.set_singleCoreN(this->singleCoreN_);
        matmulTiling.set_singleCoreK(this->singleCoreK_);
        matmulTiling.set_baseM(this->baseM_);
        matmulTiling.set_baseN(this->baseN_);
        matmulTiling.set_baseK(this->baseK_);
        matmulTiling.set_depthA1(this->depthA1_);
        matmulTiling.set_depthB1(this->depthB1_);
        matmulTiling.set_stepM(this->stepM_);
        matmulTiling.set_stepN(this->stepN_);
        matmulTiling.set_stepKa(this->stepKa_);
        matmulTiling.set_stepKb(this->stepKb_);
        matmulTiling.set_isBias(isBias);
        matmulTiling.set_transLength(this->transLength_);
        matmulTiling.set_iterateOrder(this->iterateOrder_);
        matmulTiling.set_shareL1Size(this->shareL1Size_);
        matmulTiling.set_shareL0CSize(this->shareL0CSize_);
        matmulTiling.set_dbL0A(this->dbL0A_);
        matmulTiling.set_dbL0B(this->dbL0B_);
        matmulTiling.set_dbL0C(this->dbL0C_);
        matmulTiling.set_usedCoreNum(1);
        matmulTiling.set_batchM(1);
        matmulTiling.set_batchN(1);
        matmulTiling.set_singleBatchM(1);
        matmulTiling.set_singleBatchN(1);
    }
};

}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H
