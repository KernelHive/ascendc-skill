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
 * \file reverse_sequence_batch_0.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_BATCH_0_H
#define REVERSE_SEQUENCE_BATCH_0_H

#include "reverse_sequence_base.h"

namespace ReverseSequence {
using namespace AscendC;

template <typename T, bool isCSmall = true>
class ReverseSequenceBatch0 : public ReverseSequenceBase<T> {
public:
    __aicore__ inline ReverseSequenceBatch0(const ReverseSequenceTilingData* tilingDataPtr)
        : ReverseSequenceBase<T>(tilingDataPtr){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void ReverseSeq(const int64_t batchIndex, const int64_t seqGroupIndex,
                                      const int64_t reverseCount);
};

template <typename T, bool isCSmall>
__aicore__ inline void ReverseSequenceBatch0<T, isCSmall>::Init(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y,
                                                                GM_ADDR workspace) {
    this->BaseInit(seq_lengths, workspace);
    int64_t curCoreGmStart = this->batchBaseIndex_ * this->seqSize_ * this->cSize_ * this->xDtypeSize_;
    this->xGM_.SetGlobalBuffer(x + curCoreGmStart);
    this->yGM_.SetGlobalBuffer(y + curCoreGmStart);
}

template <typename T, bool isCSmall>
__aicore__ inline void ReverseSequenceBatch0<T, isCSmall>::ReverseSeq(const int64_t batchIndex,
                                                                      const int64_t seqGroupIndex,
                                                                      const int64_t reverseCount) {
    int64_t prefixOffset = batchIndex * this->seqSize_ + seqGroupIndex * this->seqDimValue_;
    int64_t gmInOffset = prefixOffset * this->cSize_;
    int64_t gmOutOffset = (prefixOffset + reverseCount - 1) * this->cSize_;
    for (int64_t i = 0; i < reverseCount; i++) {
        if constexpr (isCSmall) {
            this->CopyInX(gmInOffset, this->cSize_);
            this->CopyOutX(gmOutOffset, this->cSize_);
        } else {
            this->SingleReverseCBig(gmInOffset, gmOutOffset);
        }
        gmInOffset += this->cSize_;
        gmOutOffset -= this->cSize_;
    }
    for (int64_t i = reverseCount; i < this->seqDimValue_; i++) {
        if constexpr (isCSmall) {
            this->CopyInX(gmInOffset, this->cSize_);
            this->CopyOutX(gmInOffset, this->cSize_);
        } else {
            this->SingleCopyCBig(gmInOffset);
        }
        gmInOffset += this->cSize_;
    }
}

}  // namespace ReverseSequence

#endif  // REVERSE_SEQUENCE_BATCH_0_H