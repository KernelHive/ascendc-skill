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
 * \file reverse_sequence.cpp
 * \brief
 */
#include "reverse_sequence_batch_0.h"
#include "reverse_sequence_batch_1.h"

#define BATCH_DIM_0_C_SMALL 101
#define BATCH_DIM_0_C_BIG 201
#define BATCH_DIM_1_C_SMALL 301
#define BATCH_DIM_1_C_BIG 401

using namespace ReverseSequence;

extern "C" __global__ __aicore__ void reverse_sequence(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y, GM_ADDR workspace,
                                                       GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(BATCH_DIM_0_C_SMALL)) {
        ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS>, &ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS>::ReverseSeq>(
            &op);
    } else if (TILING_KEY_IS(BATCH_DIM_0_C_BIG)) {
        ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS, false> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS, false>,
                   &ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS, false>::ReverseSeq>(&op);
    } else if (TILING_KEY_IS(BATCH_DIM_1_C_SMALL)) {
        ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS>, &ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS>::ReverseSeq>(
            &op);
    } else if (TILING_KEY_IS(BATCH_DIM_1_C_BIG)) {
        ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS, false> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS, false>,
                   &ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS, false>::ReverseSeq>(&op);
    }
}