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
 * \file conv_util.h
 * \brief
 */

#ifndef CONV_UTIL_H
#define CONV_UTIL_H

#include "kernel_utils.h"

using namespace AscendC;

namespace conv {
const uint32_t K0_BIAS = 8;
const uint32_t NUM_TWO = 2;
const uint32_t BLOCK_L0_N = 16;
const uint32_t BLOCK_L0_M = 16;
const uint32_t DATA_COPY_OP_LEN = 16;
const uint64_t C0_SIZE = 32;
const uint64_t FRACTAL_SIZE = 512;
const uint64_t BT_SIZE = 64;
const uint64_t SIZE_OF_FP16 = 2;
const uint64_t PAD_IDX_T = 2;
const uint64_t PAD_IDX_B = 3;
const uint64_t PAD_IDX_L = 0;
const uint64_t PAD_IDX_R = 1;
const uint64_t MAX_PAD_R = 255;
const uint64_t FMAP_BATCH_DIM = 0;
const uint64_t FMAP_CIN_DIM = 1;
const uint64_t FMAP_H_DIM = 2;
const uint64_t FMAP_W_DIM = 3;
const uint64_t KERNEL_COUT_DIM = 0;
const uint64_t KERNEL_H_DIM = 2;
const uint64_t KERNEL_W_DIM = 3;
const uint64_t OUTPUT_H_DIM = 2;
const uint64_t OUTPUT_W_DIM = 3;
const uint32_t BLOCK_SIZE = 512;
const uint32_t AL1_BLOCK_SIZE = 32;
const uint32_t BT_BLOCK_SIZE = 64;
const uint64_t F8_NUM_IN_F16 = 2;
const uint64_t MAX_UINT16 = 65535;
const uint64_t INT32_OUT_16_FOR_8 = 2;

__aicore__ inline size_t AlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return ((a + b - 1UL) / b) * b;
}

__aicore__ inline size_t CeilDIV(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1UL) / b;
}

template <class Intf>
__aicore__ inline size_t GetInputkInOneC0Block()
{
    return C0_SIZE / sizeof(typename Intf::FmapT);
}

enum class IterateMNOrder {
    ORDER_MTERFIRST = 0,
    ORDER_NTERFIRST,
    UNDEF,
};
}  // namespace conv
#endif  // __CONV3D_UTIL_H__
