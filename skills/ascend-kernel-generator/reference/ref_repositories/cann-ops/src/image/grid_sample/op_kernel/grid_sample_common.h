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
 * \file grid_sample_common.h
 * \brief
 */
#ifndef GRID_SAMPLE_COMMON
#define GRID_SAMPLE_COMMON

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;
// const define
constexpr int64_t REFLECT_RATIO = 2;
constexpr int64_t PADDING_MODE_ZEROS = 0;
constexpr int64_t PADDING_MODE_BORDER = 1;
constexpr int64_t PADDING_MODE_REFLECTION = 2;
constexpr int64_t LAYOUT_NHWC = 1;

constexpr uint64_t B32_VECTOR_MASK = 64;
constexpr uint64_t B32_BLOCK_STRIDE = 1;
constexpr uint64_t B32_REPEAT_STRIDE = 8;
constexpr int64_t B32_ALIGN_FACTOR = 8;
constexpr int64_t B16_ALIGN_FACTOR = 16;

const int64_t BLOCK_SIZE = 32;

const int64_t TRANSE_REP_STRIDE = 128;
const int64_t B32_MASK = 64;
const int64_t CHANNEL_BLOCK = 64;
const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;

const int64_t CAL_D_H_W_BLOCK = 512;
const int64_t MASK_UB_SIZE = CAL_D_H_W_BLOCK / 8;

const int64_t NUM_0 = 0;
const int64_t NUM_1 = 1;
const int64_t NUM_2 = 2;
const int64_t NUM_3 = 3;
const int64_t NUM_4 = 4;
const int64_t NUM_5 = 5;

const int64_t MIN_CHANNEL_ALIGN = 8;
const int64_t REPEAT_TIME = 8;
const int64_t NUM_16 = 16;
const int64_t NUM_8 = 8;
const int64_t NUM_64 = 64;

const float FLOAT_2 = 2.0f;

}  // namespace GridSample
#endif  //  GRID_SAMPLER_2D_COMMON