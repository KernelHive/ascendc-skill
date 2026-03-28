/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file apply_adam_wv2_tiling.h
 */
#ifndef APPLY_ADAM_W_V2_H_
#define APPLY_ADAM_W_V2_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"

namespace optiling {

const int64_t DTYPE_BF16_AND_STEP_FLOAT_KEY = 101;
const int64_t DTYPE_BF16_AND_STEP_INT64_KEY = 102;
const int64_t DTYPE_FP16_AND_STEP_FLOAT_KEY = 103;
const int64_t DTYPE_FP16_AND_STEP_INT64_KEY = 104;
const int64_t DTYPE_FP32_AND_STEP_FLOAT_KEY = 105;
const int64_t DTYPE_FP32_AND_STEP_INT64_KEY = 106;
const int64_t DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_FLOAT_KEY = 107;
const int64_t DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_INT64_KEY = 108;
const int64_t DTYPE_DIFF_DTYPE_GRAD_BF16_AND_STEP_FLOAT_KEY = 109;
const int64_t DTYPE_DIFF_DTYPE_GRAD_BF16_STEP_INT64_KEY = 110;

BEGIN_TILING_DATA_DEF(ApplyAdamWV2TilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, handleExtraLoopCoreNum);   // 前多少个核需要处理最后一次loop
    TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);              // 实际使用的核数
    TILING_DATA_FIELD_DEF(int64_t, numPerLoop);               // 每个loop处理多少数
    TILING_DATA_FIELD_DEF(int64_t, loopNumPerCore);           // 每个核处理多少次loop
    TILING_DATA_FIELD_DEF(int64_t, numLastLoop);              // 最后一个loop需要处理多少数
    TILING_DATA_FIELD_DEF(int64_t, isBfloat16);
    TILING_DATA_FIELD_DEF(float, lr);
    TILING_DATA_FIELD_DEF(float, beta1);
    TILING_DATA_FIELD_DEF(float, beta2);
    TILING_DATA_FIELD_DEF(float, weightDecay);
    TILING_DATA_FIELD_DEF(float, eps);
    TILING_DATA_FIELD_DEF(int64_t, amsgrad);
    TILING_DATA_FIELD_DEF(int64_t, maximize);
    TILING_DATA_FIELD_DEF(int64_t, tilingKey)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyAdamWV2, ApplyAdamWV2TilingData)

struct ApplyAdamWV2CompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

struct ApplyAdamWV2TilingParam {
  int64_t totalCoreNum;
  int64_t ubSize;
  int64_t handleExtraLoopCoreNum;
  int64_t usedCoreNum;
  int64_t numPerLoop;
  int64_t loopNumPerCore;
  int64_t numLastLoop;
  int64_t numLastLoopFor2ByteType{0};
  int64_t isBfloat16{0};
  float lr;
  float beta1;
  float beta2;
  float weightDecay;
  float eps;
  int64_t amsgrad;
  int64_t maximize;
  std::vector<ge::DataType> dtypeLst;
  int64_t tilingKey;
  bool isDiffDtype{false};
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_V2_H_