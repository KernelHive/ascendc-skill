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
 * \file calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CALCULATOR_H_

#include "cube/algorithm/entity/status.h"
#include "cube/include/cube_tiling_param.h"
#include "cube/util/cube_util.h"
#include "cube/util/math_util.h"
#include "cube/util/timer.h"
#include "op_log.h"

namespace optiling {
namespace cachetiling {
static constexpr int32_t kAttachFlagZero = 0;
static constexpr int32_t kAttachFlagOne = 1;
static constexpr int32_t kAttachFlagTwo = 2;
static constexpr int32_t kAttachFlagThree = 3;

static constexpr int32_t kFullLoadFlag = 0;
static constexpr int32_t kKFullLoadFlag = 1;
static constexpr int32_t kKPartLoadFlag = 2;
static constexpr int32_t kKFullLoadMNPartLoadFlag = 3;

static constexpr int32_t kKal1EqualKbl1Flag = 0;
static constexpr int32_t kKal1LargerThanKbl1Flag = 1;
static constexpr int32_t kKal1SmallerThanKbl1Flag = 2;

static constexpr int32_t kAttachConditionNum = 11;

// 512 is the batch threshold, which prevents a large number of binding core
static constexpr int32_t KBatchBindCoreLimit = 512;

// if dma mode and 910A, fmap_ub_for_dma tensor need 32 size
static const int32_t KDmaFmapUbSize = 32;

enum KLoadType {
  kFullLoad,
  kAl1FullLoad,
  kBl1FullLoad,
  kAl1KFullLoad,
  kBl1KFullLoad,
  kNeitherFullLoad
};

class Calculator {
 public:
  Calculator(SingleCoreStatus &core_status) : single_core_status_(core_status) {}
  virtual ~Calculator() = default;
  virtual bool Exec() = 0;
  bool IsL1SizeValid(int64_t l1_size) const;
  virtual bool Init(const CubeTilingParam &params);
  void Clear();
  inline bool IsLargeWi(int32_t wi) const {
    return wi >= 2000 && wi <= 4096;  // 2000: kWiLargeLower, 4096: kWiLargeUpper
  }
  inline int64_t GetExtendShapeK(int64_t ori_k, int32_t shape_extend_time) const {
    return ori_k * shape_extend_time;
  }

 protected:
  const CubeTilingParam *params_ = nullptr;
  SingleCoreStatus &single_core_status_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CALCULATOR_H_