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
 * \file ub_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_UB_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_UB_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
class UbCalculator : public Calculator {
 public:
  UbCalculator(SingleCoreStatus &core_status);
  virtual ~UbCalculator() = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  int32_t GetTilingWiBub() const;

 private:
  void CalcUbStatus();
  void UpdateSinleCoreStatus();
  int32_t CalcWiBub(int32_t limit_hn, int32_t k_bub) const;
  void CalcKBub(int32_t limit_hn, int32_t bl1_hi);

  UbStatus ub_status_;
  int32_t wi_bub_ = 0;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_UB_CALCULATOR_H_