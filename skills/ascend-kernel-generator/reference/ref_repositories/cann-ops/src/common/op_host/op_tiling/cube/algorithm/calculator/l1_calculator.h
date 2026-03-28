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
 * \file l1_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_L1_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_L1_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"
#include "cube/algorithm/entity/shape.h"

namespace optiling {
namespace cachetiling {
namespace {
constexpr size_t kL1FactorMaxSize = 4;
}
class L1Factor {
 public:
  L1Factor() {}
  L1Factor(KLoadType in_load_type, int64_t in_m, int64_t in_k_a, int64_t in_k_b, int64_t in_n)
      : load_type(in_load_type), m(in_m), k_a(in_k_a), k_b(in_k_b), n(in_n) {}
  bool IsValid() const { return m <= INT32_MAX && k_a <= INT32_MAX && k_b <= INT32_MAX && n <= INT32_MAX;}
  KLoadType load_type = kFullLoad;
  int64_t m = 1;
  int64_t k_a = 1;
  int64_t k_b = 1;
  int64_t n = 1;
};

class L1Calculator : public Calculator {
 public:
  L1Calculator(SingleCoreStatus &core_status);
  virtual ~L1Calculator() = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;

 private:
  size_t GenL1Factors(std::array<L1Factor, kL1FactorMaxSize> &l1_factors) const;
  bool CalcL1Status(const L1Factor &factor);
  int64_t CalcLoadSize(KLoadType load_type);
  int64_t FullLoadSize() const;
  int64_t Al1FullLoadSize();
  int64_t Bl1FullLoadSize();
  int64_t NeitherFullLoadSize();
  bool NeitherFullLoadHelper();
  void UpdateSinleCoreStatus();
  bool NeedUpdate(int64_t load_size, int64_t scope) const;
  void CalcL1RemainStatus();
  void Update(int64_t load_size, int64_t scope);

  const TilingShape &shape_;
  const L0Status &l0_status_;
  L1Status l1_status_;

  int64_t extend_shape_k_ = 0;
  int64_t load_size_ = INT64_MAX;
  int64_t scope_ = 0;
  bool load2d_mode_ = false;
  L1Status best_l1_status_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_L1_CALCULATOR_H_