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
 * \file conv2d_bp_input_l0_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_L0_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_L0_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
class Conv2DBpInputL0Calculator : public Calculator {
 public:
  Conv2DBpInputL0Calculator(SingleCoreStatus &core_status);
  ~Conv2DBpInputL0Calculator() override = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  void Clear();
  bool IsLoopMnFromScToL0Is1() const { return loop_mn_from_sc_to_l0_is_1_; }

 private:
  void UpdateSinleCoreStatus();
  void CalcLoadSize();
  bool CalcL0Status();
  bool CalcL0LoadSizeAndUpdateFactor(int32_t m_l0_min);
  bool GetL0FactorsGeneral();
  bool CheckUbDb(int32_t m0) const;
  bool GetM0(int32_t &n0, int32_t &k0, int32_t &m0);
  bool CheckL1Overflow(int32_t m0, int32_t n0) const;
  bool NeedUpdate(int64_t load_size, int32_t mkn, int32_t mk);
  void GenL0Factors();
  int64_t loop_sc_to_nl0() const;
  int64_t loop_sc_to_ml0() const;
  int32_t GetL0MinFactorAndCalcLoadSize();

  const TilingShape &shape_;
  const Conv2DBpInputTilingParam *dx_param_ = nullptr;
  const Shape *dedy_ = nullptr;
  const Shape *kernel_ = nullptr;
  const Shape *dedx_ = nullptr;
  L0Status l0_status_;
  L1Status l1_status_;
  UbStatus ub_status_;
  size_t idx_n_ = 0;
  size_t idx_k_ = 0;
  int32_t max_mkn_ = 1;
  int32_t max_mk_ = 1;
  int64_t min_load_size_ = 1;
  int32_t n0_factors_[64];
  int32_t k0_factors_[64];
  bool loop_mn_from_sc_to_l0_is_1_ = false;
  bool bias_table_flag_ = false;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_L0_CALCULATOR_H_