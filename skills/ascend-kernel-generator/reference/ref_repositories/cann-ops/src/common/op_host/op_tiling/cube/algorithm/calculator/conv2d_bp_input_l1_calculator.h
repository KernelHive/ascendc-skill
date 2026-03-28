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
 * \file conv2d_bp_input_l1_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_L1_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_L1_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
struct FactorArray {
  int32_t min_kl1_dim = 1;
  int32_t kn_factors[3L];
  int32_t size_para[3L];
};

class Conv2DBpInputL1Calculator : public Calculator {
 public:
  Conv2DBpInputL1Calculator(SingleCoreStatus &core_status, int32_t &h_num);
  ~Conv2DBpInputL1Calculator() override = default;
  bool Exec() override;
  int32_t GetBiasTableBound() const { return bias_table_bound_; }
  bool Init(const CubeTilingParam &params) override;
  void Clear();

 private:
  int32_t GetAl1MExtent(int32_t al1_m) const;
  int32_t GetBiasL1Size(const L1Status &l1_status, const L0Status &l0_status) const;
  bool GetWnum(const int32_t m_params[2L], int32_t wo_max, int32_t l1_para[3L]) const;
  bool GetHnumSplitWCase(int32_t k_al1, int32_t h_num, int32_t m_size, int32_t m0_size, int32_t l1_para[3L]) const;
  bool GetHnumSplitHWCase(int32_t k_al1, int32_t h_num, int32_t m0_size, int32_t l1_para[3L]) const;
  bool GetHnum(const int32_t m_params[2L], int32_t h2, int32_t l1_para[3L]) const;
  bool GetMnum(const int32_t kn_factors[3L], int32_t m, int32_t l1_para[3L]) const;
  bool GetInitialL1(int32_t k_hw, int32_t &min_kl1_dim);
  void GetMinLoadSize(const FactorArray &factor_size, int32_t co1, int32_t load_m, int64_t &load_size, L1Status &l1);
  void CalConv1DAL1Size(int64_t &min_load_size, int32_t *size_para);
  bool CalConv2DAL1Size(int64_t &min_load_size, int32_t *size_para);
  bool CalcL1Status();
  bool GetL1FactorsOpti(const FactorArray &factor_size, int64_t &min_load_size, bool &first_flag);
  bool CheckL1Size(const L1Status &l1_status) const;
  void UpdateSinleCoreStatus();
  void GenL1Factors();
  int32_t CalAL1H();
  int64_t CalAL1Size(int32_t &h_num);
  int32_t GetBiasTableSize(const L0Status &l0_status) const;

  const Conv2DBpInputTilingParam *dx_param_ = nullptr;
  const Shape *dedy_ = nullptr;
  const Shape *kernel_ = nullptr;
  const Shape *dedx_ = nullptr;
  L1Status l1_status_;
  L0Status l0_status_;
  const DimFactor &block_dims_;
  const TilingShape &shape_;
  vector<int32_t> k_factors_;
  vector<int32_t> nl1_factors_;
  int32_t init_db_al1_ = 1;
  int32_t init_db_bl1_ = 1;
  int32_t &h_num_;
  int32_t kernel_hw_ = 0;
  int32_t bias_table_bound_ = 0;
  int32_t bias_table_flag_ = 0;  // 该变量将参与算术运算，所以声明为 int32_t
  bool update_l1_ = false;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_L1_CALCULATOR_H_