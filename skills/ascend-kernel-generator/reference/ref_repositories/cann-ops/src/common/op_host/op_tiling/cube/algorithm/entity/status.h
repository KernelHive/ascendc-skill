/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! * \file status.h

 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_STATUS_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_STATUS_H_

#include <string>
#include "cube/platform/platform_info.h"
#include "cube/constants/constants_define.h"
#include "cube/algorithm/entity/shape.h"

namespace optiling {
namespace cachetiling {
class DimFactor {
 public:
  DimFactor() : batch(1), m(1), k(1), n(1), d(1), group(1) {}
  DimFactor(int32_t in_batch, int32_t in_m, int32_t in_k, int32_t in_n, int32_t in_group = 1)
      : batch(in_batch), m(in_m), k(in_k), n(in_n), d(1), group(in_group) {}
  std::string ToString() const;
  int32_t ReduceMul() const;
  void Init();
  bool IsValid() const;

  int32_t batch = 1;
  int32_t m = 1;
  int32_t k = 1;
  int32_t n = 1;
  int32_t d = 1;
  int32_t group = 1;
};

class L0Status {
 public:
  std::string ToString() const;
  void Init();

  int32_t batch = 1;
  int32_t m = 1;
  int32_t k = 1;
  int32_t n = 1;
  int32_t din = 1;
  int32_t dk = 1;
  int32_t dout = 1;
  int32_t db_l0a = kDbOn;
  int32_t db_l0b = kDbOn;
  int32_t db_l0c = kDbOff;
};

class L1Status {
 public:
  std::string ToString() const;
  void Init();

  int32_t m_al1 = 1;
  int32_t k_al1 = 1;
  int32_t k_bl1 = 1;
  int32_t n_bl1 = 1;
  int32_t db_al1 = kDbOff;
  int32_t db_bl1 = kDbOff;
  int32_t ho = 0;
  int32_t al1_bound = 0;
  int32_t bl1_bound = 0;
  int32_t d_al1 = 1;
  int32_t d_bl1 = 1;
  int64_t al1_repeat_time = 1;
  int64_t bl1_repeat_time = 1;
};

class UbStatus {
 public:
  std::string ToString() const;
  void Init();

  int32_t m_aub = 1;
  int32_t k_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t n_cub = 1;
  int32_t batch_cub = 1;
  int32_t batch_aub = 1;
  int32_t batch_bub = 1;
  int32_t db_aub = kDbOn;
  int32_t db_bub = kDbOn;
  int32_t db_cub = kDbOn;
};

class ResourceStatistic {
 public:
  void set_core_used(int32_t core_used) { core_used_ = core_used; }
  void set_l1_used(int32_t l1_used) { l1_used_ = l1_used; }
  void set_l0a_used(int32_t l0a_used) { l0a_used_ = l0a_used; }
  void set_l0b_used(int32_t l0b_used) { l0b_used_ = l0b_used; }
  void set_l0c_used(int32_t l0c_used) { l0c_used_ = l0c_used; }
  void set_ub_used(int32_t ub_used) { ub_used_ = ub_used; }
  std::string Show(const PlatformInfo &platform_info) const;

 private:
  int32_t core_used_ = 0;
  int32_t l1_used_ = 0;
  int32_t l0a_used_ = 0;
  int32_t l0b_used_ = 0;
  int32_t l0c_used_ = 0;
  int32_t ub_used_ = 0;
};

struct CoreStatus {
  int32_t m_mapped = 1;
  int32_t k_mapped = 1;
  int32_t n_mapped = 1;
  int32_t batch_mapped = 1;
  int32_t m_single_core = 1;
  int32_t n_single_core = 1;
  int32_t n0_max = 1;
  int64_t cycle = INT64_MAX;
  int32_t load_size = INT32_MAX;
  int64_t repeat_load_size = INT64_MAX;
  int64_t mad_cycle = 1;
  int32_t block_value = 0;
  int32_t l0c_multi_batch = 0;
  int32_t dtype_bias = 0;
  int32_t multi_m_al1 = 1;
  int32_t multi_n_bl1 = 1;
  bool split_k_fp32 = false;
  bool non_factor_k = false;
  int32_t load_2d_times = INT32_MAX;

  // for l1
  int32_t kal1_factor = 1;
  int32_t kbl1_factor = 1;
  bool both_full_load = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  bool al1_k_full_load = false;
  bool bl1_k_full_load = false;

  // for ub
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int32_t cub_dtype_multi = 1;
  float fused_double_operand_num = 0.0;
  bool flag_cub_solving_bank_conflict = false;
};

class HardwareStatus {
 public:
  std::string ToString() const;
  void Init(bool support_l0c2out, bool is_fp32_in);
  int32_t full_cache_line = 0;
  int32_t mte1_l0a_bandwidth = 0;
  int32_t mte1_l0b_bandwidth = 0;
  int32_t l0c_factor_limit = 0;
  int32_t perf_min_core_num = 0;
  int32_t input_dtype_bytes = 0;
  int32_t output_dtype_bytes = 0;
  int32_t reduce_block_size = 0;
  int32_t full_cache_line_unalign = 0;
  int32_t l0_factor_limit = 0;
  int32_t nl0_prefer_size = 0;
  int32_t kl0_prefer_size = 0;
  int32_t bl1_prefer_size = 0;
  int32_t ml0_prefer_size_outer_axis = 0;
};

class SingleCoreStatus {
 public:
  const TilingShape &orig_shape() const { return orig_shape_; }
  const TilingShape &shape() const { return shape_; }
  const DimFactor &block_dims() const { return block_dims_; }
  const L0Status &l0_status() const { return l0_status_; }
  const L1Status &l1_status() const { return l1_status_; }
  const UbStatus &ub_status() const { return ub_status_; }
  const CoreStatus &core_status() const { return core_status_; }

  void UpdateOrigShape(const TilingShape &shape) { orig_shape_ = shape; }
  void UpdateShape(const TilingShape &shape) { shape_ = shape; }
  void UpdateBlockDims(const DimFactor &factor) { block_dims_ = factor; }
  void UpdateL0Status(const L0Status &l0_status) { l0_status_ = l0_status; }
  void UpdateL1Status(const L1Status &l1_status) { l1_status_ = l1_status; }
  void UpdateUbStatus(const UbStatus &ub_status) { ub_status_ = ub_status; }
  void UpdateCoreStatus(const CoreStatus &core_status) { core_status_ = core_status; }

  std::string ToString() const;
  void Init();

 private:
  TilingShape orig_shape_;
  TilingShape shape_;
  DimFactor block_dims_;
  L0Status l0_status_;
  L1Status l1_status_;
  UbStatus ub_status_;
  CoreStatus core_status_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_STATUS_H_