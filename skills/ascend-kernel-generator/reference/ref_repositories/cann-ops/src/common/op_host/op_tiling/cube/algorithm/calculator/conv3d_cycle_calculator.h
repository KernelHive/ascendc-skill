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
 * \file conv3d_cycle_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_CYCLE_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_CYCLE_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
constexpr int32_t kMmadResBytes = kFp32Bytes;
class Conv3DTilingIdParam {
 public:
  Conv3DTilingIdParam()
      : binary_mode_(0),
        load_mode_(0),
        load3d_special_(0),
        cycle_buffer_flag_(0),
        bl0_attach_flag_(0),
        bl1_attach_flag_(0),
        al1_attach_flag_(0),
        abkl1_attach_flag_(0),
        db_cub_(0),
        db_l0c_(0),
        db_bl1_(0),
        db_al1_(0),
        pad_greater_than_filter_(0),
        reserved_(0) {}

  Conv3DTilingIdParam(int32_t al1_attach_flag, int32_t bl1_attach_flag, int32_t abkl1_attach_flag,
                      int32_t bl0_attach_flag)
      : binary_mode_(0),
        load_mode_(0),
        load3d_special_(0),
        cycle_buffer_flag_(0),
        bl0_attach_flag_(static_cast<uint32_t>(bl0_attach_flag)),
        bl1_attach_flag_(static_cast<uint32_t>(bl1_attach_flag)),
        al1_attach_flag_(static_cast<uint32_t>(al1_attach_flag)),
        abkl1_attach_flag_(static_cast<uint32_t>(abkl1_attach_flag)),
        db_cub_(0),
        db_l0c_(0),
        db_bl1_(0),
        db_al1_(0),
        pad_greater_than_filter_(0),
        reserved_(0) {}

  int32_t al1_attach_flag() const { return static_cast<int32_t>(al1_attach_flag_); }
  int32_t bl1_attach_flag() const { return static_cast<int32_t>(bl1_attach_flag_); }
  int32_t tiling_id() const { return *(reinterpret_cast<const int32_t *>(this)); }
  void Calc(const CubeTilingParam *params, const TilingShape &shape, const L0Status &l0_status,
            const L1Status &l1_status, const UbStatus &ub_status);
  void SetDbFlag(const L0Status &l0_status, const L1Status &l1_status, const UbStatus &ub_status);
  void SetPadGreaterThanFilter(bool pad_greater_than_filter);
  void RetainAttachFlags();

 private:
  void CheckSpecialTemplate(const CubeTilingParam *params, const TilingShape &shape, const L0Status &l0_status,
                            const L1Status &l1_status);

  uint32_t binary_mode_ : 2;
  uint32_t load_mode_ : 2;
  uint32_t load3d_special_ : 1;
  uint32_t cycle_buffer_flag_ : 1;
  uint32_t bl0_attach_flag_ : 1;
  uint32_t bl1_attach_flag_ : 2;
  uint32_t al1_attach_flag_ : 2;
  uint32_t abkl1_attach_flag_ : 2;
  uint32_t db_cub_ : 1;
  uint32_t db_l0c_ : 1;
  uint32_t db_bl1_ : 1;
  uint32_t db_al1_ : 1;
  uint32_t pad_greater_than_filter_ : 1;
  uint32_t reserved_ : 14;  // to fullfil 32 bit, if add new template bit then decrease this number
};

struct Conv3DCycleStatus {
  int64_t mte1_a = 0;
  int64_t mte1_b = 0;
  int64_t mte1_bias = 0;
  int64_t mte2_a = 0;
  int64_t mte2_b = 0;
  int64_t mte2_bias = 0;
  int64_t mte3_ = 0;
  int64_t fixp = 0;
  int64_t mad = 0;
};

class Conv3DCycleCalculator {
 public:
  Conv3DCycleCalculator(const TilingShape &shape, const DimFactor &block_dims, const L0Status &l0_status,
                        const L1Status &l1_status)
      : shape_(shape), block_dims_(block_dims), l0_status_(l0_status), l1_status_(l1_status) {}
  virtual ~Conv3DCycleCalculator() = default;

  virtual void Init(const Conv3DTilingParam *params);

  bool GetCycleByModel(const Conv3DTilingIdParam &tiling_id_param, int64_t &cycle);

  int64_t GetMte1ACycle() const;
  int64_t GetMte1BCycle() const;
  int64_t GetMte2ACycle() const;
  int64_t GetMte2BCycle() const;
  virtual int64_t GetMte2BiasCycle() const = 0;
  int64_t GetMadCycle() const;
  virtual int64_t GetMte1BiasCycle() const { return 0; }
  virtual int64_t GetFixpCycle() const { return 0; }
  virtual int64_t GetMte3Cycle() const { return 0; }

  int64_t CalCycleBothFullLoad() const;
  int64_t CalCycleAL1FullLoadBL1KFullLoad() const;
  int64_t CalCycleAL1KFullLoadBL1FullLoad() const;
  int64_t CalCycleBothKFullLoad() const;
  int64_t CalCycleBothNotFullLoadSameK() const;
  int64_t CalCycleOnlyAL1FullLoadKal1GtKbl1() const;
  int64_t CalCycleOnlyAL1KFullLoadKal1GtKbl1() const;
  int64_t CalCycleBothNotFullLoadKal1GtKbl1() const;
  int64_t CalCycleAL1FullLoadBL0FullLoad() const;
  int64_t CalCycleAL1FullLoadBL1NotLoadBL0NotFullLoad() const;
  int64_t CalCycleAL1KFullLoadBL0FullLoad() const;
  int64_t CalCycleAL1KFullLoadBL1NotLoadBL0NotFullLoad() const;
  int64_t CalCycleAL1NotFullLoadBL1FullLoad() const;
  int64_t CalCycleAL1NotFullLoadBL1KFullLoad() const;
  int64_t CalCycleBothNotFullLoadKal1LtKbl1() const;
  int64_t CalCycleAL1NotFullLoadBL0FullLoad() const;
  int64_t CalCycleAL1NotFullLoadBL1NotLoadBL0NotFullLoad() const;

 protected:
  virtual int32_t GetPkgNumByCacheline(int32_t burst_len) const = 0;
  int64_t GetDbCycle(int32_t db_switch, int64_t cycle1, int64_t cycle2) const {
    if (db_switch == kDbOn) {
      return std::max(cycle1, cycle2);
    } else {
      return cycle1 + cycle2;
    }
  }

  int32_t hbm_band_width_ = 1;
  int32_t l1_read_band_width_ = 1;
  int32_t mte1_l0a_bandWidth_ = 1;
  int32_t mte1_l0b_bandWidth_ = 1;
  int32_t mte1_load2d_cost_ = 0;
  int32_t mte1_load3d_cost_ = 0;
  int32_t l0c_bandWidth_ = 1;
  int32_t mad_cost_ = 0;
  int32_t mte2_latency_ = 0;

  const TilingShape &shape_;
  const DimFactor &block_dims_;
  const L0Status &l0_status_;
  const L1Status &l1_status_;
  const Conv3DTilingParam *params_ = nullptr;

  int64_t batch_dout_single_core_ = 0;
  int64_t n_single_core_ = 0;
  int64_t m_single_core_ = 0;
  int64_t k_al1_factor_ = 0;
  int64_t k_bl1_factor_ = 0;
  int32_t k_al0_factor_ = 0;
  int32_t k_bl0_factor_ = 0;
  int64_t kl1_times_ = 0;

  Conv3DCycleStatus cycle_;
  using TemplateCycleCalcFunc = int64_t (Conv3DCycleCalculator::*)() const;
  std::unordered_map<int32_t, TemplateCycleCalcFunc> template_calc_funcs_;
};

class Conv3DMilanCycleCalculator : public Conv3DCycleCalculator {
 public:
  Conv3DMilanCycleCalculator(const TilingShape &shape, const DimFactor &block_dims, const L0Status &l0_status,
                             const L1Status &l1_status);
  ~Conv3DMilanCycleCalculator() override = default;

 protected:
  int64_t GetMte2BiasCycle() const override;
  int64_t GetMte1BiasCycle() const override;
  int64_t GetFixpCycle() const override;
  int32_t GetPkgNumByCacheline(int32_t burst_len) const override;
};

class Conv3DObpCycleCalculator : public Conv3DCycleCalculator {
 public:
  Conv3DObpCycleCalculator(const TilingShape &shape, const DimFactor &block_dims, const L0Status &l0_status,
                           const L1Status &l1_status, const UbStatus &ub_status);
  ~Conv3DObpCycleCalculator() override = default;

 protected:
  int64_t GetMte2BiasCycle() const;
  int64_t GetVectorCycle() const;
  int64_t GetMte3Cycle() const override;
  int32_t GetPkgNumByCacheline(int32_t burst_len) const override;

  const UbStatus &ub_status_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_CALCULATOR_CYCLE_CALCULATOR_H_