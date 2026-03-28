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
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV3D_BP_INPUT_CYCLE_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV3D_BP_INPUT_CYCLE_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
constexpr int32_t kMmadResBytes = kFp32Bytes;
class Conv3DDxTilingIdParam {
 public:
  Conv3DDxTilingIdParam()
      : db_al1_(0),
        db_bl1_(0),
        db_l0c_(0),
        db_cub_(0),
        al1_attach_flag_(0),
        bl1_attach_flag_(0),
        abkl1_attach_flag_(0),
        dilation_d_gt_one_flag_(0),
        stride_expand_flag_(0),
        sd_kd_mode_(0),
        split_axis_mode_(0),
        load3d_special_(0),
        fusion_mode_(0),
        reserved_(0) {}

  Conv3DDxTilingIdParam(int32_t abkl1_attach_flag, int32_t al1_attach_flag, int32_t bl1_attach_flag, int32_t sd_kd_mode)
      : db_al1_(0),
        db_bl1_(0),
        db_l0c_(0),
        db_cub_(0),
        al1_attach_flag_(static_cast<uint32_t>(al1_attach_flag)),
        bl1_attach_flag_(static_cast<uint32_t>(bl1_attach_flag)),
        abkl1_attach_flag_(static_cast<uint32_t>(abkl1_attach_flag)),
        dilation_d_gt_one_flag_(0),
        stride_expand_flag_(0),
        sd_kd_mode_(static_cast<uint32_t>(sd_kd_mode)),
        split_axis_mode_(0),
        load3d_special_(0),
        fusion_mode_(0),
        reserved_(0) {}

  int32_t al1_attach_flag() const { return static_cast<int32_t>(al1_attach_flag_); }
  int32_t bl1_attach_flag() const { return static_cast<int32_t>(bl1_attach_flag_); }
  int32_t sd_kd_mode() const { return static_cast<int32_t>(sd_kd_mode_); }

  int32_t tiling_id() const { return *(reinterpret_cast<const int32_t *>(this)); }

  void GetAttachFlag(const Conv3DBpInputTilingParam *conv3ddx_params, const TilingShape &singlecore_shape,
                     const L0Status &l0_status, const L1Status &l1_status);
  void GetDbFlag(const L1Status &l1_status);
  void GetAdditionalFlag(const Conv3DBpInputTilingParam *conv3ddx_params);
  void RetainAttachFlags();
  std::string ToString() const {
    std::stringstream ss;
    ss << " db_al1: "<< db_al1_
    << " db_bl1: "<< db_bl1_
    << " db_l0c: "<< db_l0c_
    << " db_cub: "<< db_cub_
    << " al1_attach_flag: "<< al1_attach_flag_
    << " bl1_attach_flag: "<< bl1_attach_flag_
    << " abkl1_attach_flag: "<< abkl1_attach_flag_
    << " dilation_d_gt_one_flag_"<< dilation_d_gt_one_flag_
    << " stride_expand_flag: "<< stride_expand_flag_
    << " sd_kd_mode: "<< sd_kd_mode_
    << " split_axis_mode: "<< split_axis_mode_
    << " load3d_special: "<< load3d_special_
    << " fusion_mode: "<< fusion_mode_;
    return ss.str();
  }

 private:
  uint32_t db_al1_ : 1;
  uint32_t db_bl1_ : 1;
  uint32_t db_l0c_ : 1;
  uint32_t db_cub_ : 1;
  uint32_t al1_attach_flag_ : 2;
  uint32_t bl1_attach_flag_ : 2;
  uint32_t abkl1_attach_flag_ : 2;
  uint32_t dilation_d_gt_one_flag_ : 1;
  uint32_t stride_expand_flag_ : 1;
  uint32_t sd_kd_mode_ : 2;
  uint32_t split_axis_mode_ : 1;
  uint32_t load3d_special_ : 1;
  uint32_t fusion_mode_ : 2;
  uint32_t reserved_ : 14;  // to fullfil 32 bit, if add new template bit then decrease this number
};

struct Conv3DDxCycleStatus {
  int64_t mte1_a = 0;
  int64_t mte1_b = 0;
  int64_t mte2_a = 0;
  int64_t mte2_b = 0;
  int64_t fixp = 0;
  int64_t mad = 0;
  int64_t mte1vsmad = 0;
};

class Conv3DDxCycleCalculator {
 public:
  Conv3DDxCycleCalculator(const TilingShape &shape, const DimFactor &block_dims, const L0Status &l0_status,
                          const L1Status &l1_status)
      : singlecore_shape_(shape), block_dims_(block_dims), l0_status_(l0_status), l1_status_(l1_status) {}
  virtual ~Conv3DDxCycleCalculator() = default;

  virtual void Init(const Conv3DBpInputTilingParam *params);

  bool GetCycleByModel(const Conv3DDxTilingIdParam &conv3ddx_tiling_id_params, int64_t &cycle);
  int64_t GetMte2ACycle(const Conv3DDxTilingIdParam &conv3ddx_tiling_id_params) const;
  int64_t GetMte2BCycle(const Conv3DDxTilingIdParam &conv3ddx_tiling_id_params) const;
  int64_t GetMte1ACycle() const;
  int64_t GetMte1BCycle() const;
  int64_t GetMadCycle() const;
  virtual int64_t GetFixpCycle() const { return 0; }

  int64_t CalCycleBothFullLoad() const;
  int64_t CalCycleAL1FullLoadBL1KFullLoad() const;
  int64_t CalCycleAL1KFullLoadBL1FullLoad() const;
  int64_t CalCycleAL1KFullLoadBL1KFullLoad() const;
  int64_t CalCycleAL1NotFullLoadBL1NotFullLoadSameK() const;
  int64_t CalCycleAL1FullLoadBL1NotFullLoad() const;
  int64_t CalCycleAL1KFullLoadBL1NotFullLoad() const;
  int64_t CalCycleAL1NotFullLoadBL1NotFullLoadKal1GtKbl1() const;
  int64_t CalCycleAL1NotFullLoadBL1FullLoad() const;
  int64_t CalCycleAL1NotFullLoadBL1KFullLoad() const;
  int64_t CalCycleAL1NotFullLoadBL1NotFullLoadKal1LtKbl1() const;

 protected:
  virtual int32_t GetPkgNumByCacheline(int32_t burst_len) const = 0;

  int64_t GetDbCycle(int32_t db_switch, int64_t cycle1, int64_t cycle2) const {
    if (db_switch == kDbOn) {
      return std::max(cycle1, cycle2);
    } else {
      return cycle1 + cycle2;
    }
  }

  int32_t sd_kd_mode = 0;
  int32_t sd_kd_mode_typenumber = 2;

  int32_t hbm_band_width_ = 1;
  int32_t l1_read_band_width_ = 1;
  int32_t mte1_l0a_bandWidth_ = 1;
  int32_t mte1_l0b_bandWidth_ = 1;
  int32_t mte1_load2d_cost_ = 0;
  int32_t mte1_load3d_cost_ = 0;
  int32_t l0c_bandWidth_ = 1;
  int32_t mad_cost_ = 0;
  int32_t mte2_latency_ = 0;

  int64_t batch_single_core = 0;
  int64_t d_single_core = 0;
  int64_t n1_single_core_ = 0;
  int64_t m1_single_core_ = 0;
  int32_t max_kl1_div_kl0 = 0;
  int32_t min_kl1_div_kl0 = 0;
  int32_t k_div_max_kl1 = 0;
  int32_t max_kl1_div_min_kl1 = 0;
  int32_t kdout_axis_ = 1;
  int32_t kd1_axis_ = 1;
  int32_t kd0_axis_ = 1;

  const TilingShape &singlecore_shape_;
  const DimFactor &block_dims_;
  const L0Status &l0_status_;
  const L1Status &l1_status_;
  const Conv3DBpInputTilingParam *conv3ddx_params_ = nullptr;

  Conv3DDxCycleStatus cycle_;
  using TemplateCycleCalcFunc = int64_t (Conv3DDxCycleCalculator::*)() const;
  std::unordered_map<int32_t, TemplateCycleCalcFunc> template_calc_map_;
};

class Conv3DDxMilanCycleCalculator : public Conv3DDxCycleCalculator {
 public:
  Conv3DDxMilanCycleCalculator(const TilingShape &shape, const DimFactor &block_dims, const L0Status &l0_status,
                               const L1Status &l1_status);
  ~Conv3DDxMilanCycleCalculator() override = default;

 protected:
  int64_t GetFixpCycle() const override;
  int32_t GetPkgNumByCacheline(int32_t burst_len) const override;
};

class Conv3DDxObpCycleCalculator : public Conv3DDxCycleCalculator {
 public:
  Conv3DDxObpCycleCalculator(const TilingShape &shape, const DimFactor &block_dims, const L0Status &l0_status,
                             const L1Status &l1_status, const UbStatus &ub_status);
  ~Conv3DDxObpCycleCalculator() override = default;
  const UbStatus &ub_status_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV3D_BP_INPUT_CYCLE_CALCULATOR_H_