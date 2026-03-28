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
 * \file cube_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_UTIL_CUBE_UTIL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_UTIL_CUBE_UTIL_H_

#include <cstdint>
#include <string>

#include "cube/algorithm/entity/status.h"
#include "cube/include/cube_tiling_param.h"
#include "cube/platform/platform_info.h"
#include "cube/util/math_util.h"
#include "cube/algorithm/entity/shape.h"

namespace optiling {
namespace cachetiling {
constexpr int32_t khnumWNoDivided = 2;
constexpr int32_t kC0 = 16;
constexpr int32_t kLargeWo = 4065;  // loadin_size * aub_db + copyout_size * cub_db <= ubsize / fp16_size --> wo <= 4064
constexpr int32_t kLargeWo_cond2 = 4033;  // when dx_no_overlap_condition_2 is true, n_cub_min = 2, then wo <= 4032
class CubeUtil {
 public:
  static inline int32_t CalcHi(int32_t ho, int32_t stride_h, int32_t kernel_h_dilation, int32_t ori_hi) {
    return MathUtil::Min(static_cast<int64_t>(ho - 1) * stride_h + kernel_h_dilation, ori_hi);
  }
  static inline int32_t GetK1(int32_t c1, int32_t kh, int32_t kw) { return c1 * kh * kw; }
  static inline int64_t GetASize(int32_t batch_dim, int32_t m, int32_t k) {
    return static_cast<int64_t>(batch_dim) * m * k;
  }
  static inline int64_t GetBSize(int32_t n, int32_t k) { return static_cast<int64_t>(n) * k; }
  static inline int64_t GetL1LoadSize(const DimFactor &blockDimsFactor, int64_t a_size, int64_t b_size) {
    return a_size * blockDimsFactor.n + b_size * blockDimsFactor.group * blockDimsFactor.batch * blockDimsFactor.m;
  }
  static inline int32_t CalcWi(int32_t wo, int32_t stride_w, int32_t kernel_w_dilation, int32_t ori_wi) {
    return MathUtil::Min(static_cast<int64_t>(wo - 1) * stride_w + kernel_w_dilation, ori_wi);
  }
  static void CalcHiWi(const CubeTilingParam *params, int32_t k_bl1, int32_t k_l0, int32_t &hi, int32_t &wi);
  static void CalcMinHiWi(const CubeTilingParam *params, int32_t min_k_l0, int32_t &hi, int32_t &wi);
  static int32_t CalcHo(int64_t k, int32_t wo, const std::string &op_type);
  static void CalcMal1(const CubeTilingParam *params, const TilingShape &shape, int64_t extend_shape_k,
                       const L0Status &l0_status, L1Status &l1_status);
  static void CalcNbl1(const CubeTilingParam *params, const TilingShape &shape, const L0Status &l0_status,
                       int64_t total_n_l1, L1Status &l1_status);
  static void CalcKal1(const CubeTilingParam *params, const TilingShape &/* shape */, int64_t extend_shape_k,
                       const L0Status &l0_status, L1Status &l1_status);
  static void CalcKbl1(const CubeTilingParam *params, int64_t extend_shape_k_, const L0Status &l0_status,
                       L1Status &l1_status);
  static void CalcL1RemainStatus(const CubeTilingParam *params, const L0Status &l0_status, L1Status &l1_status);
  static int32_t GetKFactor(const CubeTilingParam *params, int64_t orig_shape_k, int32_t used_core);
  static int64_t LoopNumFromSingleCoreToL0(const TilingShape &tilingShape, const DimFactor &blockDimsFactor,
                                           const PlatformInfo &platform_info);
  static bool CheckL0Overflow(int32_t m0, int32_t n0, int32_t k0, const CubeTilingParam &params);
  static int32_t GetMinKl1(int32_t k_l0, int32_t k_hw);
  static int64_t GetDfactorSdEqKd(const Conv3DBpInputTilingParam *conv3ddx_param_, int32_t l0c_din);
  static int64_t CalcL1Size(const CubeTilingParam *params, const L1Status &l1_status, const L0Status &l0_status);
  static int64_t CalcAL1Size(const CubeTilingParam *params, const L1Status &l1_status, const L0Status &l0_status);
  static int64_t CalcBL1Size(const CubeTilingParam *params, const L1Status &l1_status, const L0Status &l0_status);
  static int32_t CalcBL1N(const CubeTilingParam *params, int32_t bl1_n);
  static int32_t CalcBL1Cog(int32_t k_bl1, ge::DataType data_type);
  static inline ge::DataType GetComputePowerDataType(ge::DataType data_type) {
    return data_type == ge::DT_BF16 ? ge::DT_FLOAT16 : data_type;
  }

  static inline bool IsLargeWo(const Conv2DBpInputTilingParam &params) {
    return (
        (!params.conv1d_flag) && params.split_axis_mode != kSplitWAxisMode && params.stride_h == 1 &&
        params.stride_w == 1 &&
        (params.a_shape.w + params.b_shape.w - 1 >= (params.dx_no_overlap_condition_2 ? kLargeWo_cond2 : kLargeWo)));
  }
  static bool CheckL0Size(const L0Status &l0_status, const GemmTilingParam &params);
  static bool CheckL1Size(const GemmTilingParam &params, int32_t amat, int32_t bmat, int32_t cur_bias_l1_size = 0);
  static int32_t GetBiasL1Size(const GemmTilingParam &params, const L1Status &l1_status);
  static int32_t GetBiasTableSize(const GemmTilingParam &params, const L0Status &l0_status);
  static int64_t GetLoad3dThroughput(bool is_dst_l0a, int32_t row_switch_times, int32_t stride_w);
  static int32_t GetMilanPkgNumByCacheline(int32_t burst_len);
  static int32_t GetObpPkgNumByCacheline(int32_t burst_len);

  template <typename T>
  static int64_t GetDfactor(T kd_factor, const Conv3DBpInputTilingParam *conv3ddx_param_, int32_t l0c_din) {
    int64_t estimate_d = static_cast<int64_t>(MathUtil::CeilDivision(static_cast<int64_t>(kd_factor - 2 + l0c_din),
                                                                     static_cast<int64_t>(conv3ddx_param_->stride_d)) +
                                              1);
    int64_t dout_factor = std::min(estimate_d, conv3ddx_param_->a_shape.d);
    if (conv3ddx_param_->filter_d_dilation == conv3ddx_param_->stride_d) {
      dout_factor = (conv3ddx_param_->platform_info.support_l0c2out())
                        ? GetDfactorSdEqKd(conv3ddx_param_, l0c_din)
                        : std::max(dout_factor - 1, static_cast<int64_t>(1));
    };
    return dout_factor;
  }

  template <typename T>
  static int32_t GetBl1Bound(const T &params, const TilingShape &/* single_core_size */, const L1Status &l1_status,
                             const L0Status &l0_status) {
    int32_t b_l1_size = CalcBL1Cog(l1_status.k_bl1, params.b_dtype) * params.b_shape.h * params.b_shape.w *
                        l1_status.n_bl1 * l0_status.n * kBlockSize * l1_status.db_bl1;
    return b_l1_size;
  }

  template <typename T>
  static int32_t GetAl1Bound(const T &params, const TilingShape &single_core_size, const DimFactor &block_dims,
                             const L1Status &l1_status, const L0Status &l0_status) {
    int32_t a_l1_size;
    if (params.conv1d_flag) {
      int32_t w_al1 = (l1_status.m_al1 * l0_status.m * kBlockSize - 1) * 1 + params.filter_w_dilation;
      a_l1_size = l1_status.k_al1 * w_al1 * params.a_shape.c0 * l1_status.db_al1;
    } else if (params.split_axis_mode == kSplitWAxisMode) {
      int32_t h_num = std::min(l1_status.m_al1 + params.filter_h_dilation - 1L, params.a_shape.h * params.stride_h);
      a_l1_size = h_num * (l0_status.m * kBlockSize + params.filter_w_dilation - 1) * l1_status.k_al1 *
                  params.a_shape.c0 * l1_status.db_al1;
    } else {
      int32_t h_num = (params.filter_h_dilation - 1) + l1_status.m_al1 * l0_status.m * kBlockSize / params.c_shape.w +
                      khnumWNoDivided;
      if (l1_status.m_al1 * l0_status.m * kBlockSize < params.c_shape.w) {
        h_num = (params.filter_h_dilation - 1) + khnumWNoDivided;
      } else if (l1_status.m_al1 * l0_status.m * kBlockSize % params.c_shape.w == 0) {
        h_num = (params.filter_h_dilation - 1) + l1_status.m_al1 * l0_status.m * kBlockSize / params.c_shape.w;
      }
      h_num = std::min(static_cast<int64_t>(h_num), params.a_shape.h * params.stride_h);
      a_l1_size = l1_status.k_al1 * params.a_shape.w * params.stride_w * params.a_shape.c0 * h_num * l1_status.db_al1;
    }

    if (params.split_axis_mode != kSplitWAxisMode && l1_status.k_al1 == params.a_shape.c1 && block_dims.m == 1 &&
        l1_status.m_al1 * l0_status.m == single_core_size.m) {
      int32_t hw_ceil_align =
          MathUtil::Align(params.a_shape.h * params.stride_h * params.a_shape.w * params.stride_w, kBlockSize);
      a_l1_size = l1_status.k_al1 * params.a_shape.c0 * hw_ceil_align;
    }
    return a_l1_size;
  }
  template <typename T>
  static bool CheckLoad3dV2KstartOverflow(int32_t k_al1, const T &params) {
    if (!params.platform_info.support_l0c2out()) {
      return false;
    }
    return k_al1 * params.b_shape.h * params.b_shape.w * params.a_shape.c0 >= kMaxLoad3dV2Kstart;
  }
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_UTIL_CUBE_UTIL_H_