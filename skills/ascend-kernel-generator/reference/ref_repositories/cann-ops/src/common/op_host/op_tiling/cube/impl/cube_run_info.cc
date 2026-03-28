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
 * \file cube_run_info.cc
 * \brief
 */

#include "cube/include/cube_run_info.h"

#include "cube/util/math_util.h"

namespace optiling {
using namespace cachetiling;
void Conv2dBpFilterRunInfo::Update(const Conv2DBpFilterTilingParam& params,
                                   const Conv2DBpFilterTiling& tiling) {
  SetShape(params);
  SetTiling(params, tiling);
  SetSpecialScene(params);
}

void Conv2dBpFilterRunInfo::SetShape(const Conv2DBpFilterTilingParam &params) {
  // transdata vars
  batch = params.a_shape.batch;
  ci = params.b_shape.c;
  hi = params.strideh_read_flag == 1 ? params.sr_fmap_h : params.b_shape.h;
  wi = params.b_shape.w;
  co = params.a_shape.c;
  ho = params.a_shape.h;
  wo = params.a_shape.w;
  kh = params.kernel_h;
  kw = params.kernel_w;
  // dw vars
  // ci1/co1 assigned with cin1_g/cout1_g, calculate again
  // for fp32 scene, k0 is 8 and c0 is 16 here, calculate c1 with 8
  ci1 = MathUtil::CeilDivision(params.b_shape.c, static_cast<int64_t>(params.k0));
  co1 = MathUtil::CeilDivision(params.a_shape.c, static_cast<int64_t>(params.k0));
  stride_h = params.strideh_read_flag == 1 ? params.sr_stride_h : params.stride_h;
  stride_w = params.load3d_special == 1 ? params.stride_w : (params.b_shape.w + params.pad_r + params.pad_l);
  pad_u = params.pad_u;
  pad_d = params.pad_d;
  pad_l = params.pad_l;
  pad_r = params.pad_r;
  dilation_h = params.dilation_h;
  dilation_w = params.dilation_w;
  groups = params.groups;
  cin1_g = params.b_shape.c1;
  cout1_g = params.a_shape.c1;
  real_g = params.real_g;
  mag_factor = params.mag_factor;
  hf32_flag = params.hf32_flag;
  if (params.b_dtype == ge::DT_FLOAT) {
    // calculate c1 with c0(16) in conv2d_backprop_filter.cc, need to calculate again with 8
    cin1_g = MathUtil::CeilDivision(mag_factor * params.b_shape.c / params.groups, static_cast<int64_t>(params.k0));
    cout1_g = MathUtil::Align(mag_factor * params.a_shape.c / params.groups, params.a_shape.c0) / params.k0;
  }
}

void Conv2dBpFilterRunInfo::SetTiling(const Conv2DBpFilterTilingParam &params, const Conv2DBpFilterTiling &tiling) {
  // set tiling var
  // for fp32 scene, c0 is 16 here and k0 is 8, align to 16 firstly
  int64_t single_core_k = MathUtil::CeilDivision(
      MathUtil::Align(params.a_shape.h * params.a_shape.w, params.b_shape.c0) / params.k0, tiling.k_dim);
  group_dim = tiling.group_dim;
  batch_dim = tiling.batch_dim;
  h_dim = tiling.k_dim;
  batch_single_core = MathUtil::CeilDivision(params.a_shape.batch, static_cast<int64_t>(tiling.batch_dim));
  int64_t total_n = static_cast<int64_t>(cin1_g) * params.kernel_h * params.kernel_w;
  n_single_core = MathUtil::CeilDivision(total_n, tiling.n_dim * tiling.n_bl1 * tiling.n_l0);
  // for fp32 scene, n_single_core is a multiple of 2 for (2 * output_n0(8) -> L0C n0(16))
  if (params.b_dtype == ge::DT_FLOAT) {
    n_single_core = MathUtil::Align(n_single_core, 2);  // 2: align to even number
  }
  n_dim = tiling.n_dim;
  n_bl1 = tiling.n_bl1;
  n_ub_l0_time = tiling.n_l0 / tiling.n_cub;
  cub_n1 = tiling.n_cub;
  m_dim = tiling.m_dim;
  m_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(params.a_shape.c1, static_cast<int64_t>(tiling.m_dim)),
                                         static_cast<int64_t>(tiling.m_al1) * tiling.m_l0);
  m_al1 = tiling.m_al1;
  m_l0 = tiling.m_l0;
  k_l0 = tiling.k_l0;
  k_al1_factor = single_core_k * params.load3d_special / tiling.k_al1;
  k_bl1_factor = single_core_k / tiling.k_bl1;
  k_al0_factor = tiling.k_al1 / tiling.k_l0;
  k_bl0_factor = tiling.k_bl1 * params.load3d_special / tiling.k_l0;
  k_al1_16 = tiling.k_al1;
  k_bl1_16 = tiling.k_bl1;
  kl1_times = std::max(tiling.k_al1, tiling.k_bl1) / std::min(tiling.k_al1, tiling.k_bl1);
  if (params.load3d_special != 1 && tiling.k_al1 < tiling.k_bl1) {
    kl1_times = kl1_times * params.load3d_special;
  } else if (params.load3d_special != 1 && tiling.k_al1 > tiling.k_bl1) {
    kl1_times = kl1_times / params.load3d_special;
  }

  if (params.split_w_flag) {
    k_al1_factor = MathUtil::CeilDivision(params.a_shape.h, static_cast<int64_t>(tiling.k_dim)) / (k_al1_16 / k_l0);
    k_bl1_factor = MathUtil::CeilDivision(params.a_shape.h, static_cast<int64_t>(tiling.k_dim)) / (k_bl1_16 / k_l0);
  }

  bl1_bound = tiling.bl1_bound;
  m_aub = tiling.m_aub;
  n_bub = tiling.n_bub;
  k_aub = tiling.k_aub;
  k_bub = tiling.k_bub;
  ho_bl1 = tiling.ho_bl1;
  multi_n_ub_l1 = tiling.n_bl1 * tiling.n_l0 / tiling.n_bub;
  multi_m_ub_l1 = tiling.m_al1 * tiling.m_l0 / tiling.m_aub;
  multi_k_aub_l1 = tiling.k_al1 / tiling.k_aub;
  multi_k_bub_l1 = tiling.k_bl1 / tiling.k_bub;
  db_al1 = tiling.db_al1 - 1;
  db_bl1 = tiling.db_bl1 - 1;
  db_l0c = tiling.db_l0c - 1;
  // set special var
  wi_bub = tiling.wi_bub;
}

void Conv2dBpFilterRunInfo::SetSpecialScene(const Conv2DBpFilterTilingParam &params) {
  load3d_special = params.load3d_special;
  is_bf16 = static_cast<int32_t>(params.a_dtype == ge::DT_BF16);
}

void Conv3DRunInfo::Update(const cachetiling::Conv3DTilingParam &params, const cachetiling::Conv3DTiling &tiling) {
  SetShape(params);
  SetTiling(params, tiling);
  SetSpecialScene(params);
}

void Conv3DRunInfo::SetShape(const Conv3DTilingParam &params) {
  // transdata vars
  batch_n = params.a_shape.batch;
  fmap_c = params.a_shape.c;
  fmap_d = params.a_shape.d;
  fmap_h = params.a_shape.h;
  fmap_w = params.a_shape.w;
  kernel_d = params.kernel_d;
  kernel_h = params.kernel_h;
  kernel_w = params.kernel_w;
  c_out = params.c_shape.c;
  d_out = params.c_shape.d;
  h_out = params.c_shape.h;
  w_out = params.c_shape.w;
  fmap_c1 = MathUtil::CeilDivision(params.a_shape.c, static_cast<int64_t>(params.k0));
  c1_out = MathUtil::CeilDivision(params.c_shape.c, static_cast<int64_t>(params.k0));

  stride_d = params.stride_d;
  stride_h = params.stride_h;
  stride_w = params.stride_w;
  pad_f = params.pad_f;
  pad_b = params.pad_b;
  pad_u = params.pad_u;
  pad_d = params.pad_d;
  pad_l = params.pad_l;
  pad_r = params.pad_r;
  if (params.load3d_special > 1) {
    pad_r = params.pad_r + stride_w;
  }
  dilation_d = params.dilation_d;
  dilation_h = params.dilation_h;
  dilation_w = params.dilation_w;
  mag_factor = params.mag_factor;
  cin1_g = params.a_shape.c1;
  real_g = params.real_g;
  // fp16/fp32 weight(k1,n1,n0,k0), n0 is always 16, k0 = 16(fp16)/8(fp32)
  // in forward cube op, cout1_g of weight and l0c is always aligned by 16(config['mac'][2])
  cout1_g = static_cast<int32_t>(
      MathUtil::CeilDivision(mag_factor * params.c_shape.c / params.groups, static_cast<int64_t>(params.b_shape.c0)));
  hf32_flag = params.hf32_flag;
}

void Conv3DRunInfo::SetTiling(const Conv3DTilingParam &params, const Conv3DTiling &tiling) {
  // set tiling var
  group_dim = tiling.group_dim;
  batch_dim = tiling.batch_dim;
  d_dim = tiling.d_dim;
  batch_dout_single_core =
      MathUtil::CeilDivision(params.a_shape.batch, static_cast<int64_t>(tiling.batch_dim)) *
      MathUtil::CeilDivision(params.c_shape.d, static_cast<int64_t>(tiling.d_dim));
  n_single_core = MathUtil::CeilDivision(params.c_shape.c1, static_cast<int64_t>(tiling.n_dim)) /
                  (tiling.n_bl1 * tiling.n_l0);
  n_dim = tiling.n_dim;
  n_bl1 = tiling.n_bl1;
  n_ub_l0_time = tiling.n_l0 / tiling.n_cub;
  cub_n1 = tiling.n_cub;
  m_dim = tiling.m_dim;
  m_single_core = MathUtil::CeilDivision(
      MathUtil::CeilDivision(
          MathUtil::CeilDivision(params.c_shape.h * params.c_shape.w, params.a_shape.c0),
          static_cast<int64_t>(tiling.m_dim)),
      static_cast<int64_t>(tiling.m_al1) * tiling.m_l0);
  m_al1 = tiling.m_al1;
  m_l0 = tiling.m_l0;
  k_l0 = tiling.k_l0;

  int64_t total_k = params.a_shape.c1 * params.kernel_d * params.kernel_h * params.kernel_w;
  k_al1_factor = static_cast<int32_t>(total_k / tiling.k_al1);
  k_bl1_factor = 1;
  kl1_times = k_al1_factor;
  if (tiling.k_bl1 > 0) {
    k_bl1_factor = static_cast<int32_t>(total_k / tiling.k_bl1);
    kl1_times = std::max(tiling.k_al1, tiling.k_bl1) / std::min(tiling.k_al1, tiling.k_bl1);
  }

  k_al0_factor = tiling.k_al1 / tiling.k_l0;
  k_bl0_factor = tiling.k_bl1 / tiling.k_l0;

  al1_bound = tiling.al1_bound;
  bt_bound = 0;
  if (params.platform_info.support_l0c2out()) {
    int32_t n_l0 = tiling.n_l0;
    if (params.c_dtype == ge::DT_FLOAT16) {
      // if fp16, when move_l1_to_bt need cast to fp32, BT buffer need align to 128B, then n_l0 in BT align to 2
      n_l0 = MathUtil::Align(n_l0, 2);
    }
    bt_bound = params.bias_flag * n_l0 * params.b_shape.c0;
  }
}

void Conv3DRunInfo::SetSpecialScene(const Conv3DTilingParam &params) {
  load3d_special = params.load3d_special;
}

void Conv3dBpFilterRunInfo::Update(const cachetiling::Conv3DBpFilterTilingParam &params,
                                   const cachetiling::Conv3DBpFilterTiling &tiling) {
  SetShape(params);
  SetTiling(params, tiling);
  SetSpecialScene(params);
}

void Conv3dBpFilterRunInfo::SetShape(const cachetiling::Conv3DBpFilterTilingParam &params) {
  // transdata vars
  batch = params.a_shape.batch;
  ci = params.b_shape.c;
  di = params.b_shape.d;
  hi = params.b_shape.h;
  wi = params.b_shape.w;
  co = params.a_shape.c;
  dout = params.a_shape.d;
  ho = params.a_shape.h;
  wo = params.a_shape.w;
  kd = params.kernel_d;
  kh = params.kernel_h;
  kw = params.kernel_w;
  // dw vars
  // ci1/co1 assigned with cin1_g/cout1_g, calculate again
  // for fp32 scene, k0 is 8 and c0 is 16 here, calculate c1 with 8
  ci1 = MathUtil::CeilDivision(params.b_shape.c, static_cast<int64_t>(params.k0));
  co1 = MathUtil::CeilDivision(params.a_shape.c, static_cast<int64_t>(params.k0));
  stride_d = params.stride_d;
  stride_h = params.stride_h;
  stride_w = params.load3d_special == 1 ? params.stride_w : (params.b_shape.w + params.pad_r + params.pad_l);
  pad_f = params.pad_f;
  pad_b = params.pad_b;
  pad_u = params.pad_u;
  pad_d = params.pad_d;
  pad_l = params.pad_l;
  pad_r = params.pad_r;
  dilation_d = params.dilation_d;
  dilation_h = params.dilation_h;
  dilation_w = params.dilation_w;
  cin1_g = params.b_shape.c1;
  cout1_g = params.a_shape.c1;
  real_g = params.real_g;
  mag_factor = params.mag_factor;
  if (params.b_dtype == ge::DT_FLOAT) {
    // calculate c1 with c0(16) in conv3d_backprop_filter.cc, need to calculate again with 8
    cin1_g = MathUtil::CeilDivision(mag_factor * params.b_shape.c / params.groups, static_cast<int64_t>(params.k0));
    cout1_g = MathUtil::Align(mag_factor * params.a_shape.c / params.groups, params.a_shape.c0) / params.k0;
  }
}

void Conv3dBpFilterRunInfo::SetTiling(const Conv3DBpFilterTilingParam &params,
                                      const Conv3DBpFilterTiling &tiling) {
  // set tiling var
  // for fp32 scene, c0 is 16 here and k0 is 8, align to 16 firstly
  int64_t single_core_k = MathUtil::CeilDivision(
      MathUtil::Align(params.a_shape.h * params.a_shape.w, params.b_shape.c0) / params.k0, tiling.k_dim);
  group_dim = tiling.group_dim;
  batch_dim = tiling.batch_dim * tiling.d_dim;
  h_dim = tiling.k_dim;
  batch_dout_single_core = MathUtil::CeilDivision(params.a_shape.batch, static_cast<int64_t>(tiling.batch_dim)) *
                           (params.a_shape.d / tiling.d_dim);
  int32_t kd_split_num = std::max(MathUtil::GetGcd(params.kernel_d, tiling.n_dim), 1);
  n_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(cin1_g, tiling.n_dim / kd_split_num) *
                                             static_cast<int32_t>(params.kernel_d / kd_split_num) * params.kernel_h *
                                             params.kernel_w,
                                         tiling.n_bl1 * tiling.n_l0);
  // for fp32 scene, n_single_core is a multiple of 2 for (2 * output_n0(8) -> L0C n0(16))
  if (params.b_dtype == ge::DT_FLOAT) {
    n_single_core = MathUtil::Align(n_single_core, 2);  // 2: align to even number
  }
  n_dim = tiling.n_dim;
  n_bl1 = tiling.n_bl1;
  n_ub_l0_time = tiling.n_l0 / tiling.n_cub;
  cub_n1 = tiling.n_cub;
  m_dim = tiling.m_dim;
  m_single_core = MathUtil::CeilDivision(
      MathUtil::CeilDivision(params.a_shape.c1, static_cast<int64_t>(tiling.m_dim)),
      static_cast<int64_t>(tiling.m_al1) * tiling.m_l0);
  m_al1 = tiling.m_al1;
  m_l0 = tiling.m_l0;
  k_l0 = tiling.k_l0;
  k_al1_factor = single_core_k * params.load3d_special / tiling.k_al1;
  k_bl1_factor = single_core_k / tiling.k_bl1;
  k_al0_factor = tiling.k_al1 / tiling.k_l0;
  k_bl0_factor = tiling.k_bl1 * params.load3d_special / tiling.k_l0;
  kl1_times = std::max(tiling.k_al1, tiling.k_bl1) / std::min(tiling.k_al1, tiling.k_bl1);
  if (params.load3d_special != 1 && tiling.k_al1 < tiling.k_bl1) {
    kl1_times = kl1_times * params.load3d_special;
  } else if (params.load3d_special != 1 && tiling.k_al1 > tiling.k_bl1) {
    kl1_times = kl1_times / params.load3d_special;
  }

  if (params.split_w_flag) {
    k_al1_factor = MathUtil::CeilDivision(params.a_shape.h, static_cast<int64_t>(tiling.k_dim)) / (tiling.k_al1 / k_l0);
    k_bl1_factor = MathUtil::CeilDivision(params.a_shape.h, static_cast<int64_t>(tiling.k_dim)) / (tiling.k_bl1 / k_l0);
  }

  bl1_bound = tiling.bl1_bound;
  ho_bl1 = tiling.ho_bl1;
}

void Conv3dBpFilterRunInfo::SetSpecialScene(const Conv3DBpFilterTilingParam &params) {
  load3d_special = params.load3d_special;
  is_bf16 = static_cast<int32_t>(params.a_dtype == ge::DT_BF16);
}
}  // namespace optiling
