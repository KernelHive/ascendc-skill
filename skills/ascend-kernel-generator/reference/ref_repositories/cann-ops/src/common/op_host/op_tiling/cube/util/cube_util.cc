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
 * \file cube_util.cc
 * \brief
 */

#include "cube/util/cube_util.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "cube/constants/constants_define.h"
#include "cube/util/math_util.h"
#include "op_log.h"

namespace optiling {
namespace cachetiling {
int32_t CubeUtil::CalcHo(int64_t k, int32_t wo, const std::string &op_type) {
  // OP_LOGE_IF((k == 0 || wo == 0), 0, op_type, "ho is 0 in CalcHo function.");
  // 完整K是ho*wo，k可能超int32，但是下面除完以后的wo不可能超int32
  int32_t ho = static_cast<int32_t>(MathUtil::CeilDivision(k, wo));
  if (k % wo == 0 || wo % k == 0) {
    return ho;
  } else {
    return ho + 1;
  }
}

int64_t CubeUtil::LoopNumFromSingleCoreToL0(const TilingShape &tilingShape, const DimFactor &blockDimsFactor,
                                            const PlatformInfo &platform_info) {
  if (!blockDimsFactor.IsValid()) {
    return 0;
  }

  int32_t l0c_size = platform_info.l0c_size() / kFp32Bytes / kDbOn / kCubeTileNumSize;
  int32_t group_single_core_size = MathUtil::CeilDivision(tilingShape.group, blockDimsFactor.group);
  int32_t batch_single_core_size = MathUtil::CeilDivision(tilingShape.batch, blockDimsFactor.batch);
  int64_t m_single_core_size = MathUtil::CeilDivision(tilingShape.m, blockDimsFactor.m) * tilingShape.w;
  int64_t n_single_core_size = tilingShape.n / blockDimsFactor.n;

  l0c_size = MathUtil::Min(m_single_core_size * n_single_core_size, l0c_size);
  if (l0c_size == 0) {
    return 0;
  }
  int64_t loop_num =
      group_single_core_size * batch_single_core_size * m_single_core_size * n_single_core_size / l0c_size;
  // special scence n_single_core_size is 1
  if (n_single_core_size == 1) {
    int32_t l0a_size = MathUtil::Min(m_single_core_size, kL0aNzSize);
    int32_t m_single_ml0_num = MathUtil::CeilDivision(m_single_core_size, l0a_size);
    if (m_single_ml0_num == 0) {
      return 0;
    }
    loop_num = group_single_core_size * batch_single_core_size * m_single_ml0_num * n_single_core_size;
  }
  return loop_num;
}


static uint64_t CalSpecialL0BSize(const PlatformInfo &platform_info, ge::DataType data_type,
                                  cachetiling::OpType op_type, int32_t n0) {
  uint64_t l0b_size = platform_info.l0b_size();
  if (data_type == ge::DT_FLOAT && op_type == cachetiling::kConv2DBackpropInput) {
    // dx算子fp32输入时，B矩阵使用load3d指令载入到L0,从load3d参数计算出的空间占用比实际使用的空间要小，
    // 所以在L0B上预留一部分空间，避免出现超空间或精度问题
    l0b_size -= n0 * kDtypeCubeTileNumSizeMap.at(data_type) * kDbOn * kFp32Bytes;
  }
  return l0b_size;
}

bool CubeUtil::CheckL0Overflow(int32_t m0, int32_t n0, int32_t k0, const CubeTilingParam &params) {
  int32_t cube_tile_num_size_l0c = kCubeTileNumSize * kDbOn;
  int32_t cube_tle_num_size_l0ab = kDtypeCubeTileNumSizeMap.at(params.a_dtype) * kDbOn;
  PlatformInfo platform_info = params.platform_info;
  uint64_t l0b_size = CalSpecialL0BSize(platform_info, params.a_dtype, params.type, n0);
  bool l0a_invalid = static_cast<uint64_t>(m0 * k0 * params.a_dtype_bytes * cube_tle_num_size_l0ab) >
                     platform_info.l0a_size();
  bool l0b_invalid = static_cast<uint64_t>(n0 * k0 * params.b_dtype_bytes * cube_tle_num_size_l0ab) > l0b_size;
  bool l0c_invalid = static_cast<uint64_t>(n0 * m0 * kFp32Bytes * cube_tile_num_size_l0c) > platform_info.l0c_size();
  bool l0_invalid = l0a_invalid || l0b_invalid || l0c_invalid;
  return !l0_invalid;
}

bool CubeUtil::CheckL0Size(const L0Status &l0_status, const GemmTilingParam &params) {
  uint32_t al0_size = static_cast<uint32_t>(l0_status.m * l0_status.k * params.block_size * params.reduce_block_size *
                                            ge::GetSizeByDataType(params.a_dtype));
  uint32_t bl0_size = static_cast<uint32_t>(l0_status.k * l0_status.n * params.block_size * params.reduce_block_size *
                                            ge::GetSizeByDataType(params.b_dtype));
  uint32_t cl0_size = static_cast<uint32_t>(l0_status.m * l0_status.n * params.block_size * params.block_size *
                                            ge::GetSizeByDataType(params.c_dtype));
  return (al0_size <= params.platform_info.l0a_size() &&
          bl0_size <= params.platform_info.l0b_size() &&
          cl0_size <= params.platform_info.l0c_size());
}

bool CubeUtil::CheckL1Size(const GemmTilingParam &params, int32_t amat, int32_t bmat, int32_t cur_bias_l1_size) {
  uint64_t load_size_bytes = (static_cast<uint64_t>(amat) + bmat) * params.block_size *
                              params.reduce_block_size * ge::GetSizeByDataType(params.a_dtype) +
                              cur_bias_l1_size;
  return load_size_bytes <= params.platform_info.l1_size();
}

int32_t CubeUtil::GetBiasL1Size(const GemmTilingParam &params, const L1Status &l1_status) {
  // the number 32 means reserve 32B of l1 space for bias_l1_to_bt
  return l1_status.n_bl1 * params.block_size * ge::GetSizeByDataType(params.bias_dtype) + 32;
}

int32_t CubeUtil::GetBiasTableSize(const GemmTilingParam &params, const L0Status &l0_status) {
  return l0_status.n * params.block_size * kFp32Bytes;
}

int32_t CubeUtil::GetMinKl1(int32_t k_l0, int32_t k_hw) {
  int32_t gcd_num = MathUtil::GetGcd(k_l0, k_hw);
  if (gcd_num == 0 || k_hw == 0) {
    return 0;
  }
  int32_t lcm_num = k_l0 * k_hw / gcd_num;
  // the k_l1 is factor of c1
  int32_t k_l1 = lcm_num / k_hw;
  return k_l1;
}

int64_t CubeUtil::CalcAL1Size(const CubeTilingParam *params, const L1Status &l1_status, const L0Status &l0_status) {
  return static_cast<int64_t>(l1_status.m_al1) * l0_status.m * params->a_shape.c0 * l1_status.k_al1 * params->k0 *
         l1_status.db_al1 * params->a_dtype_bytes;
}

int64_t CubeUtil::CalcBL1Size(const CubeTilingParam *params, const L1Status &l1_status, const L0Status &l0_status) {
  if (params->dma_flag) {
    return static_cast<int64_t>(l1_status.k_bl1) * params->k0 * l1_status.n_bl1 * l0_status.n * params->b_shape.c0 *
           l1_status.db_bl1 * params->b_dtype_bytes;
  }
  int32_t hi = 1;
  int32_t wi = 1;
  CalcHiWi(params, l1_status.k_bl1, l0_status.k, hi, wi);

  int32_t bl1_n = l1_status.n_bl1 * l0_status.n;
  int32_t cur_ci1_factor = CalcBL1N(params, bl1_n);
  if (params->load2d_flag) {
    // in schedule, set storage_align in load2d template to CUBE_MUL_SHAPE(16 * 16), no load2d template in fp32
    return MathUtil::Align(static_cast<int64_t>(hi) * wi * params->b_shape.c0, kCubeTileNumSize) * cur_ci1_factor *
           l1_status.db_bl1 * params->b_dtype_bytes;
  } else {
    return static_cast<int64_t>(hi) * wi * cur_ci1_factor * kBlockSize * l1_status.db_bl1 * params->b_dtype_bytes;
  }
}

void CubeUtil::CalcHiWi(const CubeTilingParam *params, int32_t k_bl1, int32_t k_l0, int32_t &hi, int32_t &wi) {
  if (params->conv1d_flag) {
    hi = 1;
    wi = CalcWi(k_bl1 * params->k0, params->stride_w, params->kernel_w_dilation, params->b_shape.w);
    return;
  }

  if (params->split_w_flag) {
    int32_t ho = k_bl1 / k_l0;
    hi = CalcHi(ho, params->stride_h, params->kernel_h_dilation, params->b_shape.h);
    wi = CalcWi(k_l0 * params->k0, params->stride_w, params->kernel_w_dilation, params->b_shape.w);
    return;
  }

  int32_t ho = CalcHo(k_bl1 * params->k0, params->a_shape.w, params->op_type);
  hi = CalcHi(ho, params->stride_h, params->kernel_h_dilation, params->b_shape.h);
  wi = params->b_shape.w;
}

void CubeUtil::CalcMinHiWi(const CubeTilingParam *params, int32_t min_k_l0, int32_t &hi, int32_t &wi) {
  if (params->conv1d_flag) {
    hi = 1;
    wi = CalcWi(min_k_l0 * params->k0, params->stride_w, params->kernel_w_dilation, params->b_shape.w);
    return;
  }

  if (params->split_w_flag) {
    hi = CalcHi(1, params->stride_h, params->kernel_h_dilation, params->b_shape.h); // ho is 1
    wi = CalcWi(min_k_l0 * params->k0, params->stride_w, params->kernel_w_dilation, params->b_shape.w);
    return;
  }

  int32_t ho = CalcHo(min_k_l0 * params->k0, params->a_shape.w, params->op_type);
  hi = CalcHi(ho, params->stride_h, params->kernel_h_dilation, params->b_shape.h);
  wi = params->b_shape.w;
}

int32_t CubeUtil::CalcBL1N(const CubeTilingParam *params, int32_t bl1_n) {
  int32_t kernel_hw = params->kernel_h * params->kernel_w;
  if (params->b_shape.c0 == kSmallChannelSize) {
    kernel_hw = MathUtil::CeilDivision(kernel_hw, kSmallChannelSize);
  }
  int32_t cur_ci1_factor = MathUtil::CeilDivision(bl1_n, kernel_hw);
  int32_t extend_line = 0;
  int32_t double_bl1_n = bl1_n * 2; // 2 * bl1_n means special cases of loading
  if (kernel_hw > bl1_n) {
    if ((bl1_n != 0) && kernel_hw % bl1_n != 0) {
      extend_line = 1;
    }
  } else {
    if (bl1_n % kernel_hw == 0) {
      extend_line = 0;
    } else if (double_bl1_n % kernel_hw == 0) {
      extend_line = 1;
    } else {
      extend_line = 2; // 2 means Load 2 additional lines
    }
  }
  return cur_ci1_factor + extend_line;
}


int64_t CubeUtil::GetDfactorSdEqKd(const Conv3DBpInputTilingParam *conv3ddx_param_, int32_t l0c_din) {
  // ----[该函数用于计算StrideD=KerneD时 由Din和Dk 反推的Dout的大小]----

  int64_t kernel_idx = conv3ddx_param_->pad_h % conv3ddx_param_->filter_d_dilation - 1;
  int64_t dedy_dout_used = 1;
  int64_t dedy_dout_used_max = 1;
  for (int dedx_idx = 0; dedx_idx < conv3ddx_param_->c_shape.d; ++dedx_idx) {
      kernel_idx += 1;
      if (kernel_idx == conv3ddx_param_->filter_d_dilation) {
          kernel_idx = 0;
          dedy_dout_used += 1;
      }
      if (dedx_idx % l0c_din == 0) {
          dedy_dout_used = 1;
      }
      if (dedy_dout_used > dedy_dout_used_max) {
          dedy_dout_used_max = dedy_dout_used;
      }
  }
  return dedy_dout_used_max;
}

int64_t CubeUtil::CalcL1Size(const CubeTilingParam *params, const L1Status &l1_status, const L0Status &l0_status) {
  return CalcAL1Size(params, l1_status, l0_status) + CalcBL1Size(params, l1_status, l0_status);
}

int32_t CubeUtil::CalcBL1Cog(int32_t k_bl1, ge::DataType data_type) {
  if (data_type == ge::DT_FLOAT) {
    // fp32输入时，k_bl1是按8对齐的Co1g的因子
    return MathUtil::CeilDivision(k_bl1, kDtypeCompensateFactor) * kBlockSize;
  }
  return k_bl1 * kBlockSize;
}


void CubeUtil::CalcMal1(const CubeTilingParam *params, const TilingShape &shape, int64_t extend_shape_k,
                        const L0Status &l0_status, L1Status &l1_status) {
  // when split w, al1/bl1 attach flag is 2, then attach to L0C, so n_bl1 and m_al1 should be 1
  if (shape.batch != 1 || l1_status.k_al1 != extend_shape_k || params->split_w_flag || params->dma_flag) {
    return;
  }
  int64_t remain_al1_size = params->platform_info.l1_size() - CubeUtil::CalcBL1Size(params, l1_status, l0_status);
  int32_t max_m_al1 = MathUtil::CeilDivision(shape.m, l0_status.m);
  l1_status.m_al1 = 1;
  l1_status.m_al1 = MathUtil::Min(remain_al1_size / CubeUtil::CalcAL1Size(params, l1_status, l0_status), max_m_al1);
  l1_status.m_al1 = MathUtil::NearestFactor(max_m_al1, l1_status.m_al1);
}

void CubeUtil::CalcNbl1(const CubeTilingParam *params, const TilingShape &shape, const L0Status &l0_status,
                        int64_t total_n_l1, L1Status &l1_status) {
  // when split w, al1/bl1 attach flag is 2, then attach to L0C, so n_bl1 and m_al1 should be 1
  if (shape.batch != 1 || l1_status.k_bl1 != shape.k || params->split_w_flag || params->dma_flag) {
    return;
  }

  int64_t remain_bl1_size = params->platform_info.l1_size() - CubeUtil::CalcAL1Size(params, l1_status, l0_status);
  int64_t max_n_bl1 = MathUtil::CeilDivision(total_n_l1, l0_status.n);
  l1_status.n_bl1 = 1;
  l1_status.n_bl1 = MathUtil::Min(remain_bl1_size / CubeUtil::CalcBL1Size(params, l1_status, l0_status), max_n_bl1);
  l1_status.n_bl1 = MathUtil::NearestFactor(max_n_bl1, l1_status.n_bl1);
  if (params->load3d_flag && params->platform_info.support_l0c2out()) {
    while (l1_status.n_bl1 * l0_status.n * params->b_shape.c0 > kMaxLoad3dV2Kstart) {
      l1_status.n_bl1 = MathUtil::NearestFactor(max_n_bl1, l1_status.n_bl1 - 1);
    }
  }
  // load3d special scene nbl1 under l0c, n_bl1 must 1
  // for fp32 scene, n_l0 and n_bl1 can only be 1 for now
  if (params->load3d_special != 1) {
    l1_status.n_bl1 = 1;
  }
}

void CubeUtil::CalcKal1(const CubeTilingParam *params, const TilingShape &/* shape */, int64_t extend_shape_k,
                        const L0Status &l0_status, L1Status &l1_status) {
  int32_t remain_al1_size = params->platform_info.l1_size() - CubeUtil::CalcBL1Size(params, l1_status, l0_status);
  int32_t kal1 = std::min(remain_al1_size / (l1_status.m_al1 * l0_status.m * params->a_shape.c0 * params->k0 *
                                             l1_status.db_al1 * params->a_dtype_bytes),
                          static_cast<int64_t>(extend_shape_k));
  int32_t al1_time = kal1 / l0_status.k;
  int64_t max_times = extend_shape_k / l0_status.k;
  al1_time = MathUtil::NearestFactor(max_times, al1_time);
  l1_status.k_al1 = al1_time * l0_status.k;
}

void CubeUtil::CalcKbl1(const CubeTilingParam *params, int64_t extend_shape_k, const L0Status &l0_status,
                        L1Status &l1_status) {
  int32_t remain_bl1_size = params->platform_info.l1_size() - CubeUtil::CalcAL1Size(params, l1_status, l0_status);
  int32_t cur_ci1_factor = CubeUtil::CalcBL1N(params, l1_status.n_bl1 * l0_status.n);
  int32_t hi = remain_bl1_size /
               (params->b_shape.w * cur_ci1_factor * params->b_shape.c0 * l1_status.db_bl1 * params->b_dtype_bytes);
  int32_t ho = (hi - params->dilation_h * (params->kernel_h - 1) - 1) / params->stride_h + 1;
  int64_t k_max =
      MathUtil::CeilDivision(ho * params->load3d_special * params->a_shape.w, static_cast<int64_t>(params->k0));
  if (params->conv1d_flag) {
    int32_t wi = remain_bl1_size /
                 (params->b_shape.h * cur_ci1_factor * params->b_shape.c0 * l1_status.db_bl1 * params->b_dtype_bytes);
    if (wi <= params->b_shape.w) {
      int32_t wo = (wi - params->dilation_w * (params->kernel_w - 1) - 1) / params->stride_w + 1;
      k_max = 1 * wo / params->k0;
    } else {
      k_max = extend_shape_k;
    }
  }
  int64_t max_times = extend_shape_k / l0_status.k;
  int64_t bl1_times = std::max(MathUtil::Min(k_max / l0_status.k, max_times), 1L);
  if (params->split_w_flag) {
    int32_t wi =
        CubeUtil::CalcWi(l0_status.k * params->k0, params->stride_w, params->kernel_w_dilation, params->b_shape.w);
    hi = remain_bl1_size / (wi * cur_ci1_factor * params->b_shape.c0 * l1_status.db_bl1 * params->b_dtype_bytes);

    ho = (hi - params->kernel_h_dilation) / params->stride_h + 1;
    bl1_times = ho > 0 ? MathUtil::Min(ho, max_times) : max_times;
  }
  if (ho <= 0 && !params->split_w_flag && !params->conv1d_flag) {
    bl1_times = max_times;
  }
  bl1_times = MathUtil::NearestFactor(max_times, bl1_times);
  int64_t k_bl1 = bl1_times * l0_status.k / params->load3d_special;
  while (k_bl1 > INT32_MAX) {
    bl1_times--;
    bl1_times = MathUtil::NearestFactor(max_times, bl1_times);
    k_bl1 = bl1_times * l0_status.k / params->load3d_special;
  }
  l1_status.k_bl1 = k_bl1;

  int32_t bl1_size = CubeUtil::CalcBL1Size(params, l1_status, l0_status);
  while (bl1_size > remain_bl1_size) {
    bl1_times--;
    bl1_times = MathUtil::NearestFactor(max_times, bl1_times);
    l1_status.k_bl1 = bl1_times * l0_status.k / params->load3d_special;
    bl1_size = CubeUtil::CalcBL1Size(params, l1_status, l0_status);
  }

  OP_LOGD(params->op_type, "get final k_bl1: %d", l1_status.k_bl1);
}

int32_t CubeUtil::GetKFactor(const CubeTilingParam *params, int64_t orig_shape_k, int32_t used_core) {
  if (used_core <= 0) {
    OP_LOGD(params->op_type, "used_core(%d) <= 0", used_core);
    return 0;
  }
  // conv2d dw support non factor k
  int32_t k_factor = params->platform_info.core_num() / used_core;
  k_factor = MathUtil::Min(k_factor, orig_shape_k);
  if (params->linear_embedding_opti_flag) {
    k_factor = std::max(k_factor, 1);
  }
  // only cut ho as k_dim when split w, make sure wo factor in L0 is even number is enough
  if (params->b_dtype == ge::DT_FLOAT && !(params->split_w_flag)) {
    int64_t single_core_k = MathUtil::CeilDivision(orig_shape_k, k_factor);
    // for fp32 scene, k0 is 8 and k_l0 * k0 should align to 16, so single_core_k should be even number
    while ((single_core_k & 0x1) != 0) {
      --k_factor;
      if (k_factor == 0) {
        return 0;
      }

      single_core_k = MathUtil::CeilDivision(orig_shape_k, k_factor);
    }
  }

  return k_factor;
}

void CubeUtil::CalcL1RemainStatus(const CubeTilingParam *params, const L0Status &l0_status, L1Status &l1_status) {
  if (params->conv1d_flag) {
    l1_status.ho = 1;
  } else if (params->split_w_flag) {
    l1_status.ho = l1_status.k_bl1 / l0_status.k;
  } else {
    l1_status.ho = CubeUtil::CalcHo(l1_status.k_bl1 * params->k0, params->a_shape.w, params->op_type);
  }

  int32_t bl1_n = l0_status.n * l1_status.n_bl1;
  int32_t bl1_ci = CubeUtil::CalcBL1N(params, bl1_n);
  int32_t hi = 1;
  int32_t wi = 1;
  CubeUtil::CalcHiWi(params, l1_status.k_bl1, l0_status.k, hi, wi);
  if (params->load2d_flag) {
    // in schedule, set storage_align in load2d template to CUBE_MUL_SHAPE(16 * 16), no load2d template in fp32
    l1_status.bl1_bound = MathUtil::Align(hi * wi * params->b_shape.c0, kCubeTileNumSize) * bl1_ci;
  } else if (params->dma_flag) {
    l1_status.bl1_bound = bl1_n * l1_status.k_bl1 * params->k0 * params->b_shape.c0;
  } else {
    l1_status.bl1_bound = hi * wi * bl1_ci * params->b_shape.c0;
  }
}

int64_t CubeUtil::GetLoad3dThroughput(bool is_dst_l0a, int32_t row_switch_times, int32_t stride_w) {
  /*
    | stride | case               | throughput-L0A (B/cycle) | throughput-L0B (B/cycle)    |
    ----------------------------------------------------------------------------------------
    | 1      | no row switch      | 512 / 2                    | 512 / 4                   |
    ----------------------------------------------------------------------------------------
    | 1      | row switch n times | 512 / (n + 2)              | 512 / (n + 4)             |
    ----------------------------------------------------------------------------------------
    | 2/4/8  | no row switch      | 512 / (stride * 2)         | 512 / (stride * 2 + 2)    |
    ----------------------------------------------------------------------------------------
    | 2/4/8  | row switch n times | 512 / (stride * 2 + n)     | 512 / (stride * 2 + n + 2)|
    ----------------------------------------------------------------------------------------
    | other  | all case           | 32                         | 512 / 18                  |
  */
  int64_t throughput = is_dst_l0a ? 32 : 28;  // 512 / 18;
  int64_t factor_band = is_dst_l0a ? 0 : 2;
  if (stride_w == 1) {
    throughput = 512 / (row_switch_times + 2 + factor_band); // 512 / (row_switch_times + 4)
  } else if (stride_w == 2 || stride_w == 4 || stride_w == 8) { // stridew is 2,4,8
    throughput = 512 / (row_switch_times + factor_band + stride_w * 2); // 512 / (row_switch_times + 4 + stride_w);
  }
  return throughput;
}

int32_t CubeUtil::GetMilanPkgNumByCacheline(int32_t burst_len) {
  // 1971 kCachelinePkgSize is 16, 8, 4, 3, 2, 1
  int32_t pkg_num_1 = burst_len & 0x1;
  int32_t pkg_num_2 = (burst_len >> 1) & 0x1;
  int32_t pkg_num_4 = (burst_len >> 2) & 0x1;
  int32_t pkg_num_8 = (burst_len >> 3) & 0x1;
  int32_t pkg_num_16 = burst_len >> 4;
  int32_t pkg_num_3 = std::min(pkg_num_1, pkg_num_2);
  pkg_num_1 -= pkg_num_3;
  pkg_num_2 -= pkg_num_3;
  return pkg_num_1 + pkg_num_2 + pkg_num_3 + pkg_num_4 + pkg_num_8 + pkg_num_16;
}

int32_t CubeUtil::GetObpPkgNumByCacheline(int32_t burst_len) {
  // cache line pkg size: {8, 4, 3, 2, 1}, calc without 3 first
  int64_t pkg_num_1 = burst_len & 0x1;
  int64_t pkg_num_2 = (burst_len >> 1) & 0x1;
  int64_t pkg_num_4 = (burst_len >> 2) & 0x1;
  int64_t pkg_num_8 = burst_len >> 3;
  int64_t pkg_num_3 = std::min(pkg_num_1, pkg_num_2);
  pkg_num_1 -= pkg_num_3;
  pkg_num_2 -= pkg_num_3;
  return pkg_num_1 + pkg_num_2 + pkg_num_3 + pkg_num_4 + pkg_num_8;
}
}  // namespace cachetiling
}  // namespace optiling