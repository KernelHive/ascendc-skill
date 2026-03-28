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
 * \file status.cc
 * \brief
 */
#include "cube/algorithm/entity/status.h"

#include <iomanip>
#include <sstream>

namespace optiling {
namespace cachetiling {
static const int32_t kNumTwo = 2;

std::string DimFactor::ToString() const {
  std::stringstream ss;
  ss << "batch: " << batch
     << " m: " << m
     << " k: " << k
     << " n: " << n
     << " d: " << d
     << " group: " << group;
  return ss.str();
}

int32_t DimFactor::ReduceMul() const {
  return batch * m * k * n * d * group;
}

void DimFactor::Init() {
  batch = 1;
  m = 1;
  k = 1;
  n = 1;
  d = 1;
  group = 1;
}

bool DimFactor::IsValid() const {
  return group > 0 && batch > 0 && m > 0 && k > 0 && n > 0 && d > 0;
}

std::string L0Status::ToString() const {
  std::stringstream ss;
  ss << " m: " << m
     << " k: " << k
     << " din: " << din
     << " dk: " << dk
     << " dout: " << dout
     << " n: " << n
     << " db_l0a: " << db_l0a
     << " db_l0b: " << db_l0b
     << " db_l0c: " << db_l0c;
  return ss.str();
}

void L0Status::Init() {
  batch = 1;
  m = 1;
  k = 1;
  n = 1;
  din = 1;
  dk = 1;
  dout = 1;
  db_l0a = kDbOn;
  db_l0b = kDbOn;
  db_l0c = kDbOff;
}

std::string L1Status::ToString() const {
  std::stringstream ss;
  ss << "m_al1: " << m_al1
     << " k_al1: " << k_al1
     << " k_bl1: " << k_bl1
     << " n_bl1: " << n_bl1
     << " db_al1: " << db_al1
     << " db_bl1: " << db_bl1
     << " al1_repeat_time: " << al1_repeat_time
     << " bl1_repeat_time: " << bl1_repeat_time
     << " ho: " << ho
     << " al1_bound: " << al1_bound
     << " bl1_bound: " << bl1_bound
     << " d_al1: " << d_al1
     << " d_bl1: " << d_bl1;
  return ss.str();
}

void L1Status::Init() {
  m_al1 = 1;
  k_al1 = 1;
  k_bl1 = 1;
  n_bl1 = 1;
  db_al1 = kDbOff;
  db_bl1 = kDbOff;
  al1_repeat_time = 1;
  bl1_repeat_time = 1;
  ho = 0;
  al1_bound = 0;
  bl1_bound = 0;
  d_al1 = 1;
  d_bl1 = 1;
}

std::string UbStatus::ToString() const {
  std::stringstream ss;
  ss << "m_aub: " << m_aub
     << " k_aub: " << k_aub
     << " k_bub: " << k_bub
     << " n_bub: " << n_bub
     << " n_cub: " << n_cub
     << " db_aub: " << db_aub
     << " db_bub: " << db_bub
     << " db_cub: " << db_cub;
  return ss.str();
}

void UbStatus::Init() {
  m_aub = 1;
  k_aub = 1;
  k_bub = 1;
  n_bub = 1;
  n_cub = 1;
  batch_cub = 1;
  batch_aub = 1;
  batch_bub = 1;
  db_aub = kDbOn;
  db_bub = kDbOn;
  db_cub = kDbOn;
}

std::string ResourceStatistic::Show(const PlatformInfo &platform_info) const {
  float core_used_ratio = core_used_ * 100.0f / platform_info.core_num();
  float l1_used_ratio = l1_used_ * 100.0f / platform_info.l1_size();
  float l0a_used_ratio = l0a_used_ * 100.0f / platform_info.l0a_size();
  float l0b_used_ratio = l0b_used_ * 100.0f / platform_info.l0b_size();
  float l0c_used_ratio = l0c_used_ * 100.0f / platform_info.l0c_size();
  float ub_used_ratio = ub_used_ * 100.0f / platform_info.ub_size();

  std::stringstream ss;
  ss << std::setiosflags(std::ios::fixed) << std::setprecision(2);  // keep 2 decimal places
  ss << "core_used: " << core_used_ << "[" << core_used_ratio << "%]"
     << " l1_used: " << l1_used_ / kKiloByte << "KB[" << l1_used_ratio << "%]"
     << " l0a_used: " << l0a_used_ / kKiloByte << "KB[" << l0a_used_ratio << "%]"
     << " l0b_used: " << l0b_used_ / kKiloByte << "KB[" << l0b_used_ratio << "%]"
     << " l0c_used: " << l0c_used_ / kKiloByte << "KB[" << l0c_used_ratio << "%]"
     << " ub_used: " << ub_used_ / kKiloByte << "KB[" << ub_used_ratio << "%]";
  return ss.str();
};

std::string SingleCoreStatus::ToString() const {
  std::stringstream ss;
  ss << " orig_shape: " << orig_shape_.ToString()
     << " shape: " << shape_.ToString()
     << " block_dims: " << block_dims_.ToString()
     << " l0_status: " << l0_status_.ToString()
     << " l1_status: " << l1_status_.ToString()
     << " ub_status: " << ub_status_.ToString();
  return ss.str();
}

void SingleCoreStatus::Init() {
  orig_shape_.Init();
  shape_.Init();
  block_dims_.Init();
  l0_status_.Init();
  l1_status_.Init();
  ub_status_.Init();
}

std::string HardwareStatus::ToString() const {
  std::stringstream ss;
  ss << " full_cache_line: " << full_cache_line
     << " mte1_l0a_bandwidth: " << mte1_l0a_bandwidth
     << " mte1_l0b_bandwidth: " << mte1_l0b_bandwidth
     << " l0c_factor_limit: " << l0c_factor_limit
     << " perf_min_core_num: " << perf_min_core_num
     << " input_dtype_bytes: " << input_dtype_bytes
     << " output_dtype_bytes: " << output_dtype_bytes
     << " reduce_block_size: " << reduce_block_size
     << " full_cache_line_unalign: " << full_cache_line_unalign
     << " l0_factor_limit: " << l0_factor_limit
     << " nl0_prefer_size: " << nl0_prefer_size
     << " kl0_prefer_size: " << kl0_prefer_size
     << " bl1_prefer_size: " << bl1_prefer_size
     << " ml0_prefer_size_outer_axis: " << ml0_prefer_size_outer_axis;
  return ss.str();
}

void HardwareStatus::Init(bool support_l0c2out, bool is_fp32_in) {
  // the value for CVsplit
  if (support_l0c2out) {
    full_cache_line = 16; // 16 full cacheline
    mte1_l0a_bandwidth = 256; // 256 Bytes per cycle
    mte1_l0b_bandwidth = 128; // 128 Bytes per cycle
    l0c_factor_limit = 128; // max 128 factors
    perf_min_core_num = 16; // perf 16 core
  } else {
    full_cache_line = 8; // 8 full cacheline
    mte1_l0a_bandwidth = 512; // 512 Bytes per cycle
    mte1_l0b_bandwidth = 256; // 256 Bytes per cycle
    l0c_factor_limit = 256; // max 256 factors
    perf_min_core_num = 24; // perf 24 core
  }

  if (is_fp32_in) {
    full_cache_line /= kNumTwo;
    input_dtype_bytes = kFp32Bytes;
    output_dtype_bytes = kFp32Bytes;
    reduce_block_size = kFP32BlockReduce;
    full_cache_line_unalign = 16; // 16 full cacheline in fp32 for unalign empirically
    l0_factor_limit = 32; // 32 is the buffer of l0c divided by block_size and dtype
    nl0_prefer_size = 8; // 8 is best tiling_nl0 for fp32
    kl0_prefer_size = 2; // 2 is best tiling_kl0 for fp32
    bl1_prefer_size = 8; // 8 is best tiling_k_bl1 for fp32
    ml0_prefer_size_outer_axis = 16; // 16 is best tiling_m_l0 for fp32 when m is out axis
  } else {
    input_dtype_bytes = kFp16Bytes;
    output_dtype_bytes = kFp16Bytes;
    reduce_block_size = kBlockSize;
    full_cache_line_unalign = 32; // 32 full cacheline in fp16 for unalign empirically
    l0_factor_limit = 64; // 64 is the buffer of l0c divided by block_size and dtype
    nl0_prefer_size = 16; // 16 is best tiling_nl0 for fp16
    kl0_prefer_size = 4; // 4 is best tiling_kl0 for fp16
    bl1_prefer_size = 4; // 4 is best tiling_k_bl1 for fp16
    ml0_prefer_size_outer_axis = 8;  // 8 is best tiling_m_l0 for fp16 when m is out axis
  }
}
}  // namespace cachetiling
}  // namespace optiling