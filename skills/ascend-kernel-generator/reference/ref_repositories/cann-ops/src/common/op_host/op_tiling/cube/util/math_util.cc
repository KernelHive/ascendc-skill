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
 * \file math_util.cc
 * \brief function of math_util
 */
#include "cube/util/math_util.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "op_log.h"

namespace optiling {
namespace cachetiling {
static const int32_t kSeedMapMin = 16;
static const int32_t kSeedMapMax = 1024;
static const int32_t FactorNumLimit = 4;
static const int32_t kL0FactorNumLimit = 2;
static const int32_t kL1FactorNumLimit = 4;
static const int32_t kMinFactorLimit = 8;
static const int32_t kL0FactorLimit = 64;
static const int32_t kL1FactorLimit = 128;
bool MathUtil::IsEqual(float l_value, float r_value) {
  return std::fabs(l_value - r_value) <= std::numeric_limits<float>::epsilon();
}

int32_t MathUtil::Min(int64_t num1, int32_t num2) {
  return static_cast<int32_t>(std::min(num1, static_cast<int64_t>(num2)));
}

int32_t MathUtil::Min(int32_t num1, int64_t num2) {
  return static_cast<int32_t>(std::min(num2, static_cast<int64_t>(num1)));
}

int64_t MathUtil::Align(int32_t num1, int32_t num2) { return static_cast<int64_t>(CeilDivision(num1, num2)) * num2; }

uint64_t MathUtil::Align(uint32_t num1, uint32_t num2) {
  return static_cast<uint64_t>(CeilDivision(num1, num2)) * num2;
}

int64_t MathUtil::Align(int64_t num1, int32_t num2) { return CeilDivision(num1, static_cast<int64_t>(num2)) * num2; }

int64_t MathUtil::Align(int64_t num1, int64_t num2) { return CeilDivision(num1, num2) * num2; }

uint64_t MathUtil::Align(uint64_t num1, uint64_t num2) { return CeilDivision(num1, num2) * num2; }

int64_t MathUtil::Align(int64_t num1, uint32_t num2) { return CeilDivision(num1, static_cast<int64_t>(num2)) * num2; }

int64_t MathUtil::Align(int32_t num1, int64_t num2) { return CeilDivision(static_cast<int64_t>(num1), num2) * num2; }

int32_t MathUtil::AlignDown(int32_t num1, int32_t num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 / num2) * num2;
}

int32_t MathUtil::GetGcd(int32_t param1, int32_t param2) {
  // get greatest common divisor of param1 and param2
  if (param1 < param2) {
    std::swap(param1, param2);
  }
  if (param2 == 0) {
    return 0;
  }
  if (param1 % param2 == 0) {
    return param2;
  } else {
    return GetGcd(param2, param1 - param2);
  }
}

void MathUtil::GetFactors(int32_t factor_list[], int32_t src_num, size_t &index, int32_t max_factor,
                          int32_t min_factor) {
  for (int32_t factor = max_factor; factor >= min_factor && factor != 0; factor--) {
    if (src_num % factor == 0) {
      factor_list[index++] = factor;
    }
  }
}

void MathUtil::GetFactors(int32_t factor_list[], int32_t src_num, size_t &index,
                          const struct FactorConfig &factor_config) {
  for (int32_t factor = factor_config.max; factor >= factor_config.min && factor != 0; factor-=factor_config.step) {
    if (src_num % factor == 0) {
      factor_list[index++] = factor;
    }
  }
}

bool MathUtil::CheckFactorNumSatisfy(const int64_t dim) {
  if (dim <= kMinFactorLimit) {
    return true;
  }
  int32_t factor_l0_cnt = 0;
  int32_t factor_l1_cnt = 0;
  MathUtil::GetFactorLayerCnt(dim, factor_l0_cnt, 1, kL0FactorLimit);
  if (dim > kL1FactorLimit) {
    MathUtil::GetFactorLayerCnt(dim, factor_l1_cnt, kL0FactorLimit + 1, kL1FactorLimit);
  }
  bool factor_num_not_satisfied = (factor_l0_cnt <= kL0FactorNumLimit) ||
                                  ((dim > kL1FactorLimit) && (factor_l0_cnt + factor_l1_cnt <= kL1FactorNumLimit));
  return !factor_num_not_satisfied;
}

int64_t MathUtil::GetNonFactorMap(std::vector<int32_t> &factor_list, int64_t src_num, int32_t max_factor) {
  int32_t factor_cnt = 0;
  int64_t map_factor = src_num;
  MathUtil::GetFactorLayerCnt(src_num, factor_cnt, 1, max_factor);
  if (src_num > 1 && factor_cnt <= FactorNumLimit) {
    map_factor = MathUtil::MapShape(src_num, true);
  }
  GetFactors(factor_list, map_factor, max_factor);
  return map_factor;
}

int64_t MathUtil::FindBestSingleCore(const int64_t ori_shape, const int64_t mapped_shape, const int32_t core_num,
                                     bool is_k_dim) {
  int64_t best_single_core = ori_shape;
  int64_t real_single_core = MathUtil::CeilDivision(ori_shape, core_num);
  int64_t mapped_single_core = MathUtil::CeilDivision(mapped_shape, core_num);

  if (is_k_dim) {
    int64_t best_shape = ori_shape % core_num == 0 ? ori_shape : mapped_shape;
    best_single_core = MathUtil::CeilDivision(best_shape, core_num);
    return best_single_core;
  }

  if (core_num == 1 && CheckFactorNumSatisfy(best_single_core)) {
    return best_single_core;
  }

  mapped_single_core = MathUtil::CeilDivision(mapped_shape, core_num);
  best_single_core = real_single_core;
  while (best_single_core != mapped_single_core) {
    if (CheckFactorNumSatisfy(best_single_core)) {
      return best_single_core;
    }
    if (best_single_core < mapped_single_core) {
      ++best_single_core;
    } else {
      --best_single_core;
    }
  }
  return best_single_core;
}

void MathUtil::GetFactors(std::vector<int32_t> &factor_list, int64_t src_num, int32_t max_factor) {
  int32_t max_num = MathUtil::Min(src_num, max_factor);
  for (int32_t factor = 1; factor <= max_num; factor++) {
    if (src_num % factor == 0) {
      factor_list.push_back(factor);
    }
  }
}

bool MathUtil::GenNearestFactor(int32_t factor, int32_t dim, int32_t factor_optional[]) {
  int32_t cur_factor = std::min(factor + 1, dim);
  if (factor == 0 || cur_factor == 0) {
    return false;
  }

  while (dim % cur_factor != 0) {
    cur_factor++;
  }
  factor_optional[0] = cur_factor;

  cur_factor = factor;
  while (dim % cur_factor != 0) {
    cur_factor--;
  }
  factor_optional[1] = cur_factor;
  return true;
}

size_t MathUtil::GetTwoFactors(std::array<int32_t, kExtremeNumSize>& res, int32_t base, int64_t dim,
                               std::array<int32_t, kExtremeNumSize>& limit, int32_t cur_factor) {
  // for up bigger or equal to base + 1, find the smallest num which is a factor of dim
  // form down smaller or equal to base, find the biggest num which is a factor of dim
  // the result number must be smaller than max_num and bigger than min_num
  // and cur_factor must be a factor of the result number
  int32_t max_num = limit[0];
  int32_t min_num = limit[1];
  size_t idx = 0;
  int32_t up = MathUtil::Align(base + 1, cur_factor);
  while (up <= dim && up <= max_num && up != 0) {
    if (dim % up == 0) {
      res[idx++] = up;
      break;
    }
    up += cur_factor;
  }

  int32_t down = base / cur_factor * cur_factor;
  down = MathUtil::Min(dim, down);
  while (down >= min_num && down != 0) {
    if (dim % down == 0) {
      res[idx++] = down;
      if (idx == kExtremeNumSize) {
        break;
      }
    }
    down -= cur_factor;
  }
  return idx;
}

int64_t MathUtil::NearestFactor(int64_t base, int64_t factor, bool even_factor) {
  if (factor >= base) {
    return base;
  }

  if (factor == 0) {
    return 0;
  }

  if (even_factor && ((base & 0x1) != 0 || factor == 1)) {
    return 0;
  }

  // if need even factor, ignore odd factor
  while (base % factor != 0 || (even_factor && (factor & 0x1) != 0)) {
    factor--;
  }
  return factor;
}

int64_t MathUtil::MapShape(int64_t shape, bool round_up_flag) {
  int64_t seed = static_cast<int64_t>(kSeedMapMin);
  if (shape < seed) {
    return shape;
  }
  while (seed < kSeedMapMax) {
    if (seed < shape && (seed << 1 >= shape)) {
      break;
    }
    seed = seed << 1;
  }
  if (round_up_flag) {
    return seed << 1;
  }
  return seed;
}

void MathUtil::GetFactorCnt(const int32_t shape, int32_t &factor_cnt, const int32_t factor_start,
                            const int32_t factor_end) {
  for (int32_t i = factor_start; i <= factor_end; i++) {
    if (shape < i || i == 0) {
      return;
    }
    if (shape % i == 0) {
      ++factor_cnt;
    }
  }
}

void MathUtil::GetFactorLayerCnt(const int64_t shape, int32_t &factor_cnt, const int32_t factor_start,
                                 const int32_t factor_end) {
  std::vector<int32_t> factor_list;
  MathUtil::GetFactors(factor_list, shape, factor_start, factor_end);
  for (const auto factor : factor_list) {
    int32_t fcnt = 0;
    GetFactorCnt(factor, fcnt, 1, factor + 1);
    factor_cnt = fcnt >= factor_cnt ? fcnt : factor_cnt;
  }
}

void MathUtil::GetFactors(std::vector<int64_t> &factor_list, int64_t src_num, int32_t min_factor, int32_t max_factor) {
  for (int32_t factor = max_factor; factor >= min_factor && factor != 0; factor--) {
    if (src_num % factor == 0) {
      factor_list.push_back(factor);
    }
  }
}

void MathUtil::GetFactors(std::vector<int32_t> &factor_list, int32_t src_num, int32_t min_factor, int32_t max_factor) {
  for (int32_t factor = max_factor; factor >= min_factor && factor != 0; factor--) {
    if (src_num % factor == 0) {
      factor_list.push_back(factor);
    }
  }
}

void MathUtil::AddCoreFactor(int32_t dim, int32_t core_num, std::vector<int32_t> &dims_factors) {
  for (auto iter = dims_factors.cbegin(); iter != dims_factors.cend(); ++iter) {
    if (*iter > dim) {
      iter = dims_factors.erase(iter, dims_factors.cend());
      break;
    }
  }

  int32_t max_core_num = std::min(core_num, dim);
  MathUtil::GetFactors(dims_factors, core_num, max_core_num);
  sort(dims_factors.begin(), dims_factors.end());
  (void)dims_factors.erase(unique(dims_factors.begin(), dims_factors.end()), dims_factors.cend());
}

bool MathUtil::IsPrime(int32_t num) {
  if (num == 1) {
    return false;
  }
  // 2 and 3 are prime
  if (num == 2 || num == 3) {
    return true;
  }
  // when num >= 5, if num is prime, it must be on both sides of 6x(x>=1), otherwise, filter out it.
  if (num % 6 != 1 && num % 6 != 5) {
    return false;
  }
  // both sides of 6x(x>=1) are not necessarily prime, so need judge whether it is on both sides of 6x from 5
  for (int32_t i = 5; i * i <= num; i += 6) {
    // i and i+2 are numbers on both sides of 6x
    if (num % i == 0 || num % (i + 2) == 0) {
      return false;
    }
  }
  return true;
}
}  // namespace cachetiling
}  // namespace optiling
