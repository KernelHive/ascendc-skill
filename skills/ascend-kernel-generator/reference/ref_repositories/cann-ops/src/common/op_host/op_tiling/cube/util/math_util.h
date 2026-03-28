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
 * \file math_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "op_util.h"
#include "cube/include/cube_tiling_param.h"

namespace optiling {
namespace cachetiling {
constexpr int32_t kExtremeNumSize = 2;
class MathUtil {
 public:
  template <typename T>
  static T CeilDivision(T num1, T num2) {
    return ops::CeilDiv(num1, num2);
  }

  static int64_t CeilDivision(int64_t num1, int32_t num2) {
    return ops::CeilDiv(num1, static_cast<int64_t>(num2));
  }

  template <typename T>
  static T Min(T num1, T num2) {
    return std::min(num1, num2);
  }

  static int32_t Min(int64_t num1, int32_t num2);
  static int32_t Min(int32_t num1, int64_t num2);
  static bool IsEqual(float l_value, float r_value);
  static int64_t Align(int32_t num1, int32_t num2);
  static uint64_t Align(uint32_t num1, uint32_t num2);
  static int64_t Align(int64_t num1, int32_t num2);
  static int64_t Align(int64_t num1, int64_t num2);
  static uint64_t Align(uint64_t num1, uint64_t num2);
  static int64_t Align(int64_t num1, uint32_t num2);
  static int64_t Align(int32_t num1, int64_t num2);
  static int32_t AlignDown(int32_t num1, int32_t num2);
  static int32_t GetGcd(int32_t param1, int32_t param2);
  static void GetFactors(int32_t factor_list[], int32_t src_num, size_t &index, int32_t max_factor,
                         int32_t min_factor = 1);
  static void GetFactors(std::vector<int32_t> &factor_list, int64_t src_num, int32_t max_factor);
  static void GetFactors(int32_t factor_list[], int32_t src_num, size_t &index,
                         const struct FactorConfig &factor_config);
  static size_t GetTwoFactors(std::array<int32_t, kExtremeNumSize> &res, int32_t base, int64_t dim,
                               std::array<int32_t, kExtremeNumSize> &limit, int32_t cur_factor);

  static int64_t NearestFactor(int64_t base, int64_t factor, bool even_factor = false);
  static int32_t NearestFactor(int32_t base, int32_t factor, bool even_factor = false) {
    return NearestFactor(static_cast<int64_t>(base), static_cast<int64_t>(factor), even_factor);
  }
  static int32_t NearestFactor(int64_t base, int32_t factor, bool even_factor = false) {
    return NearestFactor(base, static_cast<int64_t>(factor), even_factor);
  }
  static bool GenNearestFactor(int32_t factor, int32_t dim, int32_t factor_optional[]);
  static inline bool CheckRange(int32_t val, int32_t lower, int32_t upper) { return val >= lower && val <= upper; }
  static int64_t MapShape(int64_t shape, bool round_up_flag);
  static int64_t GetNonFactorMap(std::vector<int32_t> &factor_list, int64_t src_num, int32_t max_factor);
  static int64_t FindBestSingleCore(const int64_t ori_shape, const int64_t mapped_shape, const int32_t core_num,
                                    bool is_k_dim);
  static void GetFactorCnt(const int32_t shape, int32_t &factor_cnt, const int32_t factor_start,
                           const int32_t factor_end);
  static void GetFactors(std::vector<int32_t> &factor_list, int32_t src_num, int32_t min_factor, int32_t max_factor);
  static void GetFactors(std::vector<int64_t> &factor_list, int64_t src_num, int32_t min_factor, int32_t max_factor);
  static void GetFactorLayerCnt(const int64_t shape, int32_t &factor_cnt, const int32_t factor_start,
                                const int32_t factor_end);
  static bool CheckFactorNumSatisfy(const int64_t shape);
  static void AddCoreFactor(int32_t dim, int32_t core_num, std::vector<int32_t> &dims_factors);
  static bool IsPrime(int32_t x);
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_
