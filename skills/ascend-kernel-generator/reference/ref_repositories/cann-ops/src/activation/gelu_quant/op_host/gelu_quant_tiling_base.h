/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file gelu_quant_tiling_base.h
 */
#ifndef GELU_QUANT_TILING_BASE_H
#define GELU_QUANT_TILING_BASE_H

#include <cstdint>
#include <string>

namespace optiling {
// 公共变量区
constexpr int64_t X_INPUT_INDEX = 0;
constexpr int64_t SCALE_INPUT_INDEX = 1;
constexpr int64_t OFFSET_INPUT_INDEX = 2;

constexpr int64_t Y_OUTPUT_INDEX = 0;
constexpr int64_t SCALE_OUTPUT_INDEX = 1;

constexpr int64_t APPROXIMATE_ATTR_INDEX = 0;
constexpr int64_t QUANT_MODE_ATTR_INDEX = 1;

constexpr int64_t STATIC_QUANT_MODE = 0;
constexpr int64_t DYNAMIC_QUANT_MODE = 1;

constexpr int64_t APPROXIMATE_NONE = 0;
constexpr int64_t APPROXIMATE_TANH = 1;

constexpr int64_t EMPTY_TENSOR = 0;
constexpr int64_t SCALAR_TENSOR = 1;
constexpr int64_t NORMAL_TENSOR = 2;

constexpr int64_t RESERVED_UB_SIZE = 8 * 1024; // 8k
constexpr int64_t INPUT_MIN_DIMENSIONS = 2;
constexpr int64_t INPUT_MAX_DIMENSIONS = 8;
constexpr int64_t FP32_BLOCK_NUM = 8;

constexpr int64_t STATIC_QUANT_PER_TENSOR_COEXISTING_QUANTITY = 4;
constexpr int64_t STATIC_QUANT_COEXISTING_QUANTITY = 6;
constexpr int64_t DYNAMIC_QUANT_COEXISTING_QUANTITY = 6;
constexpr int64_t DYNAMIC_QUANT_WORKSPACE_COEXISTING_QUANTITY = 7;

constexpr int64_t STATIC_PER_TENSOR_TEMPLATE = 0;
constexpr int64_t STATIC_FUNCTION_TEMPLATE = 1;
constexpr int64_t STATIC_PERFORMANCE_TEMPLATE = 2;
constexpr int64_t DYNAMIC_NORMAL_TEMPLATE = 3;
constexpr int64_t DYNAMIC_WORKSPACE_TEMPLATE = 4;

constexpr int64_t SINGLE_CORE_PROCESS_MIN_NUM = 128;
constexpr int64_t TWO_END_AXIS = 2;

constexpr int64_t WORKSPACE_BUFFER = 20 * 1024 * 1024;

enum class InputDataType : int64_t { HALF_HALF = 1, HALF_FLOAT = 2, FLOAT_FLOAT = 3, BF16_BF16 = 4, BF16_FLOAT = 5 };

// 公共方法区
template <class T> inline T AlignToCeil(const T n, const T alignSize)
{
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <class T> inline T AlignToFloor(const T n, const T alignSize)
{
    return n & (~(alignSize - 1));
}

template <class T> inline T CeilDivide(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <class T> inline T SafeDivide(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2;
}

int64_t GetMod(const int64_t l_value, const int64_t r_value) {
  if (r_value == 0) {
    return l_value;
  }
  return l_value % r_value;
}

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
  if ((ptr) == nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    std::printf("EZ9999 op[%s], %s is nullptr!", name, #ptr);                           \
    return ret;                                                                                  \
  }

inline const std::string& get_op_info(const std::string& str) {
  return str;
}

inline const char* get_op_info(const char* str) {
  return str;
}

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
} // namespace optiling

#endif // GELU_QUANT_TILING_BASE_H
