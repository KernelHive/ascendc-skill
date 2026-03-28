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
 * \file error_log.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#include <inttypes.h>
namespace optiling {

#define OPPROTO_SUBMOD_NAME "OP_PROTO"

class OpLog {
  public:
   static uint64_t GetTid() {
 #ifdef __GNUC__
     const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
 #else
     const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
 #endif
     return tid;
   }
 };

 #define OpLogSub(moduleId, level, op_info, fmt, ...)                                                                   \
    do {                                                                                                                \
      if (AlogCheckDebugLevel(static_cast<int>(moduleId), level) == 1) {                                                \
                  AlogRecord(static_cast<int>(moduleId), DLOG_TYPE_DEBUG, level,                                        \
                      "[%s:%d][%s][%s][%" PRIu64 "] OpName:[%s] " #fmt,                                                 \
                      __FILE__, __LINE__, OPPROTO_SUBMOD_NAME,                                                          \
                      __FUNCTION__, OpLog::GetTid(), get_cstr(op_info), ##__VA_ARGS__);                                 \
        }                                                                                                               \
  } while (0)


#define D_OP_LOGE(opname, fmt, ...) OpLogSub(OP, DLOG_ERROR, opname, fmt, ##__VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(opname, ...) D_OP_LOGE("custom", __VA_ARGS__)

inline const char* get_cstr(const std::string& str) {
  return str.c_str();
}

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)                                       \
  do {                                                                                               \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                         \
  } while (0)

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
