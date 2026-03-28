/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_COMMON_UTILS_KB_LOG_H_
#define RUNTIME_KB_COMMON_UTILS_KB_LOG_H_
#include <base/alog_pub.h>
#include "mmpa_api.h"

constexpr int TUNE_MODULE = static_cast<int>(TUNE);

#define CANNKB_LOGD(format, ...)                                                                    \
  do {                                                                                              \
    if (AlogCheckDebugLevel(TUNE_MODULE, DLOG_DEBUG) == 1) {                                        \
                AlogRecord(TUNE_MODULE, DLOG_TYPE_DEBUG, DLOG_DEBUG,                                \
                    "[%s:%d][%s][Tid:%d]" format "\n",                                              \
                    __FILE__, __LINE__, "CANNKB",                                                   \
                    mmGetTid(), ##__VA_ARGS__);                                                     \
            }                                                                                       \
  } while (0)

#define CANNKB_LOGI(format, ...)                                                                  \
  do {                                                                                            \
    if (AlogCheckDebugLevel(TUNE_MODULE, DLOG_INFO) == 1) {                                        \
                AlogRecord(TUNE_MODULE, DLOG_TYPE_DEBUG, DLOG_INFO,                                \
                    "[%s:%d][%s][Tid:%d]" format "\n",                                              \
                    __FILE__, __LINE__, "CANNKB",                                                   \
                    mmGetTid(), ##__VA_ARGS__);                                                     \
            }                                                                                       \
  } while (0)

#define CANNKB_LOGW(format, ...)                                                                  \
  do {                                                                                            \
    if (AlogCheckDebugLevel(TUNE_MODULE, DLOG_WARN) == 1) {                                        \
                AlogRecord(TUNE_MODULE, DLOG_TYPE_DEBUG, DLOG_WARN,                                \
                    "[%s:%d][%s][Tid:%d]" format "\n",                                              \
                    __FILE__, __LINE__, "CANNKB",                                                   \
                    mmGetTid(), ##__VA_ARGS__);                                                     \
            }                                                                                       \
  } while (0)

#define CANNKB_LOGE(format, ...)                                                                   \
  do {                                                                                             \
    if (AlogCheckDebugLevel(TUNE_MODULE, DLOG_ERROR) == 1) {                                        \
                AlogRecord(TUNE_MODULE, DLOG_TYPE_DEBUG, DLOG_ERROR,                                \
                    "[%s:%d][%s][Tid:%d]" format "\n",                                              \
                    __FILE__, __LINE__, "CANNKB",                                                   \
                    mmGetTid(), ##__VA_ARGS__);                                                     \
            }                                                                                       \
  } while (0)

#define CANNKB_LOGEVENT(format, ...)                                                               \
  do {                                                                                             \
    if (AlogCheckDebugLevel(TUNE_MODULE, DLOG_INFO) == 1) {                                        \
                AlogRecord(TUNE_MODULE, DLOG_TYPE_DEBUG, DLOG_INFO,                                \
                    "[%s:%d][%s][Tid:%d]" format "\n",                                              \
                    __FILE__, __LINE__, "CANNKB",                                                   \
                    mmGetTid(), ##__VA_ARGS__);                                                     \
            }                                                                                       \
  } while (0)
#endif
