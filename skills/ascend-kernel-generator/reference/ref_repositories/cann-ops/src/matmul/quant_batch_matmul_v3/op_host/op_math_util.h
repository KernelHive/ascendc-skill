/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"

#ifdef __cplusplus
extern "C" {
#endif

inline int64_t Ceil(int64_t x, int64_t y) {
  if (y == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
    return INT64_MIN;
  }
  return ((x + y - 1) / y) * y;
}

inline int64_t CeilDiv(int64_t x, int64_t y) {
  if (y == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
    return INT64_MIN;
  }
  return (x + y - 1) / y;
}

#ifdef __cplusplus
}
#endif
