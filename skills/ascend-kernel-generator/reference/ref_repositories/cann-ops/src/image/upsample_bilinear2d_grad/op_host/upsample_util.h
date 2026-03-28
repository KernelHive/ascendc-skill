/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_SRC_LEVEL2_UPSAMPLE_UTIL_H_
#define OP_API_SRC_LEVEL2_UPSAMPLE_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

bool CheckInputElement(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_SRC_LEVEL2_UPSAMPLE_UTIL_H_