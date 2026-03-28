/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_LEVEL0_RFFT1D_H_
#define OP_API_INC_LEVEL0_RFFT1D_H_

#include "opdev/op_executor.h"

namespace l0op {
bool IsRfft1DAiCoreSupported(const aclTensor* self, int64_t n, int64_t norm);
const aclTensor* Rfft1D(const aclTensor* self, const aclTensor* dft, int64_t n, int64_t norm, aclOpExecutor* executor);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_RFFT1D_H_
