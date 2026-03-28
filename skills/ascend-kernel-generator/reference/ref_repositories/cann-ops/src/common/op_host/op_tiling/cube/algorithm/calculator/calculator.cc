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
 * \file calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
bool Calculator::IsL1SizeValid(int64_t l1_size) const {
  return params_->platform_info.IsValidL1Size(l1_size);
}

bool Calculator::Init(const CubeTilingParam &params) {
  params_ = &params;
  return true;
}

void Calculator::Clear() {
  params_ = nullptr;
}
}  // namespace cachetiling
}  // namespace optiling