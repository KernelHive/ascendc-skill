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
 * \file configuration.cc
 * \brief
 */

#include "cube/util/configuration.h"

#include <cstdint>
#include "op_log.h"

namespace optiling {
namespace cachetiling {
Configuration &Configuration::Instance() {
  static Configuration inst;
  return inst;
}

Configuration::Configuration() {
  int32_t enable = 1; //CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
  is_debug_mode_ = (enable == 1);
}

bool Configuration::IsDebugMode() { return Instance().is_debug_mode_; }
}  // namespace cachetiling
}  // namespace optiling