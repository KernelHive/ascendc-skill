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
 * \file shape.cc
 * \brief
 */
#include "cube/algorithm/entity/shape.h"
#include <sstream>

namespace optiling {
namespace cachetiling {
std::string TilingShape::ToString() const {
  std::stringstream ss;
  ss << "batch: " << batch
     << " m: " << m
     << " k: " << k
     << " n: " << n
     << " group: " << group
     << " h: " << h
     << " w: " << w
     << " din: " << din
     << " dk: " << dk
     << " dout: " << dout;
  return ss.str();
}

void TilingShape::Init() {
  batch = 1;
  m = 1;
  k = 1;
  n = 1;
  group = 1;
  h = 1;
  w = 1;
  din = 1;
  dk = 1;
  dout = 1;
}
}  // namespace cachetiling
}  // namespace optiling