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
 * \file shape.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_SHAPE_H
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_SHAPE_H

#include <cstdint>
#include <string>

namespace optiling {
namespace cachetiling {
class TilingShape {
 public:
  std::string ToString() const;
  void Init();

  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t group = 1;
  int64_t h = 1;
  int64_t w = 1;
  int64_t din = 1;
  int64_t dk = 1;
  int64_t dout = 1;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_SHAPE_H