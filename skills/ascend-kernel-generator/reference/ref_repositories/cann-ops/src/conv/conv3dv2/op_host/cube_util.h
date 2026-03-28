/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef OP_API_CUBE_UTIL_H_
 #define OP_API_CUBE_UTIL_H_

#include <vector>
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
struct Conv3DInputShapes {
  int64_t in = 0;
  int64_t id = 0;
  int64_t ih = 0;
  int64_t iw = 0;
  int64_t ic = 0;

  int64_t kn = 0;
  int64_t kd = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t kc = 0;
};

struct Conv3DAttrs {
  bool ceil_mode = false; // only for AvgPool3D

  int64_t strd = 0;
  int64_t strh = 0;
  int64_t strw = 0;

  int64_t dild = 1;
  int64_t dilh = 1;
  int64_t dilw = 1;

  int64_t groups = 1;

  int64_t padf = 0;
  int64_t padb = 0;
  int64_t padu = 0;
  int64_t padd = 0;
  int64_t padl = 0;
  int64_t padr = 0;
};

// NDHWC
constexpr size_t kDDimNDHWCIdx = 1;
constexpr size_t kHDimNDHWCIdx = 2;
constexpr size_t kWDimNDHWCIdx = 3;
// NCDHW
constexpr size_t kDDimNCDHWIdx = 2;
constexpr size_t kHDimNCDHWIdx = 3;
constexpr size_t kWDimNCDHWIdx = 4;

// DHWCN
constexpr size_t kDDimDHWCNIdx = 0;
constexpr size_t kHDimDHWCNIdx = 1;
constexpr size_t kWDimDHWCNIdx = 2;

bool GetConv3DXShape(gert::InferShapeContext *context, size_t x_idx, ge::Format x_format, bool avg_pool3d,
                     Conv3DInputShapes &shapes);
bool GetConv3DFilterShape(gert::InferShapeContext *context, size_t filter_idx, Conv3DInputShapes &shapes);
bool GetConv3DStridesAndDilations(const gert::InferShapeContext *context, ge::Format x_format, Conv3DAttrs &attrs);
bool GetConv3DPads(const gert::InferShapeContext *context, const Conv3DInputShapes &shapes, size_t pads_idx,
                   size_t padding_idx, Conv3DAttrs &attrs);
}  // namespace ops
#endif