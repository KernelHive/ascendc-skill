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
 * \file cube_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_CUBE_RUNTIME_RUNTIME_UTIL_H_
#define OPS_BUILT_IN_OP_PROTO_CUBE_RUNTIME_RUNTIME_UTIL_H_

#include "types.h"
#include "runtime/infer_shape_context.h"
#include <vector>

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

struct Conv2DInputShapes {
  int64_t in = 0;
  int64_t ih = 0;
  int64_t iw = 0;
  int64_t ic = 0;

  int64_t kn = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t kc = 0;
};

struct Conv2DAttrs {
  int64_t str_h = 0;
  int64_t str_w = 0;

  int64_t dil_h = 1;
  int64_t dil_w = 1;

  int64_t groups = 1;

  int64_t pad_u = 0;
  int64_t pad_d = 0;
  int64_t pad_l = 0;
  int64_t pad_r = 0;
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

// NCHW
constexpr size_t kHDimNCHWIdx = 2;
constexpr size_t kWDimNCHWIdx = 3;

// NHWC
constexpr size_t kHDimNHWCIdx = 1;
constexpr size_t kWDimNHWCIdx = 2;

// Deconv Strides IDX
constexpr size_t kHDimNCHWStridesIdx = 0;
constexpr size_t kWDimNCHWStridesIdx = 1;

bool GetConv3DXShape(gert::InferShapeContext *context, size_t x_idx, ge::Format x_format, bool avg_pool3d,
                     Conv3DInputShapes &shapes);
bool GetConv3DFilterShape(gert::InferShapeContext *context, size_t filter_idx, Conv3DInputShapes &shapes);
bool GetConv3DStridesAndDilations(const gert::InferShapeContext *context, ge::Format x_format, Conv3DAttrs &attrs);
bool GetConv3DPads(const gert::InferShapeContext *context, const Conv3DInputShapes &shapes, size_t pads_idx,
                   size_t padding_idx, Conv3DAttrs &attrs);
bool GetConvGroups(const gert::InferShapeContext *context, size_t groups_idx, Conv3DAttrs &attrs);
bool GetConv2DXShape(gert::InferShapeContext *context, size_t x_idx, ge::Format x_format, Conv2DInputShapes &shapes);
bool GetConv2DFilterShape(gert::InferShapeContext *context, size_t filter_idx, Conv2DInputShapes &shapes);
bool GetConv2DStridesAndDilations(const gert::InferShapeContext *context, ge::Format x_format, Conv2DAttrs &attrs,
                                  bool is_deconvolution);
bool GetConv2DGroups(gert::InferShapeContext *context, size_t groups_idx, Conv2DAttrs &attrs);
bool GetConv2DPads(const gert::InferShapeContext *context, const Conv2DInputShapes &shapes,
                   const std::vector<size_t> &idx_vec, Conv2DAttrs &attrs);
}  // namespace ops

#endif  // OPS_BUILT_IN_OP_PROTO_CUBE_RUNTIME_RUNTIME_UTIL_H_
