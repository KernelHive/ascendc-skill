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
 * \file param_checker.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_PARAM_CHECKER_H
#define OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_PARAM_CHECKER_H

#include <vector>
#include "cube/include/cube_tiling_param.h"

#define OFFSET(clazz, member) ((intptr_t)(&(static_cast<clazz *>(0))->member))
#define SUBCLAZZ_OFFSET(clazz, member, sub_member) ((intptr_t)(&(static_cast<clazz *>(0))->member.sub_member))
#define RANGE_ELEMENT(clazz, member, lower, upper) Range(#member, OFFSET(clazz, member), lower, upper)
#define RANGE_CLAZZ_ELEMENT(clazz, member, sub_member, lower, upper) \
  Range(#sub_member, SUBCLAZZ_OFFSET(clazz, member, sub_member), lower, upper)

namespace optiling {
namespace cachetiling {
class Range {
public:
  Range(const char *in_name, uint32_t in_offset, int32_t in_lower, int32_t in_upper) :
    name(in_name), offset(in_offset), lower(in_lower), upper(in_upper) {};
  const char *name = nullptr;
  uint32_t offset = 0;
  int32_t lower = 0;
  int32_t upper = 0;
};

// shape
constexpr int32_t kShapeLower = 1;
constexpr int32_t kShapeUpper = INT32_MAX - 1;
constexpr int32_t kResolutionLower = 1;
constexpr int32_t kResolutionUpper = 4096;

// attrs
constexpr int32_t kStrideUpper = 63;
constexpr int32_t kDilationLower = 1;
constexpr int32_t kDilationUpper = 255;
constexpr int32_t kAttrDUpper = INT32_MAX - 1;
constexpr int32_t kGroupLower = 1;
constexpr int32_t kGroupUpper = UINT16_MAX;
constexpr int32_t kRealGroupLower = 1;
constexpr int32_t kRealGroupUpper = INT32_MAX;

std::vector<Range> kConv2DBackPropFilterParamRange {
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, batch, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, c, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, h, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, w, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, batch, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, c, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, h, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, w, kResolutionLower, kShapeUpper),
};

std::vector<Range> kConv3DParamRange {
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, batch, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, c, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, d, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, h, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, w, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, batch, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, c, kShapeLower, kShapeUpper),
  RANGE_ELEMENT(Conv3DTilingParam, kernel_d, kShapeLower, kShapeUpper),  // kd follow batch
  RANGE_ELEMENT(CubeTilingParam, groups, kGroupLower, kGroupUpper),
  RANGE_ELEMENT(CubeTilingParam, real_g, kRealGroupLower, kRealGroupUpper),
};

std::vector<Range> kConv3DBackPropFilterParamRange {
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, batch, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, c, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, d, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, h, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, a_shape, w, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, batch, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, c, kShapeLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, d, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, h, kResolutionLower, kShapeUpper),
  RANGE_CLAZZ_ELEMENT(CubeTilingParam, b_shape, w, kResolutionLower, kShapeUpper),
};

std::vector<std::pair<OpType, std::vector<Range>>> kOpRange {
  {kConv2DBackpropFilter, kConv2DBackPropFilterParamRange},
  {kConv3D, kConv3DParamRange},
  {kConv3DBackpropFilter, kConv3DBackPropFilterParamRange},
  {kConv2DBackpropFilterV2, kConv2DBackPropFilterParamRange},
  {kConv3DBackpropFilterV2, kConv3DBackPropFilterParamRange},
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_PARAM_CHECKER_H