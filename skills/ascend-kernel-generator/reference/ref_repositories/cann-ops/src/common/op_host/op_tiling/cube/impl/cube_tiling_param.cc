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
 * \file cache_tiling_param.cc
 * \brief
 */
#include "cube/include/cube_tiling_param.h"

#include <map>
#include <sstream>
#include <vector>

#include "graph/utils/type_utils.h"
#include "cube/include/param_checker.h"
#include "cube/platform/instruction_param.h"
#include "cube/util/cube_util.h"
#include "cube/util/math_util.h"
#include "op_log.h"

namespace {
const size_t kNchwDimN = 0;
const size_t kNchwDimC = 1;
const size_t kNchwDimH = 2;
const size_t kNchwDimW = 3;

const size_t kNcdhwDimN = 0;
const size_t kNcdhwDimC = 1;
const size_t kNcdhwDimD = 2;
const size_t kNcdhwDimH = 3;
const size_t kNcdhwDimW = 4;

const int64_t kDefaultC0 = 16;
const int32_t kFp32BlockReduce = 8;

const int32_t kLoad3dSpecial = 2;

constexpr int32_t kPadUpper = INT32_MAX - 1;
constexpr int32_t kPadLower = INT32_MIN + 1;
}  // namespace

namespace optiling {
namespace cachetiling {
const char *kOpType2Str[kOpTypeNum] = {
    "Conv2DBackpropFilter", "Conv2DBackpropInput", "Conv2DTranspose", "DepthwiseConv2DBackpropInput", "Gemm", "Conv3D",
    "Conv3DBackpropFilter", "Conv3DBackpropInput", "Conv3DTranspose", "Conv2DBackpropFilterV2", "Conv3DBackpropFilterV2"
};

template <typename T>
static void GetNCDHWShape(const T &origin_shape, int64_t *ncdhw_shape, ge::Format origin_format) {
  // caller already checked buffer size
  size_t idx = 0;
  if (origin_format == ge::FORMAT_NDHWC) {
    ncdhw_shape[idx++] = origin_shape[0];  // 0: N
    ncdhw_shape[idx++] = origin_shape[4];  // 4: C
    ncdhw_shape[idx++] = origin_shape[1];  // 1: D
    ncdhw_shape[idx++] = origin_shape[2];  // 2: H
    ncdhw_shape[idx++] = origin_shape[3];  // 3: W
  } else if (origin_format == ge::FORMAT_NCDHW) {
    ncdhw_shape[idx++] = origin_shape[0];  // 0: N
    ncdhw_shape[idx++] = origin_shape[1];  // 1: C
    ncdhw_shape[idx++] = origin_shape[2];  // 2: D
    ncdhw_shape[idx++] = origin_shape[3];  // 3: H
    ncdhw_shape[idx++] = origin_shape[4];  // 4: W
  } else if (origin_format == ge::FORMAT_DHWCN) {
    ncdhw_shape[idx++] = origin_shape[4];  // 4: N
    ncdhw_shape[idx++] = origin_shape[3];  // 3: C
    ncdhw_shape[idx++] = origin_shape[0];  // 0: D
    ncdhw_shape[idx++] = origin_shape[1];  // 1: H
    ncdhw_shape[idx++] = origin_shape[2];  // 2: W
  }
}

template <typename T>
static void GetNCHWShape(const T &origin_shape, int64_t *nchw_shape, ge::Format origin_format) {
  size_t idx = 0;
  if (origin_format == ge::FORMAT_NHWC) {
    nchw_shape[idx++] = origin_shape[0];  // 0: N
    nchw_shape[idx++] = origin_shape[3];  // 3: C
    nchw_shape[idx++] = origin_shape[1];  // 1: H
    nchw_shape[idx++] = origin_shape[2];  // 2: W
  } else if (origin_format == ge::FORMAT_NCHW) {
    nchw_shape[idx++] = origin_shape[0];  // 0: N
    nchw_shape[idx++] = origin_shape[1];  // 1: C
    nchw_shape[idx++] = origin_shape[2];  // 2: H
    nchw_shape[idx++] = origin_shape[3];  // 3: W
  } else if (origin_format == ge::FORMAT_HWCN) {
    nchw_shape[idx++] = origin_shape[3];  // 3: N
    nchw_shape[idx++] = origin_shape[2];  // 2: C
    nchw_shape[idx++] = origin_shape[0];  // 0: H
    nchw_shape[idx++] = origin_shape[1];  // 1: W
  }
}

template <typename T>
static void NormalizeShape(const T &origin_shape, int64_t *normalized_shape, size_t dim_num, ge::Format origin_format) {
  if (dim_num == kConv2DOriShapeDim) {
    GetNCHWShape(origin_shape, normalized_shape, origin_format);
  } else if (dim_num == kConv3DOriShapeDim) {
    GetNCDHWShape(origin_shape, normalized_shape, origin_format);
  }
}

std::string Shape::ToString() const {
  std::stringstream ss;
  ss << "batch: " << batch << " c: " << c << " d: "<< d << " h: " << h << " w: " << w << " c1: " << c1 << " c0: " << c0;
  return ss.str();
}

bool Shape::operator==(const Shape &param) const {
  return batch == param.batch && c == param.c && h == param.h && w == param.w;
}

CubeTilingParam::CubeTilingParam(OpType cube_type) : type(cube_type) {
  if (type < kOpTypeNum) {
    op_type = kOpType2Str[type];
  } else {
    op_type = "Default";
  }
}

bool CubeTilingParam::ParseOpInfo(const TilingContext *context, const CubeCompileInfo &compile_info) {
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(!SetShape(context), false, op_name, "fail to set shape from tiling context");
  OP_LOGE_IF(!SetAttrs(context), false, op_name, "fail to set attrs from tiling context");
  platform_info.SetRuntimePlatformInfo(compile_info);
  OP_LOGE_IF(!platform_info.IsValid(), false, op_name, "invalid platform info");
  UpdateInstrictionParam();
  SetSpecialSceneParams();
  return true;
}

bool CubeTilingParam::SetShape(const gert::TilingContext *context) {
  const auto op_name = context->GetNodeName();
  const auto fmap_desc = GetFmapTensorDesc(context);
  const auto y_desc = GetYTensorDesc(context);
  const auto filter_desc = GetFilterTensorDesc(context);
  const auto fmap_shape = GetFmapTensorShape(context);
  const auto y_shape = GetYTensorShape(context);
  const auto filter_shape = GetFilterTensorShape(context);
  OP_LOGE_IF(fmap_desc == nullptr || y_desc == nullptr || filter_desc == nullptr || fmap_shape == nullptr ||
                 y_shape == nullptr || filter_shape == nullptr,
             false, op_name, "null input/output shape/desc.");

  OP_LOGE_IF(!CheckFormat(context), false, op_name, "Input or Output format is wrong.");
  auto filter_ori_format = filter_desc->GetOriginFormat();
  auto fmap_ori_format = fmap_desc->GetOriginFormat();
  auto y_ori_format = y_desc->GetOriginFormat();

  size_t shape_dim_num = GetValidOriShapeDimNum();
  OP_LOGE_IF(shape_dim_num != fmap_shape->GetOriginShape().GetDimNum(), false, op_name, "invalid fmap ori shape");
  OP_LOGE_IF(shape_dim_num != filter_shape->GetOriginShape().GetDimNum(), false, op_name, "invalid filter ori shape.");
  OP_LOGE_IF(shape_dim_num != y_shape->GetOriginShape().GetDimNum(), false, op_name, "invalid y ori shape.");

  std::vector<int64_t> normalized_fmap_shape(shape_dim_num, 0);
  std::vector<int64_t> normalized_filter_shape(shape_dim_num, 0);
  std::vector<int64_t> normalized_y_shape(shape_dim_num, 0);

  NormalizeShape(fmap_shape->GetOriginShape(), normalized_fmap_shape.data(), shape_dim_num, fmap_ori_format);
  NormalizeShape(filter_shape->GetOriginShape(), normalized_filter_shape.data(), shape_dim_num, filter_ori_format);
  NormalizeShape(y_shape->GetOriginShape(), normalized_y_shape.data(), shape_dim_num, y_ori_format);
  OP_LOGE_IF(!SetOneShapeTensor(normalized_fmap_shape.data(), shape_dim_num, fmap()), false, op_name,
             "invalid fmap x ori shape");
  OP_LOGE_IF(!SetOneShapeTensor(normalized_filter_shape.data(), shape_dim_num, filter()), false, op_name,
             "invalid filter ori shape.");
  OP_LOGE_IF(!SetOneShapeTensor(normalized_y_shape.data(), shape_dim_num, y()), false, op_name, "invalid y ori shape.");
  SetC0(context);
  kernel_h = filter().h;
  kernel_w = filter().w;
  bias_flag = GetBiasTensorDesc(context) != nullptr;
  if (bias_flag) {
    SetDtypeWithBias(fmap_desc->GetDataType(), filter_desc->GetDataType(), y_desc->GetDataType(),
             GetBiasTensorDesc(context)->GetDataType());
  } else {
    SetDtype(fmap_desc->GetDataType(), filter_desc->GetDataType(), y_desc->GetDataType());
  }
  b_dtype_bytes = ge::GetSizeByDataType(a_dtype);
  OP_LOGE_IF(b_dtype_bytes == -1, false, op_name, "a_shape dtype size is invalid");
  a_dtype_bytes = ge::GetSizeByDataType(b_dtype);
  OP_LOGE_IF(a_dtype_bytes == -1, false, op_name, "b_shape dtype size is invalid");
  c_dtype_bytes = ge::GetSizeByDataType(c_dtype);
  OP_LOGE_IF(c_dtype_bytes == -1, false, op_name, "c_shape dtype size is invalid");

  return true;
}

bool CubeTilingParam::SetOneShapeTensor(const int64_t *normalized_shape, size_t dim_num, Shape &shape) {
  if (dim_num == kConv3DOriShapeDim) {
    shape.batch = normalized_shape[kNcdhwDimN];
    shape.d = normalized_shape[kNcdhwDimD];
    shape.h = normalized_shape[kNcdhwDimH];
    shape.w = normalized_shape[kNcdhwDimW];
    shape.c = normalized_shape[kNcdhwDimC];
    OP_LOGE_IF(shape.batch <= 0, false, op_type, "shape_batch is [%ld], should be no less than 1.", shape.batch);
    OP_LOGE_IF(shape.d <= 0, false, op_type, "shape_d is [%ld], should be no less than 1.", shape.d);
    OP_LOGE_IF(shape.h <= 0, false, op_type, "shape_h is [%ld], should be no less than 1.", shape.h);
    OP_LOGE_IF(shape.w <= 0, false, op_type, "shape_w is [%ld], should be no less than 1.", shape.w);
    OP_LOGE_IF(shape.c <= 0, false, op_type, "shape_c size is [%ld], should be no less than 1.", shape.c);
  } else if (dim_num == kConv2DOriShapeDim) {
    shape.batch = normalized_shape[kNchwDimN];
    shape.h = normalized_shape[kNchwDimH];
    shape.w = normalized_shape[kNchwDimW];
    shape.c = normalized_shape[kNchwDimC];
    OP_LOGE_IF(shape.batch <= 0, false, op_type, "shape_batch is [%ld], should be no less than 1.", shape.batch);
    OP_LOGE_IF(shape.d <= 0, false, op_type, "shape_d is [%ld], should be no less than 1.", shape.d);
    OP_LOGE_IF(shape.h <= 0, false, op_type, "shape_h is is [%ld], should be no less than 1.", shape.h);
    OP_LOGE_IF(shape.w <= 0, false, op_type, "shape_w is is [%ld], should be no less than 1.", shape.w);
  }

  shape.c0 = kDefaultC0;
  shape.c1 = MathUtil::CeilDivision(shape.c, shape.c0);
  return true;
}

bool CubeTilingParam::SetAttrs(const TilingContext *context) {
  const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
  const auto attrs = context->GetAttrs();
  OP_LOGE_IF(attrs == nullptr, false, op_name, "failed to get attrs from context.");

  const auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(GetStridesIdx());
  const auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(GetPadsIdx());
  const auto dilations = attrs->GetAttrPointer<gert::ContinuousVector>(GetDialtionsIdx());

  OP_LOGE_IF(strides == nullptr, false, op_name, "get strides from context fail.");
  OP_LOGE_IF(strides->GetSize() != GetValidStridesDimNum(), false, op_name, "strides of context len is invalid.");
  OP_LOGE_IF(dilations == nullptr, false, op_name, "get dilations from context fail.");
  OP_LOGE_IF(dilations->GetSize() != GetValidDilationsDimNum(), false, op_name, "dilations of context len is invalid.");
  OP_LOGE_IF(pads == nullptr, false, op_name, "get pads from context fail.");
  OP_LOGE_IF(pads->GetSize() != GetValidPadsDimNum(), false, op_name, "pads of context len is invalid.");

  if (GetGroupsIdx() != std::numeric_limits<uint64_t>::max()) {
    const auto groups_attr = attrs->GetAttrPointer<int64_t>(GetGroupsIdx());
    OP_LOGE_IF(groups_attr == nullptr, false, op_name, "get groups from context fail.");
    groups = static_cast<int32_t>(*groups_attr);
  }

  const int64_t *strides_data = reinterpret_cast<const int64_t *>(strides->GetData());
  const int64_t *pads_data = reinterpret_cast<const int64_t *>(pads->GetData());
  const int64_t *dilations_data = reinterpret_cast<const int64_t *>(dilations->GetData());

  auto x_format = GetFmapTensorDesc(context)->GetOriginFormat();
  std::vector<int64_t> normalized_strides(strides->GetSize(),0);
  NormalizeShape(strides_data, normalized_strides.data(), strides->GetSize(), x_format);
  OP_LOGE_IF(!SetStrides(context, normalized_strides.data(), strides->GetSize()), false, op_name,
             "failed to set strides to tiling param.");

  std::vector<int64_t> normalized_dilations(dilations->GetSize(),0);
  NormalizeShape(dilations_data, normalized_dilations.data(), dilations->GetSize(), x_format);
  OP_LOGE_IF(!SetDilations(normalized_dilations.data(), dilations->GetSize()), false,
              op_name, "failed to set dilations to tiling param.");
  OP_LOGE_IF(!SetPads(context, pads_data, pads->GetSize()), false, op_name, "failed to set pads to tiling param.");

  if (Fp32Input() && GetPrecisionModeIdx() < attrs->GetAttrNum()) {
    const int32_t *precision_mode = attrs->GetAttrPointer<int32_t>(GetPrecisionModeIdx());
    if (precision_mode != nullptr && *precision_mode != -1) {
      // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
      // 0x10: support_of_bound_index  0x20: enable_float_32_execution  0x40: enable_hi_float_32_execution
      hf32_flag = (*precision_mode & 0x40) ? 1 : 0;
    }
  }

  return CalcGroupsParams(op_name);
}

bool CubeTilingParam::SetStrides(const TilingContext *context, const int64_t *strides, size_t dim_num) {
  const auto op_name = context->GetNodeName();
  if (dim_num == kConv2DOriShapeDim) {
    OP_LOGE_IF(strides[kNchwDimN] != 1 || strides[kNchwDimC] != 1, false, op_name,
               "stride_n and stride_c's dim must be 1, current stride_n is %ld, stride_c is %ld.", strides[kNchwDimN],
               strides[kNchwDimC]);
    OP_LOGE_IF(!MathUtil::CheckRange(strides[kNchwDimH], kStrideMin, kAttrDUpper), false,
              op_type, "stride_h is invalid, current is %ld, stride_h support range [%d, %d]",
              strides[kNchwDimH], kStrideMin, kAttrDUpper);
    OP_LOGE_IF(!MathUtil::CheckRange(strides[kNchwDimW], kStrideMin, kAttrDUpper), false,
              op_type, "stride_w is invalid, current is %ld, stride_w support range [%d, %d]",
              strides[kNchwDimW], kStrideMin, kAttrDUpper);
    stride_h = strides[kNchwDimH];
    stride_w = strides[kNchwDimW];
  } else if (dim_num == kConv3DOriShapeDim) {
    OP_LOGE_IF(strides[kNcdhwDimN] != 1 || strides[kNcdhwDimC] != 1, false, op_name,
               "stride_n and stride_c's dim must be 1, current stride_n is %ld, stride_c is %ld.",
               strides[kNcdhwDimN], strides[kNcdhwDimC]);
    OP_LOGE_IF(!MathUtil::CheckRange(strides[kNcdhwDimH], kStrideMin, INT32_MAX), false,
              op_type, "stride_h out of int32 range, current is %ld, int32 range is  [%d, %d]",
              strides[kNcdhwDimH], kStrideMin, INT32_MAX);
    OP_LOGE_IF(!MathUtil::CheckRange(strides[kNcdhwDimW], kStrideMin, INT32_MAX), false,
              op_type, "stride_w out of int32 range, current is %ld, int32 range is [%d, %d]",
              strides[kNcdhwDimW], kStrideMin, INT32_MAX);
    stride_h = strides[kNcdhwDimH];
    stride_w = strides[kNcdhwDimW];
  }

  return true;
}

bool CubeTilingParam::SetPads(const TilingContext *context, const int64_t *pads, size_t dim_num) {
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(dim_num != GetValidPadsDimNum(), false, op_name, "pads of context len is invalid.");
  size_t idx = 0;
  if (GetValidPadsDimNum() == kConv3DPadsDim) {
    idx += 2; // 2: skipt pad_f and pad_b, save them only in 3D param class
  }

  pad_u = pads[idx++];
  pad_d = pads[idx++];
  pad_l = pads[idx++];
  pad_r = pads[idx++];
  OP_LOGE_IF(!MathUtil::CheckRange(pad_u, kPadLower, kPadUpper), false, op_type,
    "pad_u is invalid, current is %d, pad_u support range is [%d, %d]", pad_u, kPadLower, kPadUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(pad_d, kPadLower, kPadUpper), false, op_type,
    "pad_d is invalid, current is %d, pad_d support range is [%d, %d]", pad_d, kPadLower, kPadUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(pad_l, kPadLower, kPadUpper), false, op_type,
    "pad_l is invalid, current is %d, pad_l support range is [%d, %d]", pad_l, kPadLower, kPadUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(pad_r, kPadLower, kPadUpper), false, op_type,
    "pad_r is invalid, current is %d, pad_r support range is [%d, %d]", pad_r, kPadLower, kPadUpper);

  const auto attrs = context->GetAttrs();
  if (attrs->GetAttrNum() <= GetPaddingIdx()) {
    OP_LOGD(op_name, "no padding attr, skip calc and check");
    return true;
  }

  OP_LOGE_IF(stride_h == 0 || stride_w == 0, false, op_name, "stride is 0.");
  Shape &fmap_shape = fmap();
  Shape &y_shape = y();
  const auto padding = attrs->GetAttrPointer<char>(GetPaddingIdx());
  // if pads is invalid and padding is SAME, calc pads
  if (padding != nullptr && padding[0] == 'S' && pad_u == -1 && pad_d == -1 && pad_l == -1 && pad_r == -1) {
    int64_t tails_h = fmap_shape.h % stride_h;
    int64_t tails_w = fmap_shape.w % stride_w;
    int64_t pad_h = std::max((tails_h > 0 ? kernel_h_dilation - tails_h : kernel_h_dilation - stride_h), 0L);
    int64_t pad_w = std::max((tails_w > 0 ? kernel_w_dilation - tails_w : kernel_w_dilation - stride_w), 0L);
    pad_u = pad_h / 2;  // 2 means pad_up is half size of pad_h
    pad_d = pad_h - pad_u;
    pad_l = pad_w / 2;  // 2 means pad_up is half size of pad_w
    pad_r = pad_w - pad_l;
  }

  int64_t ho_expect = (fmap_shape.h + pad_u + pad_d - kernel_h_dilation) / stride_h + 1;
  int64_t wo_expect = (fmap_shape.w + pad_l + pad_r - kernel_w_dilation) / stride_w + 1;
  OP_LOGE_IF(ho_expect != y_shape.h || wo_expect != y_shape.w, false, op_name,
             "check pads attrs failed, ho: %ld, wo: %ld, ho_expect: %ld, wo_expect: %ld", y_shape.h, y_shape.w,
             ho_expect, wo_expect);
  return true;
}

bool CubeTilingParam::SetDilations(const int64_t *dilations, size_t dim_num) {
  if (dim_num == kConv2DOriShapeDim) {
    OP_LOGE_IF(dilations[kNchwDimN] != 1 || dilations[kNchwDimC] != 1, false, op_type,
               "dilation_n and dilation_c's dim must be 1, current dilation_n is %ld, dilation_c is %ld.",
               dilations[kNchwDimN], dilations[kNchwDimC]);
    OP_LOGE_IF(!MathUtil::CheckRange(dilations[kNchwDimH], kStrideMin, kAttrDUpper), false,
              op_type, "dilation_h is invalid, current is %ld, dilation_h support range [%d, %d]",
              dilations[kNchwDimH], kStrideMin, kAttrDUpper);
    OP_LOGE_IF(!MathUtil::CheckRange(dilations[kNchwDimW], kStrideMin, kAttrDUpper), false,
              op_type, "dilation_w is invalid, current is %ld, dilation_w support range [%d, %d]",
              dilations[kNchwDimW], kStrideMin, kAttrDUpper);
    dilation_h = dilations[kNchwDimH];
    dilation_w = dilations[kNchwDimW];
  } else if (dim_num == kConv3DOriShapeDim) {
    OP_LOGE_IF(dilations[kNcdhwDimN] != 1 || dilations[kNcdhwDimC] != 1, false, op_type,
               "dilation_n and dilation_c's dim must be 1, current dilation_n is %ld, dilation_c is %ld.",
               dilations[kNcdhwDimN], dilations[kNcdhwDimC]);
    OP_LOGE_IF(!MathUtil::CheckRange(dilations[kNcdhwDimH], kDilationLower, INT32_MAX), false,
              op_type, "dilation_h out of int32 range, current is %ld, int32 range is [%d, %d]",
              dilations[kNcdhwDimH], kDilationLower, INT32_MAX);
    OP_LOGE_IF(!MathUtil::CheckRange(dilations[kNcdhwDimW], kDilationLower, INT32_MAX), false,
              op_type, "dilation_w out of int32 range, current is %ld, int32 range is [%d, %d]",
              dilations[kNcdhwDimW], kDilationLower, INT32_MAX);
    dilation_h = dilations[kNcdhwDimH];
    dilation_w = dilations[kNcdhwDimW];
  }

  kernel_h_dilation = (kernel_h - 1) * dilation_h + 1;
  kernel_w_dilation = (kernel_w - 1) * dilation_w + 1;
  return true;
}

bool CubeTilingParam::CalcGroupsParams(const char *op_name) {
  Shape &fmap_shape = fmap();
  Shape &filter_shape = filter();
  Shape &y_shape = y();

  if (filter_shape.c == 0 || static_cast<int64_t>(fmap_shape.c) % filter_shape.c != 0) {
    OP_LOGE(op_name, "fmap_channel(%ld) %% filter_channel(%ld) != 0", fmap_shape.c, filter_shape.c);
    return false;
  }

  int64_t actual_groups = static_cast<int64_t>(fmap_shape.c) / filter_shape.c;
  if (groups == 1) {
    groups = static_cast<int32_t>(actual_groups);
    OP_LOGD(op_name, "set groups=%d, fmap_channel(%ld) / filter_channel(%ld)", groups, fmap_shape.c, filter_shape.c);
  } else if (actual_groups != static_cast<int64_t>(groups)) {
    OP_LOGE(op_name, "fmap_channel(%ld) / filter_channel(%ld) != groups(%d)", fmap_shape.c, filter_shape.c, groups);
    return false;
  }

  if (y_shape.c % groups != 0) {
    OP_LOGE(op_name, "out_channels(%ld) %% groups(%d) != 0", y_shape.c, groups);
    return false;
  }

  if (groups == 1) {
    real_g = 1;
  } else {
    int64_t mag_factor0 = Lcm(fmap_shape.c / groups, k0) / (fmap_shape.c / groups);
    int64_t mag_factor1 = Lcm(y_shape.c / groups, y_shape.c0) / (y_shape.c / groups);
    mag_factor = MathUtil::Min(Lcm(mag_factor0, mag_factor1), groups);
    int32_t cin1_g = (mag_factor * fmap_shape.c / groups + fmap_shape.c0 - 1) / fmap_shape.c0;
    int32_t cout1_g = (mag_factor * y_shape.c / groups + y_shape.c0 - 1) / y_shape.c0;
    real_g = (groups + mag_factor - 1) / mag_factor;
    y_shape.c1 = cout1_g;
    fmap_shape.c1 = cin1_g;
    OP_LOGD(op_name, "cin1_g: %d, cout1_g: %d, real_g: %d", cin1_g, cout1_g, real_g);
  }

  OP_LOGE_IF(groups < 1 || groups > UINT16_MAX, false, op_name,
             "Groups [%d] is invalid, it should be in range: [1, %d]", groups, UINT16_MAX);
  return true;
}

bool CubeTilingParam::IsValid() const {
  const std::vector<Range> *range = nullptr;
  for (const auto &op_range : kOpRange) {
    if (op_range.first == type) {
      range = &op_range.second;
      break;
    }
  }

  if (range == nullptr) {
    return true;
  }

  for (const auto &item : *range) {
    int32_t offset = item.offset;
    int32_t val = (*(reinterpret_cast<const int32_t *>((reinterpret_cast<const int8_t *>(this) + offset))));
    if (!MathUtil::CheckRange(val, item.lower, item.upper)) {
      OP_LOGE(op_type, "invalid param [%s] with value [%d], valid range is[%d, %d]", item.name, val, item.lower,
              item.upper);
      return false;
    }
  }

  return CheckGEMMLimit() && IsExtraInfoValid();
}

void CubeTilingParam::UpdateInstrictionParam() {
  static bool flag = false;
  if (flag) {
    return;
  }

  if (platform_info.support_l0c2out()) {
    Load3dInstrictionParam &load3d_inst_param = InstructionParam::Instance().get_load3d_inst_param();
    load3d_inst_param.SetKernelRange(1, kKernelMaxV220);
  }
  flag = true;
}

std::string CubeTilingParam::ToString() const {
  std::stringstream ss;
  ss << "op_type: " << op_type
     << " platform_info: " << platform_info.ToString()
     << " a_shape: " << a_shape.ToString()
     << " b_shape: " << b_shape.ToString()
     << " c_shape: " << c_shape.ToString()
     << " bias_flag: " << bias_flag
     << " groups: " << groups
     << " k0: " << k0
     << " stride_h: " << stride_h
     << " stride_w: " << stride_w
     << " kernel_h: " << kernel_h
     << " kernel_w: " << kernel_w
     << " dilation_h: " << dilation_h
     << " dilation_w: " << dilation_w
     << " pad_u: " << pad_u
     << " pad_d: " << pad_d
     << " pad_l: " << pad_l
     << " pad_r: " << pad_r
     << " aub_fused_num: " << aub_fused_num
     << " bub_fused_num: " << bub_fused_num
     << " cub_fused_num: " << cub_fused_num
     << " a_dtype_bytes: " << a_dtype_bytes
     << " b_dtype_bytes: " << b_dtype_bytes
     << " c_dtype_bytes: " << c_dtype_bytes
     << " binary_mode: " << binary_mode
     << " load3d_special: " << load3d_special
     << " conv1d_flag: " << conv1d_flag
     << " hf32_flag: " << hf32_flag
     << " split_w_flag: " << split_w_flag
     << " load2d_flag: " << load2d_flag
     << " dma_flag: " << dma_flag
     << " strideh_read_flag: " << strideh_read_flag
     << " linear_embedding_opti_flag: " << linear_embedding_opti_flag
     << " load3d_flag: " << load3d_flag;

  return ss.str();
}

bool Conv2DBpFilterTilingParam::CheckFormat(const TilingContext *context) const {
  const auto op_name = context->GetNodeName();
  const auto fmap_desc = GetFmapTensorDesc(context);
  const auto dedy_desc = GetYTensorDesc(context);
  const auto filter_desc = GetFilterTensorDesc(context);
  auto filter_ori_format = filter_desc->GetOriginFormat();
  auto fmap_ori_format = fmap_desc->GetOriginFormat();
  auto dedy_ori_format = dedy_desc->GetOriginFormat();

  auto dedy_format = static_cast<ge::Format>(ge::GetPrimaryFormat(dedy_desc->GetStorageFormat()));
  auto fmap_format = static_cast<ge::Format>(ge::GetPrimaryFormat(fmap_desc->GetStorageFormat()));

  if (binary_mode == kBinaryModeNCHW) {
    OP_LOGE_IF(dedy_ori_format != ge::FORMAT_NCHW || dedy_format != ge::FORMAT_NCHW, false, op_name,
               "out_backprop format should be NCHW.");
    OP_LOGE_IF(fmap_ori_format != ge::FORMAT_NCHW || fmap_format != ge::FORMAT_NCHW, false, op_name,
               "fmap format should be NCHW.");
  } else if (binary_mode == kBinaryModeNHWC) {
    OP_LOGE_IF(dedy_ori_format != ge::FORMAT_NHWC, false, op_name,
               "out_backprop ori format should be NHWC.");
    OP_LOGE_IF(fmap_ori_format != ge::FORMAT_NHWC, false, op_name,
               "fmap ori format should be NHWC.");
  } else {
    OP_LOGE_IF(dedy_ori_format != ge::FORMAT_NCHW && dedy_ori_format != ge::FORMAT_NHWC, false, op_name,
               "out_backprop ori format should be NCHW/NHWC.");
    OP_LOGE_IF(fmap_ori_format != ge::FORMAT_NCHW && fmap_ori_format != ge::FORMAT_NHWC, false, op_name,
               "fmap ori format should be NCHW/NHWC.");
  }
  OP_LOGE_IF(dedy_ori_format != fmap_ori_format, false, op_name, "out_backprop ori format should be same with fmap.");
  OP_LOGE_IF(filter_ori_format != ge::FORMAT_NCHW && filter_ori_format != ge::FORMAT_NHWC &&
                 filter_ori_format != ge::FORMAT_HWCN,
             false, op_name, "filter ori format should be NCHW/NHWC/HWCN.");
  return true;
}

void Conv2DBpFilterTilingParam::SetDtype(ge::DataType fmap_dtype, ge::DataType filter_dtype, ge::DataType y_dtype) {
  a_dtype = y_dtype;
  b_dtype = fmap_dtype;
  c_dtype = filter_dtype;
  if (fmap_dtype == ge::DT_FLOAT) {
    k0 = kFp32BlockReduce;
  }
}

void Conv2DBpFilterTilingParam::SetC0(const TilingContext *context) {
  auto filter_desc = GetFilterTensorDesc(context);
  auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
  c04_flag = filter_format == ge::FORMAT_FRACTAL_Z_C04;

  if (c04_flag && fmap().c <= kSmallChannelSize) {
    fmap().c0 = kSmallChannelSize;
    fmap().c1 = MathUtil::CeilDivision(fmap().c, fmap().c0);
  }
}

bool Conv2DBpFilterTilingParam::IsLoad3dValid() const {
  Load3dParam load3d_param;
  load3d_param.pad_u = pad_u;
  load3d_param.pad_d = pad_d;
  load3d_param.pad_l = pad_l;
  load3d_param.pad_r = pad_r;
  load3d_param.kernel_h = kernel_h;
  load3d_param.kernel_w = kernel_w;
  load3d_param.stride_h = stride_h;
  load3d_param.stride_w = load3d_special == kLoad3dSpecial ? b_shape.w + pad_l + pad_r : stride_w;
  load3d_param.dilation_h = dilation_h;
  load3d_param.dilation_w = dilation_w;

  Load3dInstrictionParam &load3d_inst_param = InstructionParam::Instance().get_load3d_inst_param();
  return load3d_inst_param.IsValid(load3d_param);
}

void Conv2DBpFilterTilingParam::SetSpecialSceneParams() {
  conv1d_flag = IsConv1d();
  load2d_flag = IsLoad2dMode();
  split_w_flag = IsLoad3dWSplitMode();
  dma_flag = IsL0bDmaCopyMode();
  linear_embedding_opti_flag = IsLinearEmbeddingOptiMode();
  SetStrideHReadFlag();
  load3d_special = IsLoad3dSpecial();
  load3d_flag = IsUseLoad3dFlag();
}

bool Conv2DBpFilterTilingParam::CheckGEMMLimit() const {
  int64_t total_k = MathUtil::Align(a_shape.h * a_shape.w, kBlockSize);
  int64_t total_n = MathUtil::Align(b_shape.c1, kBlockSize) * kernel_h * kernel_w;
  OP_LOGE_IF(total_k > std::numeric_limits<int32_t>::max(), false, op_type,
             "get matrix multiply K (Hout * Wout) axis[%ld] exceed int32 limit.", total_k);
  OP_LOGE_IF(total_n > std::numeric_limits<int32_t>::max(), false, op_type,
             "get matrix multiply N (Cin * Kernel_h * Kenerl_w) axis[%ld] exceed int32 limit.", total_n);
  return true;
}

bool Conv2DBpFilterTilingParam::IsExtraInfoValid() const {
  // check load3d instrction constraint, 2 meas true
  OP_LOGE_IF((load3d_special == kLoad3dSpecial && (b_shape.w + pad_l + pad_r) > kStrideUpper), false, op_type,
             "fmap_w after pad exceed for load3d constraint.");
  OP_LOGE_IF(kernel_h_dilation > b_shape.h + pad_u + pad_d, false, op_type,
             "hi + pad_up + pad_down = [%ld] should greater than (kh - 1) * dilation_h + 1 = [%d]",
              b_shape.h + pad_u + pad_d, kernel_h_dilation);
  OP_LOGE_IF(kernel_w_dilation > b_shape.w + pad_l + pad_r, false, op_type,
             "wi + pad_left + pad_right = [%ld] should greater than (kw - 1) * dilation_w + 1 = [%d]",
              b_shape.w + pad_l + pad_r, kernel_w_dilation);
  OP_LOGE_IF((b_shape.h - kernel_h_dilation + pad_u + pad_d) / stride_h + 1 != a_shape.h, false, op_type,
             "ho is not match with hi.");
  OP_LOGE_IF((b_shape.w - kernel_w_dilation + pad_l + pad_r) / stride_w + 1 != a_shape.w, false, op_type,
             "wo is not match with wi.");

  // dwV2 C04 case. currently dwV2 only supports this case on 1971
  bool c04_white_list_case = type == OpType::kConv2DBackpropFilterV2 && a_shape.batch == 1024 && a_shape.c == 1024 &&
                             a_shape.h == 14 && a_shape.w == 14 && b_shape.batch == 1024 && b_shape.c == 3 &&
                             b_shape.h == 224 && b_shape.w == 224;
  if (!static_flag) {
    if (c04_flag && !c04_white_list_case) {
      OP_LOGE(op_type, "C04 is not supported now.");
      return false;
    }
  }

  if (dma_flag) {
    return true;
  }

  // check l1 size
  OP_LOGE_IF(!CheckMinL1Size(), false, op_type, "L1 min tiling excced L1 size");
  return true;
}

bool Conv2DBpFilterTilingParam::CheckMinL1Size() const {
  int32_t temp_stride_h = 0;
  if (IsStrideHRead()) {
    temp_stride_h = kernel_h;
  } else {
    temp_stride_h = stride_h;
  }
  int32_t min_k_l0 = 1;
  if (b_dtype == ge::DT_FLOAT) {
    // in fp32 scene, k axis aligned to 8 in L0 but aligned to 16 in L1, so k_l0 must be even number, min value is 2
    min_k_l0 = 2;
  }

  int32_t al1_min_size = a_shape.c0 * k0 * a_dtype_bytes * min_k_l0;
  int32_t kl1_min =
      conv1d_flag || split_w_flag ? CubeUtil::CalcWi(min_k_l0 * k0, stride_w, kernel_w_dilation, b_shape.w) : b_shape.w;
  int32_t bl1_min_size = 0;
  if (a_shape.w >= b_shape.c0) {
    if (a_shape.w % b_shape.c0 == 0) {
      bl1_min_size = kernel_h_dilation * kl1_min * min_k_l0 * k0 * b_dtype_bytes;
    } else {
      bl1_min_size = (kernel_h_dilation + temp_stride_h) * kl1_min * min_k_l0 * k0 * b_dtype_bytes;
    }
  } else {
    int32_t bl1_align_factor = MathUtil::CeilDivision(b_shape.c0, a_shape.w);
    // for example:
    // |  Wo  |  3   |  5   |  6   |  7   |  9   |  10  |  11  |  12  |  13  |  14  |  15  |
    // | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
    // |  Ho  |  6   |  4   |  4   |  4   |  3   |  3   |  3   |  2   |  3   |  2   |  2   |
    if (b_shape.c0 % a_shape.w != 0) {
      ++bl1_align_factor;
    }
    bl1_min_size = MathUtil::Align((kernel_h_dilation + (bl1_align_factor - 1) * temp_stride_h) * kl1_min, b_shape.c0) *
                   min_k_l0 * k0 * b_dtype_bytes;
  }
  if (split_w_flag) {
    bl1_min_size = kernel_h_dilation * kl1_min * min_k_l0 * k0 * b_dtype_bytes;
  }

  OP_LOGD(op_type, "al1:%d, bl1:%d, load3d_special: %d, conv1d_flag: %d, split_w_flag: %d", al1_min_size, bl1_min_size,
          load3d_special, conv1d_flag, split_w_flag);
  if (!split_w_flag && CanEnableSplitW()) {
    OP_LOGI_IF_RETURN(static_cast<uint64_t>(al1_min_size + bl1_min_size) > platform_info.l1_size(), false, op_type,
                      "L1 min tiling excced L1 size. al1:%d, bl1:%d, will enable split_w switch and check again",
                      al1_min_size, bl1_min_size);
  } else {
  }
  return static_cast<uint64_t>(al1_min_size + bl1_min_size) <= platform_info.l1_size();
}

bool Conv2DBpFilterTilingParam::CanEnableSplitW() const {
  return load3d_special == 1 && !conv1d_flag && !c04_flag && IsLoad3dValid();
}

bool Conv2DBpFilterTilingParam::IsLoad2dMode() const {
  return (b_dtype != ge::DT_FLOAT && !conv1d_flag && strideh_read_flag == 0 &&
          kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 &&
          pad_u == 0 && pad_d == 0 && pad_l == 0 && pad_r == 0);
}

bool Conv2DBpFilterTilingParam::IsConv1d() {
  conv1d_flag = true;
  if (!IsLoad3dValid() || !CheckMinL1Size()) {
    return false;
  }

  if ((b_shape.h + pad_u + pad_d) == 1 && kernel_h == 1 && ((b_shape.w + pad_l + pad_r) != 1 || kernel_w != 1)) {
    stride_h = 1;
    return true;
  }
  return false;
}

bool Conv2DBpFilterTilingParam::IsLoad3dWSplitMode() {
  if (platform_info.load3d_constraints() && a_shape.h != 1 && a_shape.w == 1) {
    return false;
  }
  if (CanEnableSplitW() && !CheckMinL1Size()) {
    // will check min L1 size with split_w_flag in Conv2DBpFilterTilingParam::IsExtraInfoValid
    OP_LOGI(op_type, "L1 min tiling excced L1 size, will enable split_w switch and check again");
    split_w_flag = true;
    load2d_flag = false;
    if (!CheckMinL1Size()) {
      return false;
    }
    return true;
  }

  return false;
}

bool Conv2DBpFilterTilingParam::IsL0bDmaCopyMode() {
  if (load2d_flag || split_w_flag) {
    return false;
  }
  if (!CheckMinL1Size()) {
    return true;
  }
  if (!IsLoad3dValid()) {
    return true;
  }

  auto is_special_case = platform_info.load3d_constraints() && (b_shape.w + pad_l + pad_r) > kStrideUpper
                          && a_shape.h != 1 && a_shape.w == 1;
  if (is_special_case) {
    return true;
  }

  return false;
}

bool Conv2DBpFilterTilingParam::IsUseLoad3dFlag() const {
  return !load2d_flag && !dma_flag;
}

bool Conv2DBpFilterTilingParam::IsStrideHRead() const{
  if (c04_flag || load2d_flag || conv1d_flag) {
    return false;
  }
  return ((pad_u == 0 && pad_d == 0) && (stride_h > kernel_h && kernel_h == 1));
}
void Conv2DBpFilterTilingParam::SetStrideHReadFlag() {
  strideh_read_flag = 0;
  if (dma_flag) {
    return;
  }

  if (IsStrideHRead() &&
      (binary_mode == kBinaryModeNC1HWC0 ||
       (platform_info.support_l0c2out() && binary_mode != kBinaryModeNC1HWC0))) {
    sr_fmap_h = b_shape.h;
    sr_stride_h = stride_h;
    b_shape.h = (b_shape.h - kernel_h) / stride_h + 1;
    stride_h = kernel_h;
    strideh_read_flag = 1;
  }
  return;
}

int32_t Conv2DBpFilterTilingParam::IsLoad3dSpecial() const {
  if (!load2d_flag && !dma_flag && !split_w_flag && platform_info.load3d_constraints() && a_shape.h != 1 &&
      a_shape.w == 1) {
    return kLoad3dSpecial; // load3d_special scene is enlarged by 2 times
  }
  return 1;
}

bool Conv2DBpFilterTilingParam::IsLinearEmbeddingOptiMode() {
  bool case_conditon =
      (dilation_h == 1 && dilation_w == 1 && pad_u == 0 && pad_d == 0 &&
       pad_l == 0 && pad_r == 0 && kernel_h == stride_h &&
       kernel_w == stride_w && b_shape.h == a_shape.h * stride_h &&
       b_shape.w == a_shape.w * stride_w);
  bool case_white_list = (kernel_h == 32 && kernel_w == 32 && b_shape.h == 224 && b_shape.w == 224 && b_shape.c == 3);
  if (split_w_flag && case_conditon && platform_info.support_l0c2out() && case_white_list) {
    split_w_flag = false;
    load2d_flag = false;
    dma_flag = true;
    return true;
  }
  return false;
}

std::string Conv2DBpInputTilingParam::ToString() const {
  std::stringstream ss;
  ss << CubeTilingParam::ToString()
     << " filter_h_dilation: " << filter_h_dilation
     << " filter_w_dilation: " << filter_w_dilation
     << " stride_expand_flag: " << stride_expand_flag
     << " dx_no_overlap_condition_1: " << dx_no_overlap_condition_1
     << " dx_no_overlap_condition_2: " << dx_no_overlap_condition_2
     << " g_extend: " << g_extend
     << " co1g: " << co1g
     << " ci1g: " << ci1g
     << " filter_co0: " << filter_co0
     << " filter_ci0: " << filter_ci0
     << " co1g_reduce: " << co1g_reduce
     << " split_axis_mode: " << split_axis_mode;
  return ss.str();
}

void Conv2DBpInputTilingParam::StrideOptimize() {
  if (a_shape.h == 1 && c_shape.h == 1) {
    stride_h = 1;
  }
  if (a_shape.w == 1 && c_shape.w == 1) {
    stride_w = 1;
  }
}

std::string Conv3DCommonTilingParam::ToString() const {
  std::stringstream ss;
  ss << " kernel_d: " << kernel_d
     << " stride_d: " << stride_d
     << " dilation_d: " << dilation_d
     << " pad_f: " << pad_f
     << " pad_b: " << pad_b
     << " pad_greater_than_filter: " << pad_greater_than_filter;

  return ss.str();
}

std::string Conv3DBpInputTilingParam::ToString() const {
  std::stringstream ss;
  ss << CubeTilingParam::ToString()
     << " stride_d: " << stride_d
     << " dilation_d: " << dilation_d
     << " filter_d_dilation: " << filter_d_dilation
     << " filter_h_dilation: " << filter_h_dilation
     << " filter_w_dilation: " << filter_w_dilation
     << " pad_h: " << pad_h
     << " pad_t: " << pad_t
     << " backprop_pad_h: " << backprop_pad_h
     << " backprop_pad_t: " << backprop_pad_t
     << " backprop_pad_u: " << backprop_pad_u
     << " backprop_pad_d: " << backprop_pad_d
     << " backprop_pad_l: " << backprop_pad_l
     << " backprop_pad_r: " << backprop_pad_r
     << " co1g: " << co1g
     << " ci1g: " << ci1g
     << " filter_co0: " << filter_co0
     << " filter_ci0: " << filter_ci0
     << " co1g_reduce: " << co1g_reduce
     << " load3d_special: " << load3d_special;

  return ss.str();
}

bool Conv3DCommonTilingParam::CheckConv3DFormat(const TilingContext *context, const CubeTilingParam *params) const {
  const auto op_name = context->GetNodeName();
  auto fmap_ori_format = params->GetFmapTensorDesc(context)->GetOriginFormat();
  auto filter_ori_format = params->GetFilterTensorDesc(context)->GetOriginFormat();
  auto y_ori_format = params->GetYTensorDesc(context)->GetOriginFormat();
  OP_LOGE_IF(fmap_ori_format != y_ori_format, false, op_name, "y ori format should be same with fmap.");
  OP_LOGE_IF(fmap_ori_format != ge::FORMAT_NDHWC && fmap_ori_format != ge::FORMAT_NCDHW, false, op_name,
             "fmap ori format should be NDHWC or NCDHW.");
  OP_LOGE_IF(filter_ori_format != ge::FORMAT_NDHWC && filter_ori_format != ge::FORMAT_NCDHW &&
                 filter_ori_format != ge::FORMAT_DHWCN,
             false, op_name, "filter ori format should be NDHWC/NCDHW/DHWCN.");

  return true;
}

bool Conv3DCommonTilingParam::SetConv3DPads(const TilingContext *context, const int64_t *pads, size_t dim_num,
                                            CubeTilingParam *params) {
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(!params->CubeTilingParam::SetPads(context, pads, dim_num), false, op_name, "failed to set pads.");

  pad_f = pads[0];
  pad_b = pads[1];
  pad_greater_than_filter = pad_f >= kernel_d_dilation || pad_b >= kernel_d_dilation;

  const auto attrs = context->GetAttrs();
  if (attrs->GetAttrNum() <= params->GetPaddingIdx()) {
    OP_LOGD(op_name, "no padding attr, skip calc and check");
    return true;
  }

  OP_LOGE_IF(stride_d == 0, false, op_name, "stride is 0.");
  Shape &fmap_shape = params->fmap();
  Shape &y_shape = params->y();
  const auto padding = attrs->GetAttrPointer<char>(params->GetPaddingIdx());
  if (padding != nullptr && padding[0] == 'S' && pad_f == -1 && pad_b == -1) {
    int64_t tails_d = fmap_shape.d % stride_d;
    int64_t pad_d = std::max((tails_d > 0 ? kernel_d_dilation - tails_d : kernel_d_dilation - stride_d), 0L);
    pad_f = pad_d / 2;  // 2 means pad_up is half size of pad_h
    pad_b = pad_d - pad_f;
  } else {
    OP_LOGE_IF(!MathUtil::CheckRange(pads[0], kPadMin, kAttrDUpper), false, op_name,
             "pad_f is invalid, current is %ld, pad_f support range [%d, %d]", pads[0], kPadMin, kAttrDUpper);
    OP_LOGE_IF(!MathUtil::CheckRange(pads[1], kPadMin, kAttrDUpper), false, op_name,
             "pad_b is invalid, current is %ld, pad_b support range [%d, %d]", pads[1], kPadMin, kAttrDUpper);
  }

  int64_t do_expect = (fmap_shape.d + pad_f + pad_b - kernel_d_dilation) / stride_d + 1;
  OP_LOGE_IF(do_expect != y_shape.d, false, op_name, "check pads attrs failed, do: %ld, do_expect: %ld", y_shape.d,
             do_expect);

  pad_greater_than_filter = pad_f >= kernel_d_dilation || pad_b >= kernel_d_dilation;

  return true;
}

bool Conv3DTilingParam::SetStrides(const TilingContext *context, const int64_t *strides, size_t dim_num) {
  OP_LOGE_IF(!CubeTilingParam::SetStrides(context, strides, dim_num), false, context->GetNodeName(),
             "failed to set strides.");
  OP_LOGE_IF(!MathUtil::CheckRange(strides[kNcdhwDimH], kStrideMin, kStrideUpper), false, context->GetNodeName(),
             "stride_h is invalid, current is %ld, stride_h support range [%d, %d]",
             strides[kNcdhwDimH], kStrideMin, kStrideUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(strides[kNcdhwDimW], kStrideMin, kStrideUpper), false, context->GetNodeName(),
             "stride_w is invalid, current is %ld, stride_w support range [%d, %d]",
             strides[kNcdhwDimW], kStrideMin, kStrideUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(strides[kNcdhwDimD], kStrideMin, kAttrDUpper), false, context->GetNodeName(),
             "stride_d is invalid, current is %ld, stride_d support range [%d, %d]",
             strides[kNcdhwDimD], kStrideMin, kAttrDUpper);
  stride_d = strides[kNcdhwDimD];
  return true;
}

bool Conv3DTilingParam::SetDilations(const int64_t *dilations, size_t dim_num) {
  OP_LOGE_IF(!CubeTilingParam::SetDilations(dilations, dim_num), false, op_type,
            "failed to set dilations.");
  OP_LOGE_IF(!MathUtil::CheckRange(dilations[kNcdhwDimH], kDilationLower, kDilationUpper), false,
              op_type, "dilation_h is invalid, current is %ld, dilation_h support range [%d, %d]",
              dilations[kNcdhwDimH], kDilationLower, kDilationUpper);
    OP_LOGE_IF(!MathUtil::CheckRange(dilations[kNcdhwDimW], kDilationLower, kDilationUpper), false,
              op_type, "dilation_w is invalid, current is %ld, dilation_w support range [%d, %d]",
              dilations[kNcdhwDimW], kDilationLower, kDilationUpper);
  kernel_d = b_shape.d;
  OP_LOGE_IF(!MathUtil::CheckRange(dilations[kNcdhwDimD], kDilationMin, kAttrDUpper), false, op_type,
             "dilation_d is invalid, current is %ld, dilation_d support range [%d, %d]",
             dilations[kNcdhwDimD], kDilationMin, kAttrDUpper);
  dilation_d = dilations[kNcdhwDimD];
  kernel_d_dilation = (kernel_d - 1) * dilation_d + 1;
  return true;
}

void Conv3DTilingParam::SetDtype(ge::DataType fmap_dtype, ge::DataType filter_dtype, ge::DataType y_dtype) {
  SetDtypeWithBias(fmap_dtype, filter_dtype, y_dtype, y_dtype);
}

void Conv3DTilingParam::SetDtypeWithBias(ge::DataType fmap_dtype, ge::DataType filter_dtype, ge::DataType y_dtype,
                                         ge::DataType in_bias_dtype) {
  a_dtype = fmap_dtype;
  b_dtype = filter_dtype;
  c_dtype = y_dtype;
  bias_dtype = in_bias_dtype;
  bias_dtype_bytes = ge::GetSizeByDataType(bias_dtype);
  if (fmap_dtype == ge::DT_FLOAT) {
    k0 = kFp32BlockReduce;
  }
}

bool Conv3DTilingParam::CalcGroupsParams(const char *op_name) {
  OP_LOGE_IF(!CubeTilingParam::CalcGroupsParams(op_name), false, op_name,
             "failed to calc group params according to base logic.");

  a_shape.c1 = MathUtil::CeilDivision(mag_factor * a_shape.c / groups, static_cast<int64_t>(k0));
  OP_LOGD(op_name, "correct cin1_g to %ld", a_shape.c1);
  return true;
}

std::string Conv3DTilingParam::ToString() const {
  std::stringstream ss;
  ss << CubeTilingParam::ToString() << Conv3DCommonTilingParam::ToString()
     << " bias_dtype: " << bias_dtype
     << " bias_dtype_bytes: " << bias_dtype_bytes;

  return ss.str();
}

int64_t Conv3DTilingParam::CalcMinAL1Size() const {
  int32_t point_per_w = ((a_shape.w - kernel_w_dilation) + pad_l + pad_r) / stride_w + 1;
  int32_t w_in = kBlockSize / point_per_w + 2;
  int64_t al1_min_m = static_cast<int64_t>((w_in - 1) * stride_h + kernel_h_dilation) * a_shape.w;
  return al1_min_m * k0 * a_dtype_bytes;
}

int64_t Conv3DTilingParam::CalcMinBL1Size() const {
  return static_cast<int64_t>(k0) * kBlockSize * kernel_h * kernel_w * b_dtype_bytes;
}

int64_t Conv3DTilingParam::CalcMinBiasL1Size() const {
  if (bias_flag == 0 || !platform_info.support_l0c2out()) {
    return 0;
  }

  return static_cast<int64_t>(kBlockSize) * bias_dtype_bytes;
}

bool Conv3DTilingParam::CheckCanLoadBL1() const {
  int64_t al1_min_size = CalcMinAL1Size();
  int64_t bl1_min_size = CalcMinBL1Size();
  int64_t bias_l1_min_size = CalcMinBiasL1Size();
  return al1_min_size + bl1_min_size + bias_l1_min_size <= static_cast<int64_t>(platform_info.l1_size());
}

bool Conv3DTilingParam::CheckGEMMLimit() const {
  int64_t total_k = MathUtil::Align(a_shape.c1, kBlockSize) * kernel_h * kernel_w * kernel_d;
  int64_t total_m = MathUtil::Align(c_shape.h * c_shape.w, kBlockSize);
  OP_LOGE_IF(total_k > std::numeric_limits<int32_t>::max(), false, op_type,
             "get matrix multiply K (Cin * Kernel_h * Kernel_w * Kenerl_d) axis[%ld] exceed int32 limit.", total_k);
  OP_LOGE_IF(total_m > std::numeric_limits<int32_t>::max(), false, op_type,
             "get matrix multiply M (Hout * Wout) axis[%ld] exceed int32 limit.", total_m);

  return true;
}

bool Conv3DTilingParam::IsExtraInfoValid() const {
  if (!platform_info.support_l0c2out() && a_shape.w == 1) {
    OP_LOGE_IF(!MathUtil::CheckRange(pad_r + stride_w, 1, kPadMax), false, op_type, "pad_r + stride_w is invalid");
  }

  OP_LOGE_IF(!platform_info.support_l0c2out() && a_dtype == ge::DT_BF16, false, op_type, "not support bf16.");
  OP_LOGE_IF(bias_flag == 1 && a_dtype == ge::DT_BF16 && bias_dtype != ge::DT_FLOAT, false, op_type,
             "bias only support fp32 in bf16 scene, but get %s.",
             ge::TypeUtils::DataTypeToAscendString(bias_dtype).GetString());
  OP_LOGE_IF(!MathUtil::CheckRange(pad_f, kPadMin, kAttrDUpper), false, op_type,
             "pad_f is invalid, current is %d, pad_f support range [%d, %d]", pad_f, kPadMin, kAttrDUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(pad_b, kPadMin, kAttrDUpper), false, op_type,
            "pad_b is invalid, current is %d, pad_b support range [%d, %d]", pad_b, kPadMin, kAttrDUpper);

  OP_LOGE_IF(!MathUtil::CheckRange(b_shape.h, kKernelMin, kKernelMaxV220), false, op_type,
             "filter_h is invalid, current is %ld, filter_h support range [%d, %d]", b_shape.h, kKernelMin, kKernelMaxV220);
  OP_LOGE_IF(!MathUtil::CheckRange(b_shape.w, kKernelMin, kKernelMaxV220), false, op_type,
             "filter_w is invalid, current is %ld, filter_w support range [%d, %d]", b_shape.w, kKernelMin, kKernelMaxV220);
  OP_LOGE_IF(kernel_d_dilation > a_shape.d + pad_f + pad_b, false, op_type, "kd value is invalid.");
  OP_LOGE_IF(kernel_h_dilation > a_shape.h + pad_u + pad_d, false, op_type, "kh value is invalid.");
  OP_LOGE_IF(kernel_w_dilation > a_shape.w + pad_l + pad_r, false, op_type, "kw value is invalid.");

  OP_LOGE_IF(!IsLoad3dValid(), false, op_type, "invalid load3d param, please check kernel/pads/dilation/stride args.");

  int64_t al1_min_size = CalcMinAL1Size();
  int64_t bias_l1_min_size = CalcMinBiasL1Size();
  OP_LOGE_IF(static_cast<uint64_t>(al1_min_size + bias_l1_min_size) > platform_info.l1_size(), false, op_type,
             "L1 min tiling excced L1 size. min al1:%ld, min bias l1: %ld", al1_min_size, bias_l1_min_size);

  if (bias_flag == 1 && platform_info.support_l0c2out()) {
    int64_t bl1_min_size = CalcMinBL1Size();
    // if kernel can't load into BL1, then bias will full load into L1
    // when pad_greater_than_filter, kernel load into L0B directly, then bias also full load into L1
    if (al1_min_size + bl1_min_size + bias_l1_min_size > static_cast<int64_t>(platform_info.l1_size()) ||
        pad_greater_than_filter) {
      int64_t single_core_n = MathUtil::CeilDivision(c_shape.c1, static_cast<int64_t>(platform_info.core_num()));
      if (c_dtype == ge::DT_FLOAT16) {
        // if move_l1_to_bt cast fp16 to fp32, BT align to 128B, L1 align to 64B, then n_l0 align to 2
        // here n_bl1 is 1, single_core_n is equal with n_l0
        single_core_n = MathUtil::CeilDivision(single_core_n, 2L) * 2L;
      }

      bias_l1_min_size = single_core_n * kBlockSize * bias_dtype_bytes;
      OP_LOGE_IF(static_cast<uint64_t>(al1_min_size + bias_l1_min_size) > platform_info.l1_size(), false, op_type,
                 "L1 min tiling excced L1 size. min al1:%ld, min bias l1(full load): %ld", al1_min_size,
                 bias_l1_min_size);
    }
  }
  return true;
}

bool Conv3DTilingParam::IsLoad3dValid() const {
  if (platform_info.support_l0c2out()) {
    int32_t min_pos_k = kernel_h * kernel_w * k0;
    OP_LOGE_IF(min_pos_k > kMaxLoad3dV2Kstart, false, op_type, "min posK[%d] excced limit[65535].", min_pos_k);
  }

  Load3dParam load3d_param;
  load3d_param.pad_u = pad_u;
  load3d_param.pad_d = pad_d;
  load3d_param.pad_l = pad_l;
  load3d_param.pad_r = pad_r;
  load3d_param.kernel_h = kernel_h;
  load3d_param.kernel_w = kernel_w;
  load3d_param.stride_h = stride_h;
  load3d_param.stride_w = stride_w;
  load3d_param.dilation_h = dilation_h;
  load3d_param.dilation_w = dilation_w;

  Load3dInstrictionParam &load3d_inst_param = InstructionParam::Instance().get_load3d_inst_param();
  return load3d_inst_param.IsValid(load3d_param);
}

bool Conv3DBpFilterTilingParam::SetStrides(const TilingContext *context, const int64_t *strides, size_t dim_num) {
  OP_LOGE_IF(!CubeTilingParam::SetStrides(context, strides, dim_num), false, context->GetNodeName(),
             "failed to set strides.");
  stride_d = strides[kNcdhwDimD];
  return true;
}

bool Conv3DBpFilterTilingParam::SetDilations(const int64_t *dilations, size_t dim_num) {
  OP_LOGE_IF(!CubeTilingParam::SetDilations(dilations, dim_num), false, op_type,
            "failed to set dilations.");
  kernel_d = c_shape.d;
  dilation_d = dilations[kNcdhwDimD];
  kernel_d_dilation = (kernel_d - 1) * dilation_d + 1;
  return true;
}

void Conv3DBpFilterTilingParam::SetSpecialSceneParams() {
  if (platform_info.load3d_constraints() && a_shape.h != 1 && a_shape.w == 1) {
    load3d_special = kLoad3dSpecial;
  }

  split_w_flag = Conv2DBpFilterTilingParam::IsLoad3dWSplitMode();
  dma_flag = Conv2DBpFilterTilingParam::IsL0bDmaCopyMode();
  load3d_flag = true;
}

std::string Conv3DBpFilterTilingParam::ToString() const {
  std::stringstream ss;
  ss << Conv2DBpFilterTilingParam::ToString() << Conv3DCommonTilingParam::ToString();

  return ss.str();
}

bool Conv3DBpFilterTilingParam::CheckGEMMLimit() const {
  if (type == kConv3DBackpropFilterV2) {
    return true;
  }

  int64_t total_k = MathUtil::Align(a_shape.h * a_shape.w, kBlockSize);
  int64_t total_n = MathUtil::Align(b_shape.c1, kBlockSize) * kernel_h * kernel_w * kernel_d;
  OP_LOGE_IF(total_k > std::numeric_limits<int32_t>::max(), false, op_type,
             "get matrix multiply K (Hout * Wout) axis[%ld] exceed int32 limit.", total_k);
  OP_LOGE_IF(total_n > std::numeric_limits<int32_t>::max(), false, op_type,
             "get matrix multiply N (Cin * Kernel_h * Kernel_w * Kernel_d) axis[%ld] exceed int32 limit.", total_n);
  return true;
}

bool Conv3DBpFilterTilingParam::IsExtraInfoValid() const {
  OP_LOGE_IF(load3d_special == kLoad3dSpecial, false, op_type, "not support load3d special yet");

  // The im2col of the depth for 3d's ops is from ddr to l1 with dma_copy, it has no limitations
  OP_LOGE_IF(!MathUtil::CheckRange(pad_f, kPadMin, kAttrDUpper), false, op_type,
    "pad_f is invalid, current is %d, pad_f support range [%d, %d]", pad_f, kPadMin, kAttrDUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(pad_b, kPadMin, kAttrDUpper), false, op_type,
    "pad_b is invalid, current is %d, pad_b support range [%d, %d]", pad_b, kPadMin, kAttrDUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(stride_d, kStrideMin, kAttrDUpper), false, op_type,
    "stride_d is invalid, current is %d, stride_d support range [%d, %d]", stride_d, kStrideMin, kAttrDUpper);
  OP_LOGE_IF(!MathUtil::CheckRange(dilation_d, kDilationMin, kAttrDUpper), false, op_type,
    "dilation_d is invalid, current is %d, dilation_d support range [%d, %d]", dilation_d, kDilationMin, kAttrDUpper);

  int32_t kernel_d_dilation_temp = (c_shape.d - 1) * dilation_d + 1;
  OP_LOGE_IF(kernel_d_dilation_temp > b_shape.d + pad_f + pad_b, false, op_type,
              "di + pad_front + pad_back = [%ld] should greater than (kd - 1) * dilation_d + 1 = [%d]",
              b_shape.d + pad_f + pad_b, kernel_d_dilation_temp);
  OP_LOGE_IF((b_shape.d - kernel_d_dilation_temp + pad_f + pad_b) / stride_d + 1 != a_shape.d, false, op_type,
             "dout is not match with di.");

  OP_LOGE_IF(split_w_flag, false, op_type, 
    "L1 min tiling exceed L1 size, current out_backprop width is %ld, please change the width to a smaller value.", a_shape.w);
  if (dma_flag) {
    std::stringstream ss;
    ss << "kernel/stride/dilation/pad H/W dimension exceed load3d instruction support range, "
       << "or minimum load size may exceed L1 buffer size " << platform_info.l1_size() << "B";
    OP_LOGE(op_type, "Error msg: %s", ss.str().c_str());
    ss.str("");
    ss << "pad up/down/left/right current is " << pad_u << " " << pad_d << " " << pad_l << " " << pad_r
       << " stride h/w current is " << stride_h << " " << stride_w << " kernel h/w current is " << kernel_h << " "
       << kernel_w << " dilation h/w current is " << dilation_h << " " << dilation_w;
    OP_LOGE(op_type, "Error msg: %s", ss.str().c_str());
    Load3dInstrictionParam &load3d_inst_param = InstructionParam::Instance().get_load3d_inst_param();
    OP_LOGE(op_type, "Error msg: %s", load3d_inst_param.ToString().c_str());
    return false;
  }
  // kd can be 1 when min split in L1
  return Conv2DBpFilterTilingParam::IsExtraInfoValid();
}
}  // namespace cachetiling
}  // namespace optiling