/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "cube_util.h"
#include <algorithm>
#include "error_util.h"
#include "op_log.h"
#include "graph/utils/type_utils.h"

using namespace std;

namespace {
constexpr size_t kConv2dInputSizeLimit = 4;
constexpr size_t kDeconvolutionStridesLimit = 2;
constexpr size_t kConv3dInputSizeLimit = 5;
constexpr size_t kConv3dPadsSizeLimit = 6;


constexpr size_t kConv3DStridesIdx = 0;
constexpr size_t kConv3DDilationsIdx = 2;

constexpr size_t kConv2DStridesIdx = 0;
constexpr size_t kConv2DDilationsIdx = 2;
}

namespace ops {
using gert::InferShapeContext;
using ge::Format;
using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NDHWC;
using ge::FORMAT_DHWCN;

bool GetConv3DXShape(InferShapeContext *context, size_t x_idx, Format x_format, bool avg_pool3d,
                     Conv3DInputShapes &shapes) {
  const auto x_shape = context->GetInputShape(x_idx);
  OP_LOGE_IF(x_shape == nullptr, false, context->GetNodeName(), "failed to get x shape.");
  OP_LOGE_IF(x_shape->GetDimNum() != kConv3dInputSizeLimit, false, context->GetNodeName(),
             "The shape of x_tensor should be 5d, actual dim num: %zu", x_shape->GetDimNum());

  size_t idx = 0;
  if (x_format == FORMAT_NCDHW) {
    shapes.in = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
    shapes.id = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
  } else if (x_format == FORMAT_NDHWC) {
    shapes.in = x_shape->GetDim(idx++);
    shapes.id = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
  } else if (avg_pool3d && x_format == FORMAT_DHWCN) {
    shapes.id = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
    shapes.in = x_shape->GetDim(idx++);
  } else {
    OP_LOGE(context->GetNodeName(), "The format of input x not support format %s.",
            ge::TypeUtils::FormatToAscendString(x_format).GetString());
    return false;
  }

  return true;
}

bool GetConv3DFilterShape(InferShapeContext *context, size_t filter_idx, Conv3DInputShapes &shapes) {
  const auto filter_desc = context->GetInputDesc(filter_idx);
  OP_LOGE_IF(filter_desc == nullptr, false, context->GetNodeName(), "failed to get filter tensor desc.");
  const auto filter_format = filter_desc->GetOriginFormat();
  const auto filter_shape = context->GetInputShape(filter_idx);
  OP_LOGE_IF(filter_shape == nullptr, false, context->GetNodeName(), "failed to get filter shape.");
    // already checked in shape range infer logic, no need to use error manager here
  OP_LOGE_IF(filter_shape->GetDimNum() != kConv3dInputSizeLimit, false, context->GetNodeName(),
             "The shape of the filter should be 5d, actual dim num: %zu", filter_shape->GetDimNum());

  size_t idx = 0;
  if (filter_format == FORMAT_NCDHW) {
    shapes.kn = filter_shape->GetDim(idx++);
    shapes.kc = filter_shape->GetDim(idx++);
    shapes.kd = filter_shape->GetDim(idx++);
    shapes.kh = filter_shape->GetDim(idx++);
    shapes.kw = filter_shape->GetDim(idx++);
  } else if (filter_format == FORMAT_NDHWC) {
    shapes.kn = filter_shape->GetDim(idx++);
    shapes.kd = filter_shape->GetDim(idx++);
    shapes.kh = filter_shape->GetDim(idx++);
    shapes.kw = filter_shape->GetDim(idx++);
    shapes.kc = filter_shape->GetDim(idx++);
  } else if (filter_format == FORMAT_DHWCN) {
    shapes.kd = filter_shape->GetDim(idx++);
    shapes.kh = filter_shape->GetDim(idx++);
    shapes.kw = filter_shape->GetDim(idx++);
    shapes.kc = filter_shape->GetDim(idx++);
    shapes.kn = filter_shape->GetDim(idx++);
  } else {
    OP_LOGE(context->GetNodeName(), "The format of input filter not support format %s.",
            ge::TypeUtils::FormatToAscendString(filter_format).GetString());
    return false;
  }

  return true;
}

bool GetConv3DStridesAndDilations(const InferShapeContext *context, Format x_format, Conv3DAttrs &attrs) {
  const auto runtime_attrs = context->GetAttrs();
  OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
  const auto strides_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv3DStridesIdx);
  OP_LOGE_IF(strides_list == nullptr, false, context->GetNodeName(), "failed to get strides attrs");
  // already checked in first infer shape logic, double check just for security
  OP_LOGE_IF(strides_list->GetSize() != kConv3dInputSizeLimit, false, context->GetNodeName(), "invalid strides");

  const auto dilations_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv3DDilationsIdx);
  OP_LOGE_IF(dilations_list == nullptr, false, context->GetNodeName(), "failed to get dilations attrs");
  OP_LOGE_IF(dilations_list->GetSize() != kConv3dInputSizeLimit, false, context->GetNodeName(), "invalid dilations");

  const auto strides = reinterpret_cast<const int64_t *>(strides_list->GetData());
  const auto dilations = reinterpret_cast<const int64_t *>(dilations_list->GetData());
  if (x_format == ge::FORMAT_NCDHW) {
    attrs.strd = strides[kDDimNCDHWIdx];
    attrs.strh = strides[kHDimNCDHWIdx];
    attrs.strw = strides[kWDimNCDHWIdx];
    attrs.dild = dilations[kDDimNCDHWIdx];
    attrs.dilh = dilations[kHDimNCDHWIdx];
    attrs.dilw = dilations[kWDimNCDHWIdx];
  } else {
    // FORMAT_NDHWC, already checked in GetConv3DXShape, else is enough
    attrs.strd = strides[kDDimNDHWCIdx];
    attrs.strh = strides[kHDimNDHWCIdx];
    attrs.strw = strides[kWDimNDHWCIdx];
    attrs.dild = dilations[kDDimNDHWCIdx];
    attrs.dilh = dilations[kHDimNDHWCIdx];
    attrs.dilw = dilations[kWDimNDHWCIdx];
  }

  OP_LOGE_IF(attrs.strd == 0 || attrs.strh == 0 || attrs.strw == 0, false, context->GetNodeName(), "get zero strides");
  return true;
}

bool GetConv3DPads(const InferShapeContext *context, const Conv3DInputShapes &shapes, size_t pads_idx,
                   size_t padding_idx, Conv3DAttrs &attrs) {
  const auto runtime_attrs = context->GetAttrs();
  OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
  const auto pads_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(pads_idx);
  OP_LOGE_IF(pads_list == nullptr, false, context->GetNodeName(), "failed to get pads attrs");
  OP_LOGE_IF(pads_list->GetSize() != kConv3dPadsSizeLimit, false, context->GetNodeName(), "invalid pads");
  const auto pads_list_data = reinterpret_cast<const int64_t *>(pads_list->GetData());

  size_t idx = 0;
  attrs.padf = pads_list_data[idx++];
  attrs.padb = pads_list_data[idx++];
  attrs.padu = pads_list_data[idx++];
  attrs.padd = pads_list_data[idx++];
  attrs.padl = pads_list_data[idx++];
  attrs.padr = pads_list_data[idx++];

  if (runtime_attrs->GetAttrNum() > padding_idx) {
    const auto padding = runtime_attrs->GetAttrPointer<char>(padding_idx);
    if (padding != nullptr && (strcmp(padding, "SAME") == 0)) {
      OP_LOGD(context->GetNodeName(), "get padding SAME");
      int64_t tails_d = shapes.id % attrs.strd;  // non zero, checked in shape range infer logic
      int64_t tails_h = shapes.ih % attrs.strh;  // non zero, checked in shape range infer logic
      int64_t tails_w = shapes.iw % attrs.strw;  // non zero, checked in shape range infer logic
      int64_t dilate_kernel_d = attrs.dild * (shapes.kd - 1) + 1;
      int64_t dilate_kernel_h = attrs.dilh * (shapes.kh - 1) + 1;
      int64_t dilate_kernel_w = attrs.dilw * (shapes.kw - 1) + 1;
      int64_t pad_d = std::max((tails_d > 0 ? dilate_kernel_d - tails_d : dilate_kernel_d - attrs.strd), 0L);
      int64_t pad_h = std::max((tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - attrs.strh), 0L);
      int64_t pad_w = std::max((tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - attrs.strw), 0L);
      attrs.padf = pad_d / 2;  // 2 means pad_up is half size of pad_d
      attrs.padb = pad_d - attrs.padf;
      attrs.padu = pad_h / 2;  // 2 means pad_up is half size of pad_h
      attrs.padd = pad_h - attrs.padu;
      attrs.padl = pad_w / 2;  // 2 means pad_up is half size of pad_w
      attrs.padr = pad_w - attrs.padl;
      return true;
    }
  }

  bool negative_pad =
      std::any_of(pads_list_data, pads_list_data + pads_list->GetSize(), [](int64_t val) -> bool { return val < 0; });
  OP_LOGE_IF(negative_pad, false, context->GetNodeName(), "The value of the pads attribute should >= 0");

  return true;
}

bool GetConvGroups(const gert::InferShapeContext *context, size_t groups_idx, Conv3DAttrs &attrs) {
  const auto runtime_attrs = context->GetAttrs();
  OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
  const auto groups = runtime_attrs->GetAttrPointer<int64_t>(groups_idx);
  OP_LOGE_IF(groups == nullptr, false, context->GetNodeName(), "failed to get groups attrs");
  attrs.groups = *groups;
  return true;
}

bool GetConv2DXShape(InferShapeContext *context, size_t x_idx, ge::Format x_format, Conv2DInputShapes &shapes) {
  const auto x_shape = context->GetInputShape(x_idx);
  OP_LOGE_IF(x_shape == nullptr, false, context->GetNodeName(), "failed to get x shape.");
  // already checked in shape range infer logic, no need to use error manager here
  OP_LOGE_IF(x_shape->GetDimNum() != kConv2dInputSizeLimit, false, context->GetNodeName(),
             "The shape of x_tensor should be 4d, actual dim num: %zu", x_shape->GetDimNum());
  size_t idx = 0;
  if (x_format == ge::FORMAT_NCHW) {
    shapes.in = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
  } else if (x_format == ge::FORMAT_NHWC) {
    shapes.in = x_shape->GetDim(idx++);
    shapes.ih = x_shape->GetDim(idx++);
    shapes.iw = x_shape->GetDim(idx++);
    shapes.ic = x_shape->GetDim(idx++);
  } else {
    OP_LOGE(context->GetNodeName(), "The format of input x only support NHWC or NCHW, not support format %s.",
            ge::TypeUtils::FormatToAscendString(x_format).GetString());
    return false;
  }

  return true;
}

bool GetConv2DFilterShape(InferShapeContext *context, size_t filter_idx, Conv2DInputShapes &shapes) {
  const auto filter_desc = context->GetInputDesc(filter_idx);
  OP_LOGE_IF(filter_desc == nullptr, false, context->GetNodeName(), "failed to get filter tensor desc.");
  const auto filter_format = filter_desc->GetOriginFormat();
  const auto filter_shape = context->GetInputShape(filter_idx);
  OP_LOGE_IF(filter_shape == nullptr, false, context->GetNodeName(), "failed to get filter shape.");
  // already checked in shape range infer logic, no need to use error manager here
  OP_LOGE_IF(filter_shape->GetDimNum() != kConv2dInputSizeLimit, false, context->GetNodeName(),
             "The shape of the filter should be 4d, actual dim num: %zu", filter_shape->GetDimNum());

  size_t idx = 0;
  if (filter_format == ge::FORMAT_NCHW) {
    shapes.kn = filter_shape->GetDim(idx++);
    shapes.kc = filter_shape->GetDim(idx++);
    shapes.kh = filter_shape->GetDim(idx++);
    shapes.kw = filter_shape->GetDim(idx++);
  } else if (filter_format == ge::FORMAT_NHWC) {
    shapes.kn = filter_shape->GetDim(idx++);
    shapes.kh = filter_shape->GetDim(idx++);
    shapes.kw = filter_shape->GetDim(idx++);
    shapes.kc = filter_shape->GetDim(idx++);
  } else {
    OP_LOGE(context->GetNodeName(), "The format of input filter only support NHWC or NCHW, not support format %s.",
            ge::TypeUtils::FormatToAscendString(filter_format).GetString());
    return false;
  }

  return true;
}

bool GetConv2DStridesAndDilations(const InferShapeContext *context, ge::Format x_format, Conv2DAttrs &attrs,
                                  bool is_deconvolution) {
  const auto runtime_attrs = context->GetAttrs();
  OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
  const auto strides_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv2DStridesIdx);
  OP_LOGE_IF(strides_list == nullptr, false, context->GetNodeName(), "failed to get strides attrs");
  // already checked in first infer shape logic, double check just for security

  if (is_deconvolution) {
    OP_LOGE_IF(strides_list->GetSize() != kDeconvolutionStridesLimit, false, context->GetNodeName(),
               "The strides should be 2d, actual dim num: %zu", strides_list->GetSize());
  } else {
    OP_LOGE_IF(strides_list->GetSize() != kConv2dInputSizeLimit, false, context->GetNodeName(),
               "The strides should be 4d, actual dim num: %zu", strides_list->GetSize());
  }

  const auto dilations_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv2DDilationsIdx);
  OP_LOGE_IF(dilations_list == nullptr, false, context->GetNodeName(), "failed to get dilations attrs");
  OP_LOGE_IF(dilations_list->GetSize() != kConv2dInputSizeLimit, false, context->GetNodeName(),
             "The dilations should be 4d, actual dim num: %zu", dilations_list->GetSize());
  const auto strides = reinterpret_cast<const int64_t *>(strides_list->GetData());
  const auto dilations = reinterpret_cast<const int64_t *>(dilations_list->GetData());

  if (is_deconvolution) {
    if (x_format == ge::FORMAT_NCHW) {
      attrs.str_h = strides[kHDimNCHWStridesIdx];
      attrs.str_w = strides[kWDimNCHWStridesIdx];
      attrs.dil_h = dilations[kHDimNCHWIdx];
      attrs.dil_w = dilations[kWDimNCHWIdx];
    }
  } else if (x_format == ge::FORMAT_NCHW) {
    attrs.str_h = strides[kHDimNCHWIdx];
    attrs.str_w = strides[kWDimNCHWIdx];
    attrs.dil_h = dilations[kHDimNCHWIdx];
    attrs.dil_w = dilations[kWDimNCHWIdx];
  } else if (x_format == ge::FORMAT_NHWC) {
    attrs.str_h = strides[kHDimNHWCIdx];
    attrs.str_w = strides[kWDimNHWCIdx];
    attrs.dil_h = dilations[kHDimNHWCIdx];
    attrs.dil_w = dilations[kWDimNHWCIdx];
  } else {
    return false;
  }

  OP_LOGE_IF(attrs.str_h == 0 || attrs.str_w == 0, false, context->GetNodeName(), "get zero strides");
  return true;
}

bool GetConv2DGroups(InferShapeContext *context, size_t groups_idx, Conv2DAttrs &attrs) {
  const gert::RuntimeAttrs *runtime_attrs = context->GetAttrs();
  OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "Get attrs failed.");
  const int64_t *groups_ptr = runtime_attrs->GetAttrPointer<int64_t>(groups_idx);
  OP_LOGE_IF(groups_ptr == nullptr, false, context->GetNodeName(), "Get groups failed.");
  attrs.groups = *groups_ptr;
  return true;
}

bool GetConv2DPads(const InferShapeContext *context, const Conv2DInputShapes &shapes, const vector<size_t> &idx_vec,
                   Conv2DAttrs &attrs) {
  size_t idx = 0;
  size_t pads_idx = idx_vec[idx++];
  size_t padding_idx = idx_vec[idx++];
  size_t autopad_idx = idx_vec[idx++];
  const auto runtime_attrs = context->GetAttrs();
  OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
  const auto pads_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(pads_idx);
  OP_LOGE_IF(pads_list == nullptr, false, context->GetNodeName(), "failed to get pads attrs");
  OP_LOGE_IF(pads_list->GetSize() != kConv2dInputSizeLimit, false, context->GetNodeName(),
             "The pads should be 4d, actual dim num: %zu", pads_list->GetSize());
  const auto pads_list_data = reinterpret_cast<const int64_t *>(pads_list->GetData());

  idx = 0;
  attrs.pad_u = pads_list_data[idx++];
  attrs.pad_d = pads_list_data[idx++];
  attrs.pad_l = pads_list_data[idx++];
  attrs.pad_r = pads_list_data[idx++];

  if (runtime_attrs->GetAttrNum() > padding_idx) {
    const auto padding = runtime_attrs->GetAttrPointer<char>(padding_idx);
    if (padding != nullptr && (strcmp(padding, "SAME") == 0)) {
      OP_LOGD(context->GetNodeName(), "get padding SAME");
      int64_t tails_h = shapes.ih % attrs.str_h;  // non zero, checked in shape range infer logic
      int64_t tails_w = shapes.iw % attrs.str_w;  // non zero, checked in shape range infer logic
      int64_t dilate_kernel_h = attrs.dil_h * (shapes.kh - 1) + 1;
      int64_t dilate_kernel_w = attrs.dil_w * (shapes.kw - 1) + 1;
      int64_t pad_h = std::max((tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - attrs.str_h), 0L);
      int64_t pad_w = std::max((tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - attrs.str_w), 0L);
      attrs.pad_u = pad_h / 2;  // 2 means pad_up is half size of pad_h
      attrs.pad_d = pad_h - attrs.pad_u;
      attrs.pad_l = pad_w / 2;  // 2 means pad_up is half size of pad_w
      attrs.pad_r = pad_w - attrs.pad_l;
      return true;
    }
  }

  // process for onnx
  if (strcmp(context->GetNodeType(), "Conv2DTranspose") == 0) {
    const auto auto_pad = runtime_attrs->GetAttrPointer<char>(autopad_idx);
    if (auto_pad != nullptr && (strcmp(auto_pad, "NOTSET") != 0)) {
      int64_t dilate_kernel_h = attrs.dil_h * (shapes.kh - 1) + 1;
      int64_t dilate_kernel_w = attrs.dil_w * (shapes.kw - 1) + 1;
      int64_t padh = dilate_kernel_h - static_cast<int64_t>(attrs.str_h);
      int64_t padw = dilate_kernel_w - static_cast<int64_t>(attrs.str_w);
      if (padh < 0) {
        padh = 0;
      }
      if (padw < 0) {
        padw = 0;
      }
      if (strcmp(auto_pad, "SAME_UPPER") == 0) {
        int64_t pad_up = padh / 2;
        int64_t pad_left = padw / 2;
        attrs.pad_u = pad_up;
        attrs.pad_d = padh - pad_up;
        attrs.pad_l = pad_left;
        attrs.pad_r = padw - pad_left;
      } else if (strcmp(auto_pad, "SAME_LOWER") == 0) {
        int64_t pad_down = padh / 2;
        int64_t pad_right = padw / 2;
        attrs.pad_u = padh - pad_down;
        attrs.pad_d = pad_down;
        attrs.pad_l = padw - pad_right;
        attrs.pad_r = pad_right;
      }
      return true;
    }
  }

  bool negative_pad =
      std::any_of(pads_list_data, pads_list_data + pads_list->GetSize(), [](int64_t val) -> bool { return val < 0; });
  OP_LOGE_IF(negative_pad, false, context->GetNodeName(), "The value of the pads attribute should >= 0");

  return true;
}
}  // namespace ops
